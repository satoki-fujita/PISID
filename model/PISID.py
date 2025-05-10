import torch
from torch import nn

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim, dropout) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        hidden = hidden + input_data
        return hidden


class PISID(nn.Module):
    def __init__(self, num_nodes, dropout=0.15, in_dim=1, in_len=28, out_len=28, layers=3, node_dim=32, wave_dim=32, embed_dim=32, if_node=True):
        super(PISID, self).__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.wave_dim = wave_dim
        self.input_len = in_len
        self.input_dim = in_dim
        self.embed_dim = embed_dim
        self.output_len = out_len
        self.num_layer = layers
        self.dropout = dropout
        self.if_spatial = if_node

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)

        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        
        self.hidden_dim = self.embed_dim + self.node_dim*int(self.if_spatial)                       
        self.encoder = nn.Sequential(*[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim, self.dropout) for _ in range(self.num_layer)])
        
        # regression
        self.regression_layer_b = nn.Conv2d(in_channels=self.hidden_dim, out_channels=1, kernel_size=(1, 1), bias=True)              
        self.regression_layer_g = nn.Conv2d(in_channels=self.hidden_dim, out_channels=1, kernel_size=(1, 1), bias=True)

    def forward(self, x_node, x_state, params):
        # x_node : [normalized new confirmed cases], x_state : [N-cumulative confirmed cases, new confirmed cases, N]
        input_data = x_node[..., range(self.input_dim)]       
        batch_size, _, num_nodes, _ = input_data.shape      
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1)
        
        node_emb = []
        if self.if_spatial:
            node_emb.append(self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        time_series_emb = self.time_series_emb_layer(input_data.transpose(1, 2).unsqueeze(-1))
        hidden = torch.cat([time_series_emb] + node_emb, dim=1)
        hidden = self.encoder(hidden)
        param_b = torch.sigmoid(self.regression_layer_b(hidden))
        param_g = torch.sigmoid(self.regression_layer_g(hidden))      
        
        # SIR
        N = x_state[:,-1,:,[2]]
        I_history = x_state[:,:,:,[1]]/(param_g.repeat(1,self.input_len,1,1))
        Iin_history = I_history[:,1:,:,:]-I_history[:,:-1,:,:]+x_state[:,:-1,:,[1]]
        Iin_history = torch.where(Iin_history < 0, torch.tensor(0.0).to(params['GPU']), Iin_history)
        weight_I = torch.exp(-torch.einsum("bijk,il->bljk", param_g, torch.arange(self.input_len-1,0,-1).unsqueeze(0).to(params['GPU'])))
        I_Tminus1 = torch.einsum("bijk,bijk->bjk", weight_I, Iin_history)
        R_Tminus1 = N-x_state[:,-2,:,[0]]
        S_Tminus1 = N-I_Tminus1-R_Tminus1

        Iin = param_b[:, 0, ...]*S_Tminus1*I_Tminus1/N
        Rin = param_g[:, 0, ...]*I_history[:,-1,:,[0]]
        Iin_history = torch.cat((Iin_history, Iin.unsqueeze(1)), dim=1)
        
        S_T = S_Tminus1-Iin
        weight_I = torch.exp(-torch.einsum("bijk,il->bljk", param_g, torch.arange(self.input_len,0,-1).unsqueeze(0).to(params['GPU'])))
        I_T = torch.einsum("bijk,bijk->bjk", weight_I, Iin_history)
        R_T = R_Tminus1 + Rin
        SIRNIinRin = []
        for i in range(self.output_len):
            Iin = param_b[:, 0, ...]*S_T*I_T/N
            Iin_history = torch.cat((Iin_history, Iin.unsqueeze(1)), dim=1)
            S_T = S_T - Iin
            Rin = param_g[:, 0, ...]*I_T
            weight_I = torch.exp(-torch.einsum("bijk,il->bljk", param_g, torch.arange(self.input_len+i+1,0,-1).unsqueeze(0).to(params['GPU'])))
            I_T = torch.einsum("bijk,bijk->bjk", weight_I, Iin_history)
            R_T = R_T + Rin
            SIRNIinRin.append(torch.cat((S_T, I_T, R_T, N, Iin, Rin), dim=-1).unsqueeze(1))

        SIRNIinRin = torch.cat(SIRNIinRin, dim=1)
        forecast_out = SIRNIinRin[:,:,:,[5]]

        return forecast_out