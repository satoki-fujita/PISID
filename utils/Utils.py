import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class DataInput(object):
    def __init__(self, data_dir: str, use_data='JP'):
        self.data_dir = data_dir
        self.use_data = use_data

    def load_data(self):
        if self.use_data=='JP':
            df_N = pd.DataFrame({'region':['Hokkaido','Aomori','Iwate','Miyagi', 'Akita','Yamagata','Fukushima','Ibaraki','Tochigi','Gunma','Saitama','Chiba','Tokyo','Kanagawa','Niigata','Toyama','Ishikawa','Fukui','Yamanashi','Nagano','Gifu','Shizuoka','Aichi','Mie','Shiga','Kyoto','Osaka','Hyogo','Nara','Wakayama','Tottori','Shimane','Okayama','Hiroshima','Yamaguchi','Tokushima','Kagawa','Ehime','Kochi','Fukuoka','Saga','Nagasaki','Kumamoto','Oita','Miyazaki','Kagoshima','Okinawa'],
                                 'population':[5248552.0, 1246138.0, 1226430.0, 2303160.0, 965968.0, 1077057.0, 1847950.0, 2868041.0, 1942312.0, 1937626.0, 7337330.0, 6279026.0, 13942856.0, 9200166.0, 2222004.0, 1042998.0, 1137181.0, 767742.0, 812056.0, 2049023.0, 1988931.0, 3639226.0, 7552873.0, 1779770.0, 1413959.0, 2583140.0, 8823453.0, 5463609.0, 1331330.0, 923721.0, 555663.0, 673891.0, 1891346.0, 2807987.0, 1355495.0, 728633.0, 956069.0, 1338811.0, 697674.0, 5110113.0, 814211.0, 1325205.0, 1746740.0, 1134431.0, 1072077.0, 1599984.0, 1454184.0]})
            df_cases = pd.read_excel(self.data_dir + '/newly_confirmed_cases_daily.xlsx').rename(columns={'Date':'date'}).drop(['ALL'], axis=1).set_index('date')
            df_cases.index = pd.to_datetime(df_cases.index)
            df_cases = df_cases.rolling(7).mean().dropna(how='all')
            df_cases_list = [df_cases.loc[df_cases.index<=pd.to_datetime('2021-12-31'),:].copy(), df_cases.loc[df_cases.index>pd.to_datetime('2021-12-31'),:].copy()]
            
        elif self.use_data=='US':
            df_N = pd.DataFrame({'region':['Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware','District of Columbia','Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri','Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York','North Carolina','North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont','Virginia','Washington','West Virginia','Wisconsin','Wyoming'],
                                 'population':[4903185.0, 731545.0, 7278717.0, 3017804.0, 39512223.0, 5758736.0, 3565287.0, 973764.0, 705749.0, 21477737.0, 10617423.0, 1415872.0, 1787065.0, 12671821.0, 6732219.0, 3155070.0, 2913314.0, 4467673.0, 4648794.0, 1344212.0, 6045680.0, 6892503.0, 9986857.0, 5639632.0, 2976149.0, 6137428.0, 1068778.0, 1934408.0, 3080156.0, 1359711.0, 8882190.0, 2096829.0, 19453561.0, 10488084.0, 762062.0, 11689100.0, 3956971.0, 4217737.0, 12801989.0, 1059361.0, 5148714.0, 884659.0, 6829174.0, 28995881.0, 3205958.0, 623989.0, 8535519.0, 7614893.0, 1792147.0, 5822434.0, 578759.0]})
            df_cases = pd.read_csv(self.data_dir + '/time_series_covid19_confirmed_US.csv').rename(columns={'Province_State':'region'})
            df_cases = df_cases.loc[df_cases['region'].isin(df_N['region']),:].drop(['UID','iso2','iso3','code3','FIPS','Admin2','Country_Region','Lat','Long_','Combined_Key'],axis=1).groupby('region').sum().transpose()
            df_cases.index = pd.to_datetime(df_cases.index, format='%m/%d/%y'); df_cases.index.name = 'date'
            df_cases = df_cases.sort_index().diff().dropna(how='all')
            for state_i in df_N['region']:
                df_cases.loc[df_cases[state_i]<0, [state_i]] = 0.0
            df_cases = df_cases.rolling(7).mean().dropna(how='all')
            df_cases_list = [df_cases.loc[df_cases.index<=pd.to_datetime('2021-11-30'),:].copy(), df_cases.loc[df_cases.index>pd.to_datetime('2021-11-30'),:].copy()]

        dataset_all = dict()
        for df_ in df_cases_list:
            dataset = dict()
            label = df_.index[0].date().strftime('%Y/%m/%d') + '~' + df_.index[-1].date().strftime('%Y/%m/%d')
            df_ = pd.merge(pd.merge(df_.stack(dropna=False).reset_index().rename(columns={'level_1':'region',0:'new'}),
                                    df_.cumsum().stack(dropna=False).reset_index().rename(columns={'level_1':'region',0:'cumulative'}), how='left'),
                                    df_N, how='left')
            df_['not_confirmed'] = df_['population']-df_['cumulative']
            dataset['node'] = np.log(df_[['new']].to_numpy().reshape([-1,df_N.shape[0],1])+1.0)
            dataset['state'] = df_[['not_confirmed','new','population']].to_numpy().reshape([-1,df_N.shape[0],3])
            dataset['y'] = df_[['new']].to_numpy().reshape([-1,df_N.shape[0],1])
            dataset_all[label] = dataset
        
        return dataset_all


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class ODDataset(Dataset):
    def __init__(self, inputs: dict, output: dict, mode: str, mode_len: dict):
        self.mode = mode
        self.mode_len = mode_len
        self.inputs, self.output = self.prepare_xy(inputs, output)

    def __len__(self):
        return self.mode_len[self.mode]

    def __getitem__(self, item):
        return self.inputs['x_node'][item], self.inputs['x_state'][item], self.output['y'][item]

    def prepare_xy(self, inputs: dict, output: dict):
        if self.mode == 'train':
            start_idx = 0
        elif self.mode == 'validate':
            start_idx = self.mode_len['train']
        else:
            start_idx = self.mode_len['train'] + self.mode_len['validate']

        x = dict()
        x['x_node'] = inputs['x_node'][start_idx: (start_idx + self.mode_len[self.mode])]
        x['x_state'] = inputs['x_state'][start_idx: (start_idx + self.mode_len[self.mode])]

        y = dict()
        y['y'] = output['y'][start_idx: start_idx + self.mode_len[self.mode]]
        return x, y

class DataGenerator(object):
    def __init__(self, obs_len: int, pred_len, data_split_ratio: tuple):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.data_split_ratio = data_split_ratio

    def split2len(self, data_len: int):
        mode_len = dict()
        mode_len['train'] = int(self.data_split_ratio[0] / sum(self.data_split_ratio) * data_len)
        mode_len['validate'] = int(self.data_split_ratio[1] / sum(self.data_split_ratio) * data_len)
        mode_len['test'] = data_len - mode_len['train'] - mode_len['validate']
        return mode_len

    def get_data_loader(self, data: dict, params: dict):
        x_node, x_state, y = self.get_feats(data)
        x_node = np.asarray(x_node)
        x_state = np.asarray(x_state)
        y = np.asarray(y)

        mode_len = self.split2len(data_len=y.shape[0])

        scaler_dic={}
        for i in range(x_node.shape[-1]):
            scaler_dic[i] = StandardScaler(mean=x_node[:mode_len['train'],..., i].mean(),
                                           std=x_node[:mode_len['train'],..., i].std())
            x_node[...,i] = scaler_dic[i].transform(x_node[...,i])

        feat_dict = dict()
        feat_dict['x_node'] = torch.from_numpy(x_node).float().to(params['GPU'])
        feat_dict['x_state'] = torch.from_numpy(x_state).float().to(params['GPU'])
        output_dict = dict()
        output_dict['y'] = torch.from_numpy(y).float().to(params['GPU'])

        print('Data split:', mode_len)

        data_loader = dict()
        for mode in ['train', 'validate', 'test']:
            dataset = ODDataset(inputs=feat_dict, output=output_dict, mode=mode, mode_len=mode_len)
            print('Data loader', '|', mode, '|', 'input node features:', dataset.inputs['x_node'].shape, '|'
                'output:', dataset.output['y'].shape)
            if mode == 'train':
                data_loader[mode] = DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=True)
            else:
                data_loader[mode] = DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=False)
        return data_loader, scaler_dic

    def get_feats(self, data: dict):
        x_node, x_state, y = [], [], []
        for i in range(self.obs_len, data['y'].shape[0] - self.pred_len + 1):
            x_node.append(data['node'][i - self.obs_len: i])
            x_state.append(data['state'][i - self.obs_len: i])
            y.append(data['y'][i: i + self.pred_len])
        return x_node, x_state, y


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def seed_torch(RANDOM_SEED=3):
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True