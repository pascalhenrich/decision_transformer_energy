import torch
from torch.utils.data import Dataset
import pandas as pd

class EnergyDataset(Dataset):
    def __init__(self, energy_path: str, price_path: str, forecast_size: int, customer: int, mode: str):
        csv_data = pd.read_csv(energy_path, header=0)
        csv_data['Price'] = pd.read_csv(price_path, header=0)['Price']
        csv_data.fillna(0, inplace=True)
        if mode == 'train':
            csv_data = csv_data.loc[0:17519].reset_index(drop=True)
        elif mode == 'eval':
            csv_data = (csv_data.loc[17520:35039]).reset_index(drop=True)
        elif mode == 'test':
            csv_data = (csv_data.loc[35040:52559]).reset_index(drop=True)
        self._data = pd.DataFrame({'prosumption': csv_data[f'prosumption_{customer}'],
                                   'load': csv_data[f'load_{customer}'],
                                   'pv': csv_data[f'pv_{customer}'],
                                   'price': csv_data['Price']})
        self._forecast_size = forecast_size
        self._batteryCapacity = self._calcBatteryCapacity(csv_data.loc[0:17519].reset_index(drop=True), customer)
    
    def __len__(self):
        return len(self._data)-self._forecast_size
    
    def __getitem__(self, idx):
        net_load = torch.tensor(self._data['prosumption'].loc[idx:idx+self._forecast_size].values, dtype=torch.float32)
        price = torch.tensor(self._data['price'].loc[idx:idx+self._forecast_size].values, dtype=torch.float32)
        return net_load, price
    
    def _calcBatteryCapacity(self, csv_data_train, customer):
        prosumption = torch.tensor(csv_data_train[f'prosumption_{customer}'].values)
        daily_values = prosumption.view(365, 48)
        daily_negative_sums = daily_values.where(daily_values < 0, torch.zeros_like(daily_values)).sum(dim=1)
        return torch.ceil(torch.abs(daily_negative_sums.mean()))
        
    def getBatteryCapacity(self):
        return self._batteryCapacity.float()