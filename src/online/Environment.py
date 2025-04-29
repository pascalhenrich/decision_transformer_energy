from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Unbounded, Composite
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvBase
from typing import Optional
import torch

class BatteryScheduling(EnvBase):
    def __init__(self, cfg, seed, dataset):
        super().__init__(device=cfg.device, batch_size=torch.Size([]))

        # Dataset
        self._dataset = dataset

        # Environment parameters
        td = self._makeParams()

        # Environment specs
        self._makeSpecs(td)

        # Environment seed
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _reset(self, td_in):
        if td_in is None or td_in.is_empty():
           td_in = self._makeParams()
        else:
            td_in['params'] = self._makeParams()['params']

        step = torch.tensor(0, dtype=torch.int64)
        prosumption_data, price_data, time_data = self._dataset[step]
        prosumption = prosumption_data[0].item()
        prosumption_forecast = prosumption_data[1:]
        price = price_data[0].item()
        price_forecast = price_data[1:]
        time_feature = torch.tensor([time_data[0][0], time_data[1][0]])

        td_out = TensorDict(
            {
                'step': step,
                'time_feature': time_feature,
                'soe': torch.tensor(0.0),
                'prosumption': prosumption,
                'prosumption_forecast': prosumption_forecast,
                'price': price,
                'price_forecast': price_forecast,
                'cost': torch.tensor(0.0),
                'params': td_in['params'],
            },
            batch_size=td_in.shape,
            device=td_in.device,
        )
        return td_out


    def _step(self, td_in):
        action = td_in['action'].squeeze().detach()
        step = td_in['step'] + 1
        old_soe = td_in['soe']
        old_cost = td_in['cost']
        params = td_in['params']

        prosumption_data, price_data, time_data = self._dataset[step]
        prosumption = prosumption_data[0]
        prosumption_forecast = prosumption_data[1:]
        price = price_data[0]
        price_forecast = price_data[1:]
        time_feature = torch.tensor([time_data[0][0], time_data[1][0]])

        new_soe = torch.clip(old_soe + action, torch.tensor(0.0), params['battery_capacity'])
        clipped_action = new_soe - old_soe
        penalty_soe  = torch.abs(action - clipped_action)

        grid = prosumption + clipped_action
        
        cost =  grid*price
        new_cost = old_cost + cost
        reward = -cost - penalty_soe

        td_out = TensorDict(
            {
                'step': step,
                'time_feature': time_feature,
                'soe': new_soe,
                'prosumption': prosumption,
                'prosumption_forecast': prosumption_forecast,
                'price': price,
                'price_forecast': price_forecast,
                'cost': new_cost,
                'params': params,
                'reward': reward,
                'done': (step + 1) >= params['max_steps'],
            },
            batch_size=td_in.shape,
            device=td_in.device,
        )

        return td_out
    
    def make_composite_from_td(self, td):
        composite = Composite(
            {
                key: self.make_composite_from_td(tensor)
                if isinstance(tensor, TensorDictBase)
                else Unbounded(dtype=tensor.dtype, device=tensor.device, shape=tensor.shape)
                for key, tensor in td.items()
            },
            shape=td.shape,
        )
        return composite
    
    def _makeSpecs(self, td_param):
        self.observation_spec = Composite(
            step=Bounded(low=0,
                         high=td_param['params', 'max_steps'],
                         shape=(),
                         dtype=torch.int64),
            time_feature=Bounded(low=0,
                            high=1,
                            shape=(2,),
                            dtype=torch.float32),
            soe=Bounded(low = 0,
                         high = td_param['params', 'battery_capacity'],
                         shape=(),
                         dtype=torch.float32),
            prosumption=Unbounded(dtype=torch.float32, 
                                  shape=()),
            prosumption_forecast=Unbounded(dtype=torch.float32, 
                                            shape=(10,)),
            price=Unbounded(dtype=torch.float32,
                            shape=()),
            price_forecast=Unbounded(dtype=torch.float32, 
                                     shape=(10,)),
            cost=Unbounded(dtype=torch.float32, 
                           shape=()),
            params=self.make_composite_from_td(td_param['params']),
            shape=torch.Size([]),
        )

        self.action_spec = Bounded(
            low=-td_param['params', 'max_power']/4,
            high=td_param['params', 'max_power']/4,
            shape=torch.Size([]),
            dtype=torch.float32,
        )
        self.reward_spec = Unbounded(shape=(*td_param.shape, 1), dtype=torch.float32)

    def _makeParams(self):
        td_param = TensorDict(
            {
                'params': TensorDict(
                    {
                        'battery_capacity': self._dataset.getBatteryCapacity(),
                        'max_power': self._dataset.getBatteryCapacity()/2,
                        'max_steps': self._dataset.__len__()
                    },
                    batch_size=torch.Size([])
                )
            },
            batch_size=torch.Size([]),
        )
        return td_param
    

