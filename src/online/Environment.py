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

        step = torch.tensor(0, dtype=torch.int64)
        prosumption_data, price_data = self._dataset[step]
        prosumption = prosumption_data[0].item()
        prosumption_forecast = prosumption_data[1:]
        price = price_data[0].item()
        price_forecast = price_data[1:]

        td_out = TensorDict(
            {
                'step': step,
                'soe': torch.tensor(0.0),
                'prosumption': prosumption,
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
        action = td_in['action']
        step = td_in['step']
        old_soe = td_in['soe']
        old_cost = td_in['cost']
        params = td_in['params']

        prosumption_data, price_data = self._dataset[step]
        prosumption = prosumption_data[0]
        prosumption_forecast = prosumption_data[1:]
        price = price_data[0]
        price_forecast = price_data[1:]

        new_soe = torch.clip(old_soe + action, 0.0, params['battery_capacity'])
        clipped_action = old_soe - new_soe
        penalty_soe  = torch.abs(action - clipped_action)

        grid = prosumption - clipped_action
        grid_buy = grid if grid > 0.0 else 0.0
        grid_sell = torch.abs(grid) if grid <= 0.0 else 0.0
        
        cost =  (grid_sell-grid_buy)*price
        new_cost = old_cost + cost
        reward = cost - penalty_soe

        td_out = TensorDict(
            {
                'step': step + 1,
                'soe': new_soe,
                'prosumption': prosumption,
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
            step=Bounded(low = 0,
                         high = td_param['params', 'max_steps'],
                         shape=(),
                         dtype=torch.int64),
            soe=Bounded(low = 0,
                         high = td_param['params', 'battery_capacity'],
                         shape=(),
                         dtype=torch.float32),
            prosumption=Unbounded(dtype=torch.float32, 
                                  shape=()),
            price=Unbounded(dtype=torch.float32,
                            shape=()),
            price_forecast=Unbounded(dtype=torch.float32, 
                                     shape=(48,)),
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
    

