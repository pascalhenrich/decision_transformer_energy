import hydra
from hydra.core.config_store import ConfigStore
from conf.config import HydraConfig
import torch


cs = ConfigStore.instance()
cs.store(name="hydra_config", node=HydraConfig)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: HydraConfig):
    # Device Setup
    device = cfg.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('Warning: CUDA is not supported on this system. Falling back to CPU!')
        device = 'cpu'
    DEVICE = torch.device(device)
    print('It is working!')

if __name__ == "__main__":
    main()