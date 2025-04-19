from dataclasses import dataclass
from typing import List

@dataclass
class HydraConfig:
    name: str
    output_path: str
    model_path: str
    device: str