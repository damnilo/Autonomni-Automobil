import json

import numpy as np
import torch
from torch.utils.data import Dataset


class DrivingDataset(Dataset):
    """
    Loads a JSON dataset of expert demonstrations.

    Expected JSON format:
        [
            { "observation": [float, ...], "action": int },
            ...
        ]

    Returns (state_tensor, action_tensor) pairs.
    """

    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        print(f"[Dataset] Loaded {len(self.data)} transitions from {json_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        item   = self.data[idx]
        state  = torch.tensor(item["observation"], dtype=torch.float32)
        action = torch.tensor(item["action"],      dtype=torch.long)
        return state, action