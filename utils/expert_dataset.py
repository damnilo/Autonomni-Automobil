from torch.utils.data import Dataset
import torch
import json
import numpy as np

class ExpertDataset(Dataset):

    def __init__(self, dataset_path, observation_builder, action_discretizer):
        
        with open(dataset_path, "r") as f:

            self.records = json.load(f)

        self.observation_builder = observation_builder
        self.action_discretizer = action_discretizer
    
    def __len__(self):

        return len(self.records)
    
    def __getitem__(self, index):
        
        sample = self.records[index]

        obs = self.observation_builder.build(
            None, sample["observation"], sample["info"]
        )

        steering = sample["action_steering"]
        throttle = sample["action_throttle"]

        action = self.action_discretizer.discretize(
            steering, throttle
        )

        return (torch.tensor(obs, dtype=torch.float32), torch.tensor(action, dtype=torch.long))