import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class BCTrainer:
    """
    Behaviour Cloning: supervised pretraining of the QNetwork
    using cross-entropy loss on expert (state, action) pairs.

    After BC pretraining, copy weights to the DQN agent's online_net
    and target_net before starting RL training.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        device: torch.device,
    ):
        self.model      = model
        self.optimizer  = optimizer
        self.dataloader = dataloader
        self.device     = device

        self.criterization = nn.CrossEntropyLoss()

    def train(self, epochs: int):
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for obs, actions in self.dataloader:

            obs = obs.to(self.device)
            actions = actions.to(self.device)

            logits = self.model(obs)

            loss = self.criterization(logits, actions)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)

            total_correct += (predictions == actions)

            total_samples += actions.size(0)

        avg_loss = total_loss / len(self.dataloader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy