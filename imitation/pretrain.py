import torch
from torch.utils.data import DataLoader
from imitation.expert_dataset import ExpertDataset
from imitation.behaviour_cloning import BCTrainer
from imitation.action_discretizer import ActionDiscretizer
from enviornments.action_mapper import ActionMapper
from enviornments.observation_builder import ObservationBuilder
from agents.q_network import QNetwork

DEVICE = "cpu"
DATASET_PATH = "logs/drive_log_20260511_132641.json"
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
CHECKPOINT_PATH = "checkpoints/bc_pretrain.pt"

def main():

    observation_builder = ObservationBuilder()
    action_mapper = ActionMapper()
    action_discretizer = ActionDiscretizer(
        action_mapper.action
    )
    
    dataset = ExpertDataset(
        dataset_path = DATASET_PATH,
        observation_builder = observation_builder,
        action_discretizer = action_discretizer
    )

    dataloader = DataLoader(
        dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0
    )

    sample_obs, _ = dataset[0]
    input_size = sample_obs.shape[0]
    num_action = action_mapper.num_actions()

    model = QNetwork(input_size=input_size, num_actions=num_action).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = BCTrainer(
        model=model, dataloader=dataloader, optimizer= optimizer
    )

    for epoch in range(EPOCHS):
        avg_loss, accuracy = (trainer.train())

        print(f"[BC] "
              f"Epoch {epoch+1:03d}"
              f"Loss {avg_loss:.4f}"
              f"Accuracy {accuracy:.4f}")
        
    torch.save(model.state_dict, CHECKPOINT_PATH)

    print()

if __name__ == "__main__":
    main()