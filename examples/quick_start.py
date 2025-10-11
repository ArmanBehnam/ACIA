from acia import ColoredMNIST, CausalRepresentationNetwork, ctrain_model
from torch.utils.data import DataLoader, ConcatDataset
import torch

# Load data
train_e1 = ColoredMNIST('e1', train=True, intervention_type='perfect')
train_e2 = ColoredMNIST('e2', train=True, intervention_type='perfect')
train_loader = DataLoader(ConcatDataset([train_e1, train_e2]), batch_size=32, shuffle=True)

test_e1 = ColoredMNIST('e1', train=False, intervention_type='perfect')
test_e2 = ColoredMNIST('e2', train=False, intervention_type='perfect')
test_loader = DataLoader(ConcatDataset([test_e1, test_e2]), batch_size=32)

# Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CausalRepresentationNetwork().to(device)
history = ctrain_model(train_loader, test_loader, model, n_epochs=3)

print(f"Final accuracy: {history['test_acc'][-1]:.3f}")