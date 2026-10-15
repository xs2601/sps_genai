import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model


# Choose device
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print(f"Using device: {device}")


# Load CIFAR-10 training data
full_train_loader = get_data_loader(
    "./data",
    batch_size=32,
    train=True
)

full_train_dataset = full_train_loader.dataset


# Split training data into train + validation
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [45000, 5000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


# Separate CIFAR-10 test set
test_loader = get_data_loader(
    "./data",
    batch_size=32,
    train=False
)


# Create CNN
model = get_model("CNN")

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005
)


# Train using training + validation data
trained_model = train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device=device,
    epochs=10,
    checkpoint_dir="checkpoints_cnn"
)


# Final evaluation on untouched test data
test_loss, test_accuracy = evaluate_model(
    trained_model,
    test_loader,
    criterion,
    device=device
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")