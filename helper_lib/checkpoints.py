import torch
import os


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    accuracy,
    checkpoint_dir="checkpoints"
):
    """
    Save model checkpoint.

    TODO: Implement checkpoint saving that includes:
    1. Model state dict
    2. Optimizer state dict
    3. Epoch number
    4. Loss and accuracy metrics
    5. Create checkpoint directory if needed

    Hint: Reference the FCNN notebook checkpoint implementation.
    """

    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "accuracy": accuracy
    }

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"model_epoch_{epoch:03d}.pth"
    )

    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Load model checkpoint and restore training state.

    TODO: Implement checkpoint loading that:
    1. Loads the checkpoint file
    2. Restores model and optimizer states
    3. Returns epoch, loss, and accuracy information

    Why save optimizer state? See FCNN notebook documentation!
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    accuracy = checkpoint["accuracy"]

    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

    return epoch, loss, accuracy