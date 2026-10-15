import torch
from tqdm import tqdm
from .checkpoints import save_checkpoint


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10,
    checkpoint_dir="checkpoints"
):
    """
    Enhanced training loop with checkpoint functionality
    """

    model = model.to(device)
    best_accuracy = 0.0

    for epoch in range(epochs):

        # --------------------
        # Training
        # --------------------
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        train_loader_with_progress = tqdm(
            train_loader,
            ncols=120,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for inputs, labels in train_loader_with_progress:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_total += labels.size(0)
            running_correct += (
                predicted == labels
            ).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = (
            100 * running_correct / running_total
        )

        # --------------------
        # Validation
        # --------------------
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(
                    outputs.data, 1
                )

                val_total += labels.size(0)
                val_correct += (
                    predicted == labels
                ).sum().item()

        val_loss = val_loss / len(val_loader)

        val_accuracy = (
            100 * val_correct / val_total
        )

        # --------------------
        # Save checkpoint
        # --------------------
        checkpoint_path = save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            val_loss,
            val_accuracy,
            checkpoint_dir
        )

        # Save best model
        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_loss,
                val_accuracy,
                checkpoint_dir=f"{checkpoint_dir}/best"
            )

            print(
                f"New best model saved! "
                f"Accuracy: {val_accuracy:.2f}%"
            )

        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss={train_loss:.4f}, "
            f"Train Accuracy={train_accuracy:.2f}%, "
            f"Val Loss={val_loss:.4f}, "
            f"Val Accuracy={val_accuracy:.2f}%"
        )

        print(
            f"Checkpoint saved: {checkpoint_path}"
        )

    return model