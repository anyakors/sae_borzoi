import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import io
from torchvision.transforms import ToTensor
from dataset import *
import json
import os


class ModelCheckpoint:
    def __init__(self, save_dir, metric_name="val_total_loss"):
        """
        Handle model checkpointing.

        Args:
            save_dir (str): Directory to save checkpoints
            metric_name (str): Name of metric to monitor for best model
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.best_metric = float("inf")

    def save_checkpoint(self, model, optimizer, epoch, metrics, is_best=False):
        """Save model checkpoint and training state"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        # Save latest checkpoint
        latest_path = self.save_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

        # Save best model if this is the best so far
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

            # Save metrics as JSON for easy reading
            metrics_path = self.save_dir / "best_model_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)

    def is_best(self, metric_value):
        """Check if current metric is the best so far"""
        if metric_value < self.best_metric:
            self.best_metric = metric_value
            return True
        return False

    @staticmethod
    def load_checkpoint(checkpoint_path, model, optimizer=None):
        """Load model and training state from checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint["epoch"], checkpoint["metrics"]


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, k, sparsity_method="topk"):
        """
        Enhanced sparse autoencoder with multiple sparsity options.

        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layer
            k (int): Number of top activations to keep or sparsity target
            sparsity_method (str): 'topk' or 'threshold' or 'smooth_topk'
        """
        super().__init__()
        self.k = k
        self.sparsity_method = sparsity_method

        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim)

        # Decoder
        self.decoder = nn.Linear(hidden_dim, input_dim)

        # Temperature parameter for smooth top-k
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def get_sparse_activations(self, h):
        """Apply sparsity using the selected method"""
        if self.sparsity_method == "topk":
            # Hard top-k selection
            topk_values, _ = torch.topk(h.abs(), k=self.k, dim=1)
            threshold = topk_values[:, -1].unsqueeze(1)
            return h * (h.abs() >= threshold)

        elif self.sparsity_method == "threshold":
            # Adaptive threshold based on activation statistics
            threshold = h.abs().mean(dim=1, keepdim=True) + h.abs().std(
                dim=1, keepdim=True
            )
            return h * (h.abs() >= threshold)

        elif self.sparsity_method == "smooth_topk":
            # Smooth top-k using softmax
            scores = h.abs() / self.temperature
            mask = torch.softmax(scores, dim=1)
            mask = mask >= torch.topk(mask, k=self.k, dim=1)[0][:, -1:]
            return h * mask

    def get_metrics(self, h_sparse):
        """Calculate additional sparsity metrics"""
        batch_size = h_sparse.size(0)

        # Sparsity ratio (% of zero activations)
        sparsity_ratio = (h_sparse == 0).float().mean(dim=1)

        # Active neuron diversity (how many different neurons are active across batch)
        active_neurons = (h_sparse != 0).sum(dim=0)
        # Now we convert to float before division to avoid the .item() issue
        neuron_diversity = float(active_neurons.nonzero().size(0)) / h_sparse.size(1)

        # Dead neuron count (never activated in batch)
        dead_neurons = (active_neurons == 0).sum().item()

        # Activation statistics
        mean_activation = h_sparse.abs().mean().item()
        std_activation = h_sparse.abs().std().item()

        return {
            "sparsity_ratio": sparsity_ratio.mean().item(),
            "neuron_diversity": neuron_diversity,  # Already a float, no need for .item()
            "dead_neurons": dead_neurons,
            "mean_activation": mean_activation,
            "std_activation": std_activation,
        }

    def forward(self, x):

        # reshape to (batch_size * length, channels)

        x = x.view(-1, x.size(-1))

        # Encode
        h = self.encoder(x)

        # Apply sparsity
        h_sparse = self.get_sparse_activations(h)

        # Decode
        x_recon = self.decoder(h_sparse)

        # Calculate metrics
        metrics = self.get_metrics(h_sparse)

        # Reshape back to (batch_size, length, channels)
        x_recon = x_recon.view(x.size(0), x.size(1), -1)

        return x_recon, h_sparse, metrics


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def plot_activation_histogram(activations):
    """Create histogram of activation values"""
    plt.figure(figsize=(10, 6))
    plt.hist(activations.cpu().detach().numpy().flatten(), bins=50)
    plt.title("Distribution of Activation Values")
    plt.xlabel("Activation Value")
    plt.ylabel("Count")

    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = ToTensor()(plt.imread(buf))
    plt.close()
    return image


def train_sparse_autoencoder(
    train_dir: str,
    val_dir: str,
    input_dim: int,
    hidden_dim: int,
    k: int,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    sparsity_factor: float = 10.0,
    patience: int = 7,
    num_workers: int = 4,
    sparsity_method: str = "topk",
    global_max: float = None,
    checkpoint_dir: str = "checkpoints",
    resume_training: bool = False,
):
    """
    Train a sparse autoencoder.

    Args:
        train_dir: Directory containing training .h5 files
        val_dir: Directory containing validation .h5 files
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layer
        k: Number of top activations to keep
        batch_size: Batch size for training
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate for optimizer
        sparsity_factor: Factor f for L1 loss (total_loss = MSE + f*L1)
        patience: Number of epochs to wait for improvement before early stopping
        num_workers: Number of workers for data loading
    """
    # Set up datasets
    train_pattern = str(Path(train_dir) / "activations_*.h5")
    val_pattern = str(Path(val_dir) / "activations_*.h5")

    train_dataset = ActivationDataset(
        train_pattern, transform=NormalizeActivations(global_max=global_max)
    )
    val_dataset = ActivationDataset(
        val_pattern, transform=NormalizeActivations(global_max=global_max)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Initialize model, optimizer, and loss functions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseAutoencoder(input_dim, hidden_dim, k).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    # Set up checkpointing
    checkpoint_handler = ModelCheckpoint(checkpoint_dir)
    start_epoch = 0

    # Resume from checkpoint if requested
    if resume_training:
        latest_checkpoint = Path(checkpoint_dir) / "latest_checkpoint.pt"
        if latest_checkpoint.exists():
            start_epoch, metrics = ModelCheckpoint.load_checkpoint(
                latest_checkpoint, model, optimizer
            )
            print(f"Resuming training from epoch {start_epoch + 1}")

    # Set up tensorboard
    writer = SummaryWriter(log_dir=checkpoint_dir)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience)

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_metrics = {
            "train_mse": 0,
            "train_l1": 0,
            "train_total": 0,
            "train_sparsity_ratio": 0,
            "train_neuron_diversity": 0,
            "train_dead_neurons": 0,
            "train_mean_activation": 0,
            "train_std_activation": 0,
        }

        # Training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for ib, batch in enumerate(pbar):
            batch = batch.to(device)

            # Forward pass
            recon, hidden, batch_metrics = model(batch)

            # Calculate losses
            mse = mse_loss(recon, batch)
            l1 = hidden.abs().mean()
            total_loss = mse + sparsity_factor * l1

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update metrics
            epoch_metrics["train_mse"] += mse.item()
            epoch_metrics["train_l1"] += l1.item()
            epoch_metrics["train_total"] += total_loss.item()

            for key, value in batch_metrics.items():
                epoch_metrics[f"train_{key}"] += value

            if ib % 10 == 0:
                # Log metrics to tensorboard
                for key, value in epoch_metrics.items():
                    writer.add_scalar(f"Metrics/{key}", value/(ib+1), ib + epoch*len(train_loader))

                    pbar.set_postfix(
                        {
                            "MSE": mse.item(),
                            "L1": l1.item(),
                            "Sparsity": batch_metrics["sparsity_ratio"],
                        }
                    )

            if ib % 100 == 0:
                activation_hist = plot_activation_histogram(hidden)
                writer.add_image("Activation_Distribution", activation_hist, epoch)

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= len(train_loader)

        # Log metrics to tensorboard
        #for key, value in epoch_metrics.items():
        #    writer.add_scalar(f"Metrics/{key}", value, epoch)

        # Log activation distribution plot
        #if epoch % 2 == 0:  # Log every 5 epochs
            #activation_hist = plot_activation_histogram(hidden)
            #writer.add_image("Activation_Distribution", activation_hist, epoch)

        # Validation
        model.eval()
        val_metrics = {"val_mse": 0, "val_l1": 0, "val_total": 0}

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, hidden = model(batch)

                mse = mse_loss(recon, batch)
                l1 = hidden.abs().mean()
                total_loss = mse + sparsity_factor * l1

                val_metrics["val_mse"] += mse.item()
                val_metrics["val_l1"] += l1.item()
                val_metrics["val_total"] += total_loss.item()

        # Average validation metrics
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)

        # Combine all metrics
        all_metrics = {**epoch_metrics, **val_metrics}

        # Log metrics to tensorboard
        for key, value in all_metrics.items():
            writer.add_scalar(f"Metrics/{key}", value, epoch)

        # Check if this is the best model
        is_best = checkpoint_handler.is_best(val_metrics["val_total"])

        # Save checkpoint
        checkpoint_handler.save_checkpoint(
            model, optimizer, epoch, all_metrics, is_best=is_best
        )

        # Early stopping check
        early_stopping(val_metrics["val_total"])
        if early_stopping.early_stop:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    return model
