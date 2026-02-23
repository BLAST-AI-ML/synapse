import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class EarlyStopping:
    def __init__(self, patience=50, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def nan_mse_loss(target, pred):
    """
    Custom MSE loss that ignores NaN values in targets
    (Here, NaN often correspond to missing values in the target data)

    Args:
        target: target values (may contain NaN)
        pred: predicted values

    Returns:
        mean squared error ignoring NaN values
    """
    # Compute squared differences
    squared_diff = (pred - target) ** 2

    # Use nanmean to ignore NaN values
    mse_loss = torch.nanmean(squared_diff)

    # Prevent NaN from contaminating backpropagation
    # See https://github.com/pytorch/pytorch/issues/4132
    if pred.requires_grad:
        nan_mask = torch.isnan(squared_diff)

        def mask_grad_hook(grad):
            return torch.where(nan_mask, 0, grad)

        pred.register_hook(mask_grad_hook)

    return mse_loss


class CombinedNN(nn.Module):
    """
    5 layer neural network trained on simulation data.
    Calibration is handled separately in a second training phase.
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=20,
        learning_rate=0.001,
        patience_LRreduction=100,
        patience_earlystopping=150,
        factor=0.5,
        threshold=1e-4,
    ):
        """
        args:
            float learning_rate: how much should NN correct when it guesses wrong
            int patience: how many repeated values (plateaus or flat data) should occur before changing learning rate
            float factor: by what factor should learning rate decrease upon scheduler step
            float threshold: how many place values to consider repeated numbers

        """
        super(CombinedNN, self).__init__()

        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.hidden3 = nn.Linear(hidden_size, hidden_size)
        self.hidden4 = nn.Linear(hidden_size, hidden_size)
        self.hidden5 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            "min",
            factor=factor,
            patience=patience_LRreduction,
            threshold=threshold,
        )
        self.early_stopper = EarlyStopping(patience=patience_earlystopping)

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        x = self.relu(self.hidden4(x))
        x = self.relu(self.hidden5(x))
        x = self.output(x)
        return x

    def train_model(
        self,
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        num_epochs=1500,
    ):
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            outputs = self(train_inputs)
            loss = nan_mse_loss(train_targets, outputs)
            loss.backward()

            self.optimizer.step()

            current_loss = loss.item()
            self.scheduler.step(current_loss)

            with torch.no_grad():
                val_outputs = self(val_inputs)
                val_loss = nan_mse_loss(val_targets, val_outputs)

            if (epoch + 1) % (num_epochs / 10) == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Loss:{loss.item():.6f}, Val Loss:{val_loss.item():.6f}"
                )

            self.early_stopper(val_loss.item())
            if self.early_stopper.early_stop:
                print(
                    f'Early stopping triggered at epoch {epoch} with val loss {val_loss.item():.6f}'
                )
                break

    def predict(self, inputs):
        inputs = inputs.to(torch.float32)
        self.eval()
        with torch.no_grad():
            output = self(inputs)
            predictions = output.detach().numpy()
        return predictions


def train_calibration(
    base_predictions,
    exp_targets,
    n_outputs,
    num_epochs=5000,
    lr=0.001,
):
    """
    Train per-output affine calibration parameters (weight * prediction + bias)
    on experimental data. The base model predictions are pre-computed and detached.

    Args:
        base_predictions: model predictions on exp inputs (detached tensor)
        exp_targets: experimental target values (may contain NaN)
        n_outputs: number of output dimensions
        num_epochs: number of training epochs
        lr: learning rate

    Returns:
        (cal_weight, cal_bias) as detached tensors
    """
    cal_weight = nn.Parameter(torch.ones(n_outputs, dtype=base_predictions.dtype))
    cal_bias = nn.Parameter(torch.zeros(n_outputs, dtype=base_predictions.dtype))

    optimizer = optim.Adam([cal_weight, cal_bias], lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=200, threshold=1e-4
    )
    early_stopper = EarlyStopping(patience=500)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        calibrated = cal_weight * base_predictions + cal_bias
        loss = nan_mse_loss(exp_targets, calibrated)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        scheduler.step(current_loss)

        if (epoch + 1) % (num_epochs / 10) == 0:
            print(
                f"Calibration Epoch [{epoch + 1}/{num_epochs}], Loss:{current_loss:.6f}"
            )

        early_stopper(current_loss)
        if early_stopper.early_stop:
            print(
                f"Calibration early stopping at epoch {epoch} with loss {current_loss:.6f}"
            )
            break

    print(f"Learned calibration weight: {cal_weight.data}")
    print(f"Learned calibration bias: {cal_bias.data}")

    return cal_weight.detach(), cal_bias.detach()
