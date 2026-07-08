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


def _run_training_loop(
    parameters,
    forward_fn,
    train_inputs,
    train_targets,
    val_inputs=None,
    val_targets=None,
    num_epochs=1500,
    lr=0.001,
    patience_lr=100,
    patience_early=150,
    factor=0.5,
    threshold=1e-4,
    label="",
    aux_loss_fn=None,
):
    prefix = f"{label} " if label else ""
    optimizer = optim.Adam(parameters, lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, "min", factor=factor, patience=patience_lr, threshold=threshold
    )
    early_stopper = EarlyStopping(patience=patience_early)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        outputs = forward_fn(train_inputs)
        loss = nan_mse_loss(train_targets, outputs)
        if aux_loss_fn is not None:
            loss = loss + aux_loss_fn()
        loss.backward()
        optimizer.step()

        current_loss = loss.item()

        if val_inputs is not None:
            with torch.no_grad():
                val_outputs = forward_fn(val_inputs)
                val_loss = nan_mse_loss(val_targets, val_outputs)
            monitor_loss = val_loss.item()
        else:
            monitor_loss = current_loss

        scheduler.step(monitor_loss)

        if (epoch + 1) % (num_epochs / 10) == 0:
            if val_inputs is not None:
                print(
                    f"{prefix}Epoch [{epoch + 1}/{num_epochs}], Loss:{current_loss:.6f}, Val Loss:{monitor_loss:.6f}"
                )
            else:
                print(
                    f"{prefix}Epoch [{epoch + 1}/{num_epochs}], Loss:{current_loss:.6f}"
                )

        early_stopper(monitor_loss)
        if early_stopper.early_stop:
            print(
                f"{prefix}Early stopping at epoch {epoch} with loss {monitor_loss:.6f}"
            )
            break


class CombinedNN(nn.Module):
    """
    5 layer neural network
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

        self.learning_rate = learning_rate
        self.patience_LRreduction = patience_LRreduction
        self.patience_earlystopping = patience_earlystopping
        self.factor = factor
        self.threshold = threshold

    def forward(self, x):
        """
        args:
            x: single value or tensor to pass
        returns:
            output of NN
        """

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
        _run_training_loop(
            parameters=list(self.parameters()),
            forward_fn=self,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            num_epochs=num_epochs,
            lr=self.learning_rate,
            patience_lr=self.patience_LRreduction,
            patience_early=self.patience_earlystopping,
            factor=self.factor,
            threshold=self.threshold,
        )


def train_calibration(
    model,
    exp_inputs,
    exp_targets,
    num_epochs=5000,
    lr=0.001,
):
    """
    Train per-output affine calibration parameters on experimental data.

    The learned parameters follow the same convention as `AffineInputTransform`:
      - coefficients (c_normcal): scale factors (initialized to 1)
      - offsets (o_normcal): shift values (initialized to 0)

    The calibrated forward pass is:
      calibrated_input  = (1 / c_normcal_input) * (x - o_normcal_input)
      calibrated_output = c_normcal_output * model(calibrated_input) + o_normcal_output

    Args:
        model: frozen callable that maps exp_inputs -> predictions
        exp_inputs: experimental input tensor
        exp_targets: experimental target values (may contain NaN)
        num_epochs: number of training epochs
        lr: learning rate

    Returns:
        (c_normcal_input, o_normcal_input, c_normcal_output, o_normcal_output)
        as detached tensors
    """
    n_outputs = exp_targets.shape[1]
    n_inputs = exp_inputs.shape[1]
    device = exp_inputs.device

    c_normcal_input = nn.Parameter(
        torch.ones(n_inputs, dtype=exp_inputs.dtype, device=device)
    )
    o_normcal_input = nn.Parameter(
        torch.zeros(n_inputs, dtype=exp_inputs.dtype, device=device)
    )
    c_normcal_output = nn.Parameter(
        torch.ones(n_outputs, dtype=exp_inputs.dtype, device=device)
    )
    o_normcal_output = nn.Parameter(
        torch.zeros(n_outputs, dtype=exp_inputs.dtype, device=device)
    )

    def calibrated_forward(x):
        calibrated_inputs = (1.0 / c_normcal_input) * (x - o_normcal_input)
        base_predictions = model(calibrated_inputs)
        return c_normcal_output * base_predictions + o_normcal_output

    _run_training_loop(
        parameters=[c_normcal_input, o_normcal_input, c_normcal_output, o_normcal_output],
        forward_fn=calibrated_forward,
        train_inputs=exp_inputs,
        train_targets=exp_targets,
        num_epochs=num_epochs,
        lr=lr,
        patience_lr=200,
        patience_early=500,
        label="Calibration",
    )

    return (
        c_normcal_input.detach(),
        o_normcal_input.detach(),
        c_normcal_output.detach(),
        o_normcal_output.detach(),
    )


def train_unified(
    model,
    sim_inputs,
    sim_targets,
    sim_val_inputs,
    sim_val_targets,
    exp_inputs,
    exp_targets,
    num_epochs=20000,
    lr=0.0001,
    exp_weight=1.0,
):
    """
    Jointly train NN weights + affine calibration on sim + exp data.

    Loss = nan_mse_loss(sim_targets, model(sim_inputs))
           + exp_weight * nan_mse_loss(exp_targets, calibrated_forward(exp_inputs))

    Calibration convention (matches AffineInputTransform / train_calibration):
      calibrated_x   = (1 / c_normcal_input) * (x - o_normcal_input)
      calibrated_out = c_normcal_output * model(calibrated_x) + o_normcal_output

    Early stopping monitors sim validation loss.

    Returns (c_normcal_input, o_normcal_ormcal_output)
    as detached tensors — same contract as train_calibration().
    """
    n_outputs = exp_targets.shape[1]
    n_inputs = exp_inputs.shape[1]
    device = exp_inputs.device

    c_normcal_input = nn.Parameter(
        torch.ones(n_inputs, dtype=exp_inputs.dtype, device=device)
    )
    o_normcal_input = nn.Parameter(
        torch.zeros(n_inputs, dtype=exp_inputs.dtype, device=device)
    )
    c_normcal_output = nn.Parameter(
        torch.ones(n_outputs, dtype=exp_inputs.dtype, device=device)
    )
    o_normcal_output = nn.Parameter(
        torch.zeros(n_outputs, dtype=exp_inputs.dtype, device=device)
    )

    def calibrated_forward(x):
        calibrated_inputs = (1.0 / c_normcal_input) * (x - o_normcal_input)
        base_predictions = model(calibrated_inputs)
        return c_normcal_output * base_predictions + o_normcal_output

    def exp_loss_fn():
        return exp_weight * nan_mse_loss(exp_targets, calibrated_forward(exp_inputs))

    _run_training_loop(
        parameters=list(model.parameters())
        + [c_normcal_input, o_normcal_input, c_normcal_output, o_normcal_output],
        forward_fn=model,
        train_inputs=sim_inputs,
        train_targets=sim_targets,
        val_inputs=sim_val_inputs,
        val_targets=sim_val_targets,
        num_epochs=num_epochs,
        lr=lr,
        patience_lr=200,
        patience_early=500,
        label="Unified",
        aux_loss_fn=exp_loss_fn,
    )

    return (
        c_normcal_input.detach(),
        o_normcal_input.detach(),
        c_normcal_output.detach(),
        o_normcal_output.detach(),
    )
