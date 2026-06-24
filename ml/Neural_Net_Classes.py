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
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=self.factor,
            patience=self.patience_LRreduction,
            threshold=self.threshold,
        )
        early_stopper = EarlyStopping(patience=self.patience_earlystopping)

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            outputs = self(train_inputs)
            loss = nan_mse_loss(train_targets, outputs)
            loss.backward()

            optimizer.step()

            current_loss = loss.item()
            scheduler.step(current_loss)

            with torch.no_grad():
                val_outputs = self(val_inputs)
                val_loss = nan_mse_loss(val_targets, val_outputs)

            if (epoch + 1) % (num_epochs / 10) == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Loss:{loss.item():.6f}, Val Loss:{val_loss.item():.6f}"
                )

            early_stopper(val_loss.item())
            if early_stopper.early_stop:
                print(
                    f"Early stopping triggered at epoch {epoch} with val loss {val_loss.item():.6f}"
                )
                break


def _calibration_penalty(
    c_normcal,
    o_normcal,
    c_guess,
    o_guess,
    c_norm,
    o_norm,
    alpha_uncertainty,
    beta_uncertainty,
):
    """Compute penalty that keeps inferred alpha/beta near their guess values.

    The inferred calibration is obtained by composing the guess calibration,
    normalization, and learned normalized calibration (see build_inferred_calibration).
    From those compositions:
        alpha_inferred = 1 / (c_guess * c_normcal)
        beta_inferred  = o_guess + c_guess*o_norm + c_guess*c_norm*o_normcal
                         - c_guess*c_normcal*o_norm

    The penalty is  sum((alpha_I - alpha_G)^2 / alpha_U^2)
                  + sum((beta_I  - beta_G )^2 / beta_U^2)
    where alpha_G = 1/c_guess and beta_G = o_guess.
    Dimensions with infinite uncertainty contribute zero penalty.
    """
    c_inferred = c_guess * c_normcal
    alpha_inferred = 1.0 / c_inferred
    alpha_guess = 1.0 / c_guess

    beta_inferred = (
        o_guess + c_guess * o_norm + c_guess * c_norm * o_normcal - c_inferred * o_norm
    )
    beta_guess = o_guess

    penalty_alpha = torch.sum(
        (alpha_inferred - alpha_guess) ** 2 / alpha_uncertainty**2
    )
    penalty_beta = torch.sum((beta_inferred - beta_guess) ** 2 / beta_uncertainty**2)
    return penalty_alpha + penalty_beta


def train_calibration(
    model,
    exp_inputs,
    exp_targets,
    c_guess_input,
    o_guess_input,
    c_norm_input,
    o_norm_input,
    alpha_uncertainty_input,
    beta_uncertainty_input,
    c_guess_output,
    o_guess_output,
    c_norm_output,
    o_norm_output,
    alpha_uncertainty_output,
    beta_uncertainty_output,
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

    A penalization term is added to the loss to keep the inferred alpha/beta
    close to their guess values (see _calibration_penalty). Dimensions with
    infinite uncertainty contribute zero penalty.

    Args:
        model: frozen callable that maps exp_inputs -> predictions
        exp_inputs: experimental input tensor
        exp_targets: experimental target values (may contain NaN)
        c_guess_input: guess calibration coefficients for inputs
        o_guess_input: guess calibration offsets for inputs
        c_norm_input: normalization coefficients for inputs
        o_norm_input: normalization offsets for inputs
        alpha_uncertainty_input: uncertainty on alpha for inputs (inf = no penalty)
        beta_uncertainty_input: uncertainty on beta for inputs (inf = no penalty)
        c_guess_output: guess calibration coefficients for outputs
        o_guess_output: guess calibration offsets for outputs
        c_norm_output: normalization coefficients for outputs
        o_norm_output: normalization offsets for outputs
        alpha_uncertainty_output: uncertainty on alpha for outputs (inf = no penalty)
        beta_uncertainty_output: uncertainty on beta for outputs (inf = no penalty)
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

    optimizer = optim.Adam(
        [c_normcal_input, o_normcal_input, c_normcal_output, o_normcal_output], lr=lr
    )
    scheduler = ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=200, threshold=1e-4
    )
    early_stopper = EarlyStopping(patience=500)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        calibrated_inputs = (1.0 / c_normcal_input) * (exp_inputs - o_normcal_input)
        base_predictions = model(calibrated_inputs)
        calibrated_outputs = c_normcal_output * base_predictions + o_normcal_output

        loss = (
            nan_mse_loss(exp_targets, calibrated_outputs)
            + _calibration_penalty(
                c_normcal_input,
                o_normcal_input,
                c_guess_input,
                o_guess_input,
                c_norm_input,
                o_norm_input,
                alpha_uncertainty_input,
                beta_uncertainty_input,
            )
            + _calibration_penalty(
                c_normcal_output,
                o_normcal_output,
                c_guess_output,
                o_guess_output,
                c_norm_output,
                o_norm_output,
                alpha_uncertainty_output,
                beta_uncertainty_output,
            )
        )

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

    return (
        c_normcal_input.detach(),
        o_normcal_input.detach(),
        c_normcal_output.detach(),
        o_normcal_output.detach(),
    )
