import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class EarlyStopping:
    def __init__(self, patience=50, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class CombinedNN(nn.Module):
    """
    Model that trains a 5 layer neural network and a calibration layer
    """
    def __init__(self, input_size, output_size, hidden_size=20,
                 learning_rate=0.001, patience_LRreduction=100, patience_earlystopping=150, factor=0.5, threshold=1e-4):
        '''
        args:
            float learning_rate: how much should NN correct when it guesses wrong
            int patience: how many repeated values (plateaus or flat data) should occur before changing learning rate
            float factor: by what factor should learning rate decrease upon scheduler step
            float threshold: how many place values to consider repeated numbers

        '''
        super(CombinedNN, self).__init__()

        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.hidden3 = nn.Linear(hidden_size, hidden_size)
        self.hidden4 = nn.Linear(hidden_size, hidden_size)
        self.hidden5 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        # Component-wise linear transformation: weight * input + bias
        # where weight and bias are vectors of size output_size
        self.sim_to_exp_calibration_weight = nn.Parameter(torch.ones(output_size))
        self.sim_to_exp_calibration_bias = nn.Parameter(torch.zeros(output_size))

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min',
                                           factor=factor, patience=patience_LRreduction, threshold=threshold)
        self.early_stopper = EarlyStopping(patience=patience_earlystopping)

    @torch.jit.export
    def calibrate(self, x):
        """Apply component-wise linear transformation: weight * x + bias."""
        return self.sim_to_exp_calibration_weight * x + self.sim_to_exp_calibration_bias

    def forward(self, x):
        '''
        args:
            x: single value or tensor to pass
        returns:
            output of NN
        '''

        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        x = self.relu(self.hidden4(x))
        x = self.relu(self.hidden5(x))
        x = self.output(x)

        return x

    def train_model(self, sim_inputs, sim_targets,
                    exp_inputs, exp_targets,
                    sim_inputs_val, sim_targets_val,
                    exp_inputs_val, exp_targets_val, num_epochs=1500):

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            loss = 0
            if len(sim_inputs) > 0:
                sim_outputs = self(sim_inputs)
                loss += self.criterion( sim_targets, sim_outputs )
            if len(exp_inputs) > 0:
                exp_outputs = self.sim_to_exp_calibration( self(exp_inputs) )
                loss += self.criterion( exp_targets, exp_outputs )
            loss.backward()

            self.optimizer.step()

            current_loss = loss.item()
            self.scheduler.step(current_loss)

            # compute validation loss for early stopping
            with torch.no_grad():
                val_loss = 0
                if len(sim_inputs_val) > 0:
                    sim_outputs_val = self(sim_inputs_val)
                    val_loss += self.criterion( sim_targets_val, sim_outputs_val )
                if len(exp_inputs_val) > 0:
                    exp_outputs_val = self(exp_inputs_val)
                    val_loss += self.criterion( exp_targets_val, exp_outputs_val )


            if(epoch+1) % (num_epochs/10) == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss:{loss.item():.6f}, Val Loss:{val_loss.item():.6f}')

            self.early_stopper(val_loss.item())
            if self.early_stopper.early_stop:
                print(f'Early stopping triggered at, {epoch}  "with val loss ", {val_loss.item():.6f}' )
                break



    def predict_sim(self, inputs):
        '''
        args:
            tensor inputs
        returns:
            numpy array with predictions
        '''
        inputs = inputs.to(torch.float32)
        self.eval()
        with torch.no_grad():
            output = self(inputs)
            predictions = output.detach().numpy()

        return predictions

    def predict_exp(self, inputs):
        '''
        args:
            tensor inputs
        returns:
            numpy array with predictions
        '''
        inputs = inputs.to(torch.float32)
        self.eval()
        with torch.no_grad():
            output = self.calibrate(self(inputs))
            predictions = output.detach().numpy()

        return predictions
