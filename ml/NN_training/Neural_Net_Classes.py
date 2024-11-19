import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau



class Calibration(nn.Module):
    def __init__(self, base_model, criterion, calibration_layer, learning_rate=0.001, patience=100, factor=0.5, threshold=1e-4):
        super(Calibration, self).__init__()
        if not isinstance(base_model, nn.Module):
            raise TypeError("Base model must be an instance of nn.Module")
        self.model =base_model
        self.calibration_layer = calibration_layer
        self.criterion = criterion
        # Freeze the existing network
        for param in self.model.parameters():
            param.requires_grad = False

        # Define an optimizer for the linear layer
        self.optimizer = optim.Adam(calibration_layer.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min',
                                           factor=factor, patience=patience, threshold=threshold)
        self.loss_data = {
            'loss':[],
            'epoch_count':[]
        }

    def forward(self, x):
        x = self.model(x)
        x = self.calibration_layer(x)
        return x

    def train(self, inputs, targets,num_epochs=10000):
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            calibrated_outputs = self.calibration_layer(outputs)
            loss = self.criterion(calibrated_outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            current_loss = loss.item()
            self.loss_data['loss'].append(loss.detach().numpy())
            self.loss_data['epoch_count'].append(epoch)
            self.scheduler.step(current_loss)
            if(epoch+1) % (num_epochs/10) == 0:
                print(f'Comb NN: Epoch [{epoch+1}/{num_epochs}], Loss:{loss.item():.6f}')
                print(f'Calibration Layer Weight: {self.calibration_layer.weight.item()}')
                print(f'Calibration Layer Bias: {self.calibration_layer.bias.item()}')


    def predict(self, inputs):
        '''
        args:
            tensor inputs
        returns:
            numpy array with predictions
        '''
        with torch.no_grad():
            output = self.model(inputs)
            calibrated_test_outputs = self.calibration_layer(output)
            predictions = calibrated_test_outputs.detach().numpy()
        return predictions

    def test_model(self, inputs, outputs):
        '''
        args:
            tensor inputs: an input dataset to pass through NN and test
            tensor outputs: an output dataset to pass through NN
        '''

        self.eval()
        with torch.no_grad():
            predictions = self(inputs)
            loss = self.criterion(predictions, outputs).item()


    def plot_loss(self, filename=None):
        '''
        Args:
            if a string is provided it will save the plot with the given name
        return:
            displays epochs vs loss graph
        '''
        fig, ax = plt.subplots()
        ax.plot(self.loss_data['epoch_count'], self.loss_data['loss'], label='loss')
        plt.title('epochs vs loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')

        if filename:
            plt.savefig(filename+'.png')
        else:
            plt.show()





class NN(nn.Module):
    '''
    5 hidden layer
    2 input, 1 output
    Neural network
    '''
    def __init__(self, hidden_size=20, learning_rate=0.001,
                 patience=100, factor=0.5, threshold=1e-4):
        '''
        args:
            float learning_rate: how much should NN correct when it guesses wrong
            int patience: how many repeated values (plateaus or flat data) should occur before changing learning rate
            float factor: by what factor should learning rate decrease upon scheduler step
            float threshold: how many place values to consider repeated numbers

        '''
        super(NN, self).__init__()

        self.hidden1 = nn.Linear(3, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.hidden3 = nn.Linear(hidden_size, hidden_size)
        self.hidden4 = nn.Linear(hidden_size, hidden_size)
        self.hidden5 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

        self.loss_data = {
            'loss':[],
            'epoch_count':[]
            }

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min',
                                           factor=factor, patience=patience, threshold=threshold)

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

    def train_model(self, inputs, outputs, num_epochs=1500):
        '''
        args:
            tensor inputs input dataset to train NN
            tensor outputs: output dataset to train NN
            int num_epochs: iterations of training
        '''
        oputputs = outputs.to(torch.float32)
        inputs = inputs.to(torch.float32)

        self.train()


        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            predictions = self(inputs)
            loss = self.criterion(predictions, outputs)
            loss.backward()

            self.optimizer.step()

            current_loss = loss.item()
            self.loss_data['loss'].append(loss.detach().numpy())
            self.loss_data['epoch_count'].append(epoch)
            self.scheduler.step(current_loss)

            if(epoch+1) % (num_epochs/10) == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss:{loss.item():.6f}')



    def test_model(self, inputs, outputs):
        '''
        args:
            tensor inputs: an input dataset to pass through NN and test
            tensor outputs: an output dataset to pass through NN
        '''
        inputs = inputs.to(torch.float32)
        outputs = outputs.to(torch.float32)
        self.eval()
        with torch.no_grad():
            predictions = self(inputs)
            loss = self.criterion(predictions, outputs).item()

            print(f'Test Loss: {loss:.4f}')


    def predict(self, inputs):
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





    def plot_loss(self, filename=None):
        '''
        Args:
            if a string is provided it will save the plot with the given name
        return:
            displays epochs vs loss graph
        '''
        fig, ax = plt.subplots()
        ax.plot(self.loss_data['epoch_count'], self.loss_data['loss'], label='loss')
        plt.title('epochs vs loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')

        if filename:
            plt.savefig(filename+'.png')
        else:
            plt.show()
