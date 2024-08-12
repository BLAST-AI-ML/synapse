import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


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

        self.hidden1 = nn.Linear(2, hidden_size)
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

    def train_model(self, x_train, y_train, z_train, num_epochs=1500):
        '''
        args:
            tensor x_train: input dataset to train NN
            tensor y_train: input dataset to train NN
            tensor z_train: output dataset to train NN
            int num_epochs: iterations of training
        '''
        x_train = x_train.to(torch.float32)
        y_train = y_train.to(torch.float32)
        z_train = z_train.to(torch.float32)
        
        inputs = torch.cat((x_train, y_train), dim=1).to(torch.float32)

        self.train()


        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            outputs = self(inputs)
            loss = self.criterion(outputs, z_train)
            loss.backward()

            self.optimizer.step()

            current_loss = loss.item()
            self.loss_data['loss'].append(loss.detach().numpy())
            self.loss_data['epoch_count'].append(epoch)
            self.scheduler.step(current_loss)

            if(epoch+1) % (num_epochs/10) == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss:{loss.item():.6f}')



    def test_model(self, x_test, y_test, z_test):
        '''
        args:
            tensor x_test: an input dataset to pass through NN and test
            tensor y_test: an input dataset to pass through NN
            tensor z_test: an output dataset to pass through NN
        returns:
            numpy array output: Returns the predictions the NN made with x_test
        '''
        x_test = x_test.to(torch.float32)
        y_test = y_test.to(torch.float32)
        z_test = z_test.to(torch.float32)
        
        inputs = torch.cat((x_test, y_test),dim=1).to(torch.float32)
        self.eval()
        with torch.no_grad():
            output = self(inputs)
            loss = self.criterion(output, z_test).item()
            predicted_values = output.detach().numpy()
            
            print(f'Test Loss: {loss:.4f}')

        




    def predict(self, x_values, y_values):
        '''
        args:
            tensor x_values
            tensor y_values
        returns:
            numpy array with predictions
        '''
        predictions = {
        'Z_target': x_values.tolist(),
        'TOD': y_values.tolist(),
        'predictions': []
        }
    
        inputs = torch.cat((x_values, y_values), dim=1).to(torch.float32)
        self.eval()
        with torch.no_grad():
            output = self(inputs)
            predictions['predictions'] = output.detach().numpy().tolist()
    
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


