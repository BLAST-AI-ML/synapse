import torch
import matplotlib.pyplot as plt

def train_model(model, x_train, y_train, z_train, num_epochs=1500):
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

    model.train()


    for epoch in range(num_epochs):
        model.optimizer.zero_grad()

        outputs = model(inputs)
        loss = model.criterion(outputs, z_train)
        loss.backward()

        model.optimizer.step()

        current_loss = loss.item()
        model.loss_data['loss'].append(loss.detach().numpy())
        model.loss_data['epoch_count'].append(epoch)
        model.scheduler.step(current_loss)

        if(epoch+1) % (num_epochs/10) == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss:{loss.item():.6f}')



def test_model(model, x_test, y_test, z_test):
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
    model.eval()
    with torch.no_grad():
        output = model(inputs)
        loss = model.criterion(output, z_test).item()

        print(f'Test Loss: {loss:.4f}')






def predict(model, x_values, y_values):
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
    model.eval()
    with torch.no_grad():
        output = model(inputs)
        predictions['predictions'] = output.detach().numpy().tolist()

    return predictions





def plot_loss(model, filename=None):
    '''
    Args:
        if a string is provided it will save the plot with the given name
    return:
        displays epochs vs loss graph
    '''
    fig, ax = plt.subplots()
    ax.plot(model.loss_data['epoch_count'], model.loss_data['loss'], label='loss')
    plt.title('epochs vs loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')

    if filename:
        plt.savefig(filename+'.png')
    else:
        plt.show()
