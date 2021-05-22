import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch import optim
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import matplotlib as mpl


"""
The 5 requested models- A,B,C,D,E
"""


# Model A
class NeuralNetworkA(nn.Module):
    """
    Neural Network with 2 hidden layers, when the first layer has a size of 100 and the second
    layer has a size of 50, both are followed by ReLU activation function.
    """
    def __init__(self, image_size):
        super(NeuralNetworkA, self).__init__()
        self.name = "Model A"
        self.image_size = image_size
        self.fc0 = nn.Linear(self.image_size, 100)  # input layer
        self.fc1 = nn.Linear(100, 50)  # first hidden layer size 100
        self.fc2 = nn.Linear(50, 10)  # second hidden layer size 50

    def forward(self, x):
        x = x.view(-1, self.image_size)
        # activation function on layers- relu
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # we do log_softmax on the last layer to get an input between 0 and 10
        return F.log_softmax(x, dim=1)


# model B
class NeuralNetworkB(nn.Module):
    """
    Neural Network with 2 hidden layers, when the first layer has a size of 100 and the second
    layer has a size of 50, both are followed by ReLU activation function. Also with dropout layers
    when p=0.2.
    """
    def __init__(self, image_size):
        super(NeuralNetworkB, self).__init__()
        self.name = "Model B"
        self.image_size = image_size
        self.fc0 = nn.Linear(self.image_size, 100)  # input layer
        self.fc1 = nn.Linear(100, 50)  # first hidden layer size 100
        self.fc2 = nn.Linear(50, 10)  # second hidden layer size 50
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        # activation function on layers- relu
        x = self.dropout(F.relu(self.fc0(x)))
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        # we do log_softmax on the last layer to get an input between 0 and 10
        return F.log_softmax(x, dim=1)


# Model C
class NeuralNetworkC(nn.Module):
    """
    Neural Network with 2 hidden layers, when the first layer has a size of 100 and the second
    layer has a size of 50, both are followed by ReLU activation function. Also with Batch Normalization
    before the activation functions
    """
    def __init__(self, image_size):
        super(NeuralNetworkC, self).__init__()
        self.name = "Model C"
        self.image_size = image_size
        self.fc0 = nn.Linear(self.image_size, 100)  # input layer
        self.fc1 = nn.Linear(100, 50)  # first hidden layer size 100
        self.fc2 = nn.Linear(50, 10)  # second hidden layer size 50
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(50)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        # activation function on layers- relu
        x = self.bn1(F.relu(self.fc0(x)))
        x = self.bn2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        # we do log_softmax on the last layer to get an input between 0 and 10
        return F.log_softmax(x, dim=1)


# model D
class NeuralNetworkD(nn.Module):
    """
    Neural Network with 5 hidden layers: [128,64,10,10,10] using ReLU.
    """
    def __init__(self, image_size):
        super(NeuralNetworkD, self).__init__()
        self.name = "Model D"
        self.image_size = image_size
        self.fc0 = nn.Linear(self.image_size, 128)  # input layer
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        # activation function on layers- relu
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        # we do log_softmax on the last layer to get an input between 0 and 10
        return F.log_softmax(x, dim=1)


# model E
class NeuralNetworkE(nn.Module):
    """
    Neural Network with 5 hidden layers: [128,64,10,10,10] using Sigmoid.
    """
    def __init__(self, image_size):
        super(NeuralNetworkE, self).__init__()
        self.name = "Model E"
        self.image_size = image_size
        self.fc0 = nn.Linear(self.image_size, 128)  # input layer
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        # activation function on layers- sigmoid
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.fc5(x)
        # we do log_softmax on the last layer to get an input between 0 and 10
        return F.log_softmax(x, dim=1)


"""
Train and Test functions for the train and validation with the train we loaded from the 
FashionMNIST data set.
"""


def train(model, optimizer, train_loader, val_set):
    """
    The training function. We train our model for 10 epochs (like requested). Every epoch we zero
    the optimizer grad from the earlier epoch, calculate our model, calculate loss, do backpropagation
    and one optimizing step.
    """
    # for the plots afterwards
    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []
    epochs = 10
    # run the main training loop
    for epoch in range(epochs):
        model.train()
        correct = 0
        avg_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            net_out = model(data)
            loss = F.nll_loss(net_out, target, reduction='sum')
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
            pred =net_out.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()

        # 48000 should be changer to 60000 if the train set is all the train loader of FashionMNIST
        # and not just 80% of it
        avg_loss /= 48000
        print('Train Epoch: {} accuracy: {}/{} = {}% \tLoss: {:.5f}'.format(
            epoch, correct, 48000, 100. * float(correct) / 48000, avg_loss))
        train_loss.append(avg_loss)
        train_accuracy.append(100. * float(correct) / 48000)
        # test function
        my_acc, my_loss = test(model, val_set)
        val_accuracy.append(my_acc)
        val_loss.append(my_loss)
    return train_accuracy, val_accuracy, train_loss, val_loss


def test(model, test_set):
    """
    The test function. for every (data, traget) in our train set we calculate the output according
    to our model, calculating the loss, get the index of the max log-probability, and claculate
    "correct" variable to help us later calculate the accuracy.
    """
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_set:
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).cpu().sum()

    # 12000 should be changer to 10000 if the train set is all the test loader of FashionMNIST
    # and not 20% of the train loader that I used as validation set
    test_loss /= 12000
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    test_loss, correct, 12000, 100. * float(correct) / 12000))
    accuracy = 100. * float(correct) / 12000
    return accuracy, test_loss


def my_train(model, optimizer, train_loader, test_set):
    """
    The training function. We train our model for 10 epochs (like requested). Every epoch we zero
    the optimizer grad from the earlier epoch, calculate our model, calculate loss, do backpropagation
    and one optimizing step.
    """
    # for the plots afterwards
    train_accuracy = []
    train_loss = []
    epochs = 10
    # run the main training loop
    for epoch in range(epochs):
        model.train()
        correct = 0
        avg_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            net_out = model(data)
            loss = F.nll_loss(net_out, target, reduction='sum')
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
            pred =net_out.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()

        avg_loss /= 60000
        print('Train Epoch: {} accuracy: {}/{} = {}% \tLoss: {:.5f}'.format(
            epoch, correct, 60000, 100. * float(correct) / 60000, avg_loss))
        train_loss.append(avg_loss)
        train_accuracy.append(100. * float(correct) / 60000)
        # test function
        predictions = my_test_x(model, test_set)
    return train_accuracy, train_loss, predictions


def my_test_x(model, test_file):
    """
    Test function for the specific "test_x" file that was provided to us. Returns the list of predictions
    """
    model.eval()
    predictions= []
    for data in test_file:
        output = model(data)
        p = output.max(1, keepdim=True)[1]
        p = int(p)
        predictions.append(p)
    return predictions


"""
functions to plot the requested graphs 
"""


def plot_graph_accuracy(model, train, val):
    epochs = [e for e in range(1, 11)]
    label1, label2, title = "", "", ""
    model_name = model.name
    label1 = 'Train Accuracy'
    label2 = 'Validation Accuracy'
    title = '%s Accuracy per Epoch' % model_name
    plt.figure(1)
    plt.plot(epochs, train, '-ok', color='red', label=label1)
    plt.plot(epochs, val, '-ok', color='blue', label=label2)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def plot_graph_loss(model, train, val):
    epochs = [e for e in range(1, 11)]
    label1, label2, title = "", "", ""
    model_name = model.name
    label1 = 'Train Average Loss'
    label2 = 'Validation Average Loss'
    title = '%s Average loss per Epoch' % model_name
    plt.figure(2)
    plt.plot(epochs, train, '-ok', color='red', label=label1)
    plt.plot(epochs, val, '-ok', color='blue', label=label2)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('average loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # for the graphs
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 20
    mpl.rcParams['axes.labelsize'] = 16

    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0,), (0.5,))])
    my_dataset = datasets.FashionMNIST("./data", download=True, train=True, transform=transforms.Compose(
        [transforms.ToTensor()]))

    suffle = True
    random_seed = 0  # so the shufflw will be the same for every run
    split_val = .2
    batch_size = 64

    """
    a code to divide the train loader to 80% train and 20% validation
    """

    size = len(my_dataset)
    index = list(range(size))
    split = int(np.floor(split_val * size))
    if suffle:
        np.random.seed(random_seed)
        np.random.shuffle(index)
    index_train, index_val = index[split:], index[:split]

    train1 = SubsetRandomSampler(index_train)
    validiation = SubsetRandomSampler(index_val)

    train_loader = DataLoader(my_dataset, sampler=train1, batch_size=64)
    validiation_loader = DataLoader(my_dataset, sampler=validiation, batch_size=64)

    # 100% train that we load and the test of build FashoinMNIST data set in pytorch
    my_train_loader = DataLoader(datasets.FashionMNIST('./data', train=True, download=True, transform=transform),
                                 batch_size=64, shuffle=True)
    test_loader = DataLoader(datasets.FashionMNIST('./data', train=False, transform=transform), batch_size=64,
                             shuffle=True)

    # load the given test file
    test_x = np.loadtxt("test_x")
    test_x /= 255.
    test_x = torch.FloatTensor(test_x)

    # creating our neural networks
    modelA = NeuralNetworkA(image_size=28 * 28)
    modelB = NeuralNetworkB(image_size=28 * 28)
    modelC = NeuralNetworkC(image_size=28 * 28)
    modelD = NeuralNetworkD(image_size=28 * 28)
    modelE = NeuralNetworkE(image_size=28 * 28)

    models_list = [modelA, modelB, modelC, modelD, modelE]

    """
    a code for the report- to train 80% of the train loader and 20% as validation
    """

    for i in range(len(models_list)):
        print("model", i, ":")
        model = models_list[i]
        # optimizer: for every model we do a different optimizer
        if i == 0:
            optimizer = optim.Adam(model.parameters(), lr=0.00025)
            train_accuracy, test_accuracy, train_loss, test_loss = train(model, optimizer,
                                                                     train_loader, validiation_loader)
            plot_graph_accuracy(model, train_accuracy, test_accuracy)
            plot_graph_loss(model, train_loss, test_loss)
        elif i == 1:
            optimizer = optim.Adam(model.parameters(), lr=0.0005)
            train_accuracy, test_accuracy, train_loss, test_loss = train(model, optimizer,
                                                                         train_loader, validiation_loader)
            plot_graph_accuracy(model, train_accuracy, test_accuracy)
            plot_graph_loss(model, train_loss, test_loss)
        elif i == 2:
            optimizer = optim.Adam(model.parameters(), lr=0.00001)
            train_accuracy, test_accuracy, train_loss, test_loss = train(model, optimizer,
                                                                         train_loader, validiation_loader)
            plot_graph_accuracy(model, train_accuracy, test_accuracy)
            plot_graph_loss(model, train_loss, test_loss)
        elif i == 3:
            optimizer = optim.Adam(model.parameters(), lr=0.0035)
            train_accuracy, test_accuracy, train_loss, test_loss = train(model, optimizer,
                                                                         train_loader, validiation_loader)
            plot_graph_accuracy(model, train_accuracy, test_accuracy)
            plot_graph_loss(model, train_loss, test_loss)
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            train_accuracy, test_accuracy, train_loss, test_loss = train(model, optimizer,
                                                                         train_loader, validiation_loader)
            plot_graph_accuracy(model, train_accuracy, test_accuracy)
            plot_graph_loss(model, train_loss, test_loss)

    """
    a code to get the accuracy of the test loader from orighinal FashionMNIST data set
    """

    acc_test = []
    for i in range(len(models_list)):
        print("model", i, ":")
        model = models_list[i]
        # optimizer: for every model we do a different optimizer
        if i == 0:
            optimizer = optim.Adam(model.parameters(), lr=0.00025)
            train_accuracy, test_accuracy, train_loss, test_loss = train(model, optimizer,
                                                                         my_train_loader, test_loader)
            acc_test.append(test_accuracy[9])
        elif i == 1:
            optimizer = optim.Adam(model.parameters(), lr=0.0005)
            train_accuracy, test_accuracy, train_loss, test_loss = train(model, optimizer,
                                                                         my_train_loader, test_loader)
            acc_test.append(test_accuracy[9])
        elif i == 2:
            optimizer = optim.Adam(model.parameters(), lr=0.00001)
            train_accuracy, test_accuracy, train_loss, test_loss = train(model, optimizer,
                                                                         my_train_loader, test_loader)
            acc_test.append(test_accuracy[9])
        elif i == 3:
            optimizer = optim.Adam(model.parameters(), lr=0.0035)
            train_accuracy, test_accuracy, train_loss, test_loss = train(model, optimizer,
                                                                         my_train_loader, test_loader)
            acc_test.append(test_accuracy[9])
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            train_accuracy, test_accuracy, train_loss, test_loss = train(model, optimizer,
                                                                         my_train_loader, test_loader)
            acc_test.append(test_accuracy[9])
    print(acc_test)


    """
    a code to create "test_y" file with my best model- model B
    """

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    acc, loss, predictions = my_train(model, optimizer, my_train_loader, test_x)
    file = open('test_y', 'w')
    for i in range(len(predictions) - 1):
        file.write(f"{predictions[i]}\n")
    file.write(f"{predictions[len(predictions)-1]}")
    file.close()
