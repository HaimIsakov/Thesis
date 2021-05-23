import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score


class JustValuesOnNodes(nn.Module):

    def __init__(self, data_size, RECEIVED_PARAMS):
        super(JustValuesOnNodes, self).__init__()
        self.data_size = data_size
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.fc1 = nn.Linear(self.data_size, self.RECEIVED_PARAMS["layer_1"])  # input layer
        self.fc2 = nn.Linear(self.RECEIVED_PARAMS["layer_1"], self.RECEIVED_PARAMS["layer_2"])
        self.fc3 = nn.Linear(self.RECEIVED_PARAMS["layer_2"], 1)

    def forward(self, x):
        # x = x.view(-1, self.data_size)
        if self.RECEIVED_PARAMS['activation'] == 'relu':
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        elif self.RECEIVED_PARAMS['activation'] == 'elu':
            x = F.elu(self.fc1(x))
            x = F.elu(self.fc2(x))
        elif self.RECEIVED_PARAMS['activation'] == 'tanh':
            x = F.tanh(self.fc1(x))
            x = F.tanh(self.fc2(x))
        # x = torch.sigmoid(x) # BCE loss automatically applies sigmoid
        x = self.fc3(x)
        return x


def my_train(model, RECEIVED_PARAMS, epochs, train_loader, test_loader, device, loss_weights):
    """
    The training function. We train our model for 10 epochs (like requested). Every epoch we zero
    the optimizer grad from the earlier epoch, calculate our model, calculate loss, do backpropagation
    and one optimizing step.
    """
    # for the plots afterwards
    # train_accuracy = []
    train_loss = []
    test_loss_vec = []
    train_acc = []
    test_acc = []
    test_auc = []
    all_targets = []
    all_pred = []
    optimizer = get_optimizer(RECEIVED_PARAMS, model)
    # run the main training loop
    for epoch in range(epochs):
        print(f"epoch {epoch}")
        model.train()
        # correct = 0
        # avg_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            net_out = model(data)
            loss = F.binary_cross_entropy_with_logits(net_out, target.unsqueeze(dim=1).float(), weight=torch.Tensor([loss_weights[i].item() for i in target]).unsqueeze(dim=1).to(device))
            loss.backward()
            optimizer.step()
        print(f"loss {loss.item()}")
        train_loss.append(loss.item())
        auc_result, test_loss = my_test(model, test_loader, loss_weights, device)
        test_loss_vec.append(test_loss)
        test_auc.append(auc_result)
    return train_loss, test_loss_vec, test_auc


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc


def my_test(model, test_set, loss_weights, device):
    """
    The test function. for every (data, traget) in our train set we calculate the output according
    to our model, calculating the loss, get the index of the max log-probability, and claculate
    "correct" variable to help us later calculate the accuracy.
    """
    model.eval()
    all_targets = []
    all_pred = []
    for data, target in test_set:
        data, target = data.to(device), target.to(device)
        output = model(data)
        output = torch.sigmoid(output)
        test_loss = F.binary_cross_entropy_with_logits(output, target.unsqueeze(dim=1).float(), weight=torch.Tensor([loss_weights[i].item() for i in target]).unsqueeze(dim=1).to(device))
        for i in target:
            all_targets.append(i.item())
        output = output.squeeze()
        for i in output:
            all_pred.append(i.item())
    auc_result = roc_auc_score(all_targets, all_pred)
    return auc_result, test_loss


def get_optimizer(RECEIVED_PARAMS, model):
    optimizer = RECEIVED_PARAMS['optimizer']
    learning_rate = RECEIVED_PARAMS['learning_rate']
    weight_decay = RECEIVED_PARAMS['regularization']
    if optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
