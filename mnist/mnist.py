import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math


class MnistMLP(nn.Module):

    def __init__(self, in_features, hidden_size, n_hidden, out_features):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc_in = nn.Linear(in_features=in_features,
                               out_features=hidden_size)
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_hidden):
            self.hidden_layers.append(
                nn.Linear(in_features=hidden_size, out_features=hidden_size))

        self.fc_out = nn.Linear(in_features=hidden_size,
                                out_features=out_features)

    def forward(self, x):
        x = self.relu(self.fc_in(x))

        for layer in self.hidden_layers:
            x = self.relu(layer(x))

        x = self.fc_out(x)
        return x


class MnistConvNet(nn.Module):

    def __init__(self, in_channels, in_height, in_width, out_features):
        super().__init__()

        # (H, W)
        conv_kernel_size = (4, 4)
        pool_kernel_size = (2, 2)
        padding = (1, 1)
        conv_stride = (1, 1)
        pool_stride = (2, 2)

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=12,
                               kernel_size=conv_kernel_size,
                               padding=padding,
                               stride=conv_stride)
        H_out1 = math.floor((in_width - conv_kernel_size[0] + 2 * padding[0]) /
                            conv_stride[0] + 1)
        W_out1 = math.floor((in_width - conv_kernel_size[1] + 2 * padding[1]) /
                            conv_stride[1] + 1)

        self.pool1 = nn.MaxPool2d(kernel_size=pool_kernel_size,
                                  stride=pool_stride)
        H_out2 = math.floor((H_out1 - pool_kernel_size[0]) / pool_stride[0] +
                            1)
        W_out2 = math.floor((W_out1 - pool_kernel_size[1]) / pool_stride[1] +
                            1)

        self.conv2 = nn.Conv2d(in_channels=12,
                               out_channels=32,
                               kernel_size=conv_kernel_size,
                               padding=padding,
                               stride=conv_stride)
        H_out3 = math.floor((H_out2 - conv_kernel_size[0] + 2 * padding[0]) /
                            conv_stride[0] + 1)
        W_out3 = math.floor((W_out2 - conv_kernel_size[1] + 2 * padding[1]) /
                            conv_stride[1] + 1)

        self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel_size,
                                  stride=pool_stride)
        H_out4 = math.floor((H_out3 - pool_kernel_size[0]) / pool_stride[0] +
                            1)
        W_out4 = math.floor((W_out3 - pool_kernel_size[1]) / pool_stride[1] +
                            1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=32 * H_out4 * W_out4, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True)

trainloader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=50,
                                          shuffle=True,
                                          num_workers=2)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True)

testloader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=4,
                                         shuffle=True,
                                         num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

size = (28, 28)
num_classes = 10

mlp_model = MnistMLP(in_features=size[0] * size[1],
                     hidden_size=512,
                     n_hidden=5,
                     out_features=num_classes)

conv_model = MnistConvNet(in_channels=1,
                          in_height=size[0],
                          in_width=size[1],
                          out_features=10)

model = conv_model
model = model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.03)

epochs = 3

# Training loop
model.train()
for epoch in range(epochs):
    running_loss = 0
    total = 0
    for i, data in enumerate(trainloader):
        X, Y = data
        X = X.to(device)
        # X = X.view(-1, X.shape[2] * X.shape[3]) # For use with MLP
        Y = F.one_hot(Y, num_classes=num_classes).float().to(device)

        optimizer.zero_grad(set_to_none=True)
        y = model(X)

        loss = criterion(y, Y)

        running_loss += loss.item()
        total += 1

        loss.backward()
        optimizer.step()
    print(epoch, 'current loss:', running_loss / total)

# Testing loop
model.eval()

total = 0
correct = 0
for i, data in enumerate(trainloader):
    X, Y = data
    X = X.to(device)
    Y = Y.to(device)
    # X = X.view(-1, X.shape[2] * X.shape[3]) # For use with MLP

    y = model(X)
    res = torch.argmax(y, dim=1)
    equal = torch.eq(res, Y)
    n_correct = torch.sum(equal)

    correct += n_correct
    total += Y.shape[0]

print(f'Accuracy: {correct} / {total}')
