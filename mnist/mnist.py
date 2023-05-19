import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision


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


train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True)

trainloader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=24,
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

size = (28, 28)
num_classes = 10

model = MnistMLP(in_features=size[0] * size[1],
                 hidden_size=512,
                 n_hidden=5,
                 out_features=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.03)

epochs = 3

# Training loop
model.train()
for epoch in range(epochs):
    running_loss = 0
    total = 0
    for i, data in enumerate(trainloader):
        X, Y = data
        X = X.view(-1, X.shape[2] * X.shape[3])
        Y = F.one_hot(Y, num_classes=num_classes).float()

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
    X = X.view(-1, X.shape[2] * X.shape[3])

    y = model(X)
    res = torch.argmax(y, dim=1)
    equal = torch.eq(res, Y)
    n_correct = torch.sum(equal)

    correct += n_correct
    total += Y.shape[0]

print(f'Accuracy: {correct} / {total}')
