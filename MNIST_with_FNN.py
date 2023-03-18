import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the training function
def train(net, train_loader,test_loader, optimizer, criterion, epochs):
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss/len(train_loader))
        test_loss = test(net, test_loader, criterion)
        test_losses.append(test_loss)
        print('Epoch %d: Training Loss: %.3f, Test Loss: %.3f' % (epoch+1, train_losses[-1], test_loss))
    return train_losses, test_losses

# Define the test function
def test(net, test_loader, criterion):
    running_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss/len(test_loader)

# Define the main function
def main():
    # Load the data
    train_data = torch.randn(100, 10)
    train_labels = torch.randn(100, 1)
    test_data = torch.randn(50, 10)
    test_labels = torch.randn(50, 1)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Initialize the network, optimizer, and loss function
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    # Train the network and get the training and test losses
    train_losses, test_losses = train(net, train_loader,test_loader, optimizer, criterion, epochs=100)

    # Plot the training and test losses
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(test_losses)), test_losses, label='Test Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
