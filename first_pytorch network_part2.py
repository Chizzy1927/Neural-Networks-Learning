import torch
import torch.nn as nn

torch.manual_seed(42)  # For reproducibility

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 3)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


model = NeuralNetwork()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

print(model)

X = torch.tensor([
    [1.0, 2.0],
    [2.0, 1.0],
    [2.0, 3.0],
    [3.0, 2.0],
    [4.0, 1.0]
])

y = torch.tensor([
    [5.0],
    [4.0],
    [8.0],
    [7.0],
    [6.0]
])

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 1000

for epoch in range(epochs):

    # 1. Forward propagation
    predictions = model(X)

    # 2. Calculate loss
    loss = loss_function(predictions, y)

    # 3. Clear gradients from previous iteration
    optimizer.zero_grad()

    # 4. Backpropagation
    loss.backward()

    # 5. Update weights and biases
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")


# Check model after training
print("\nFinal Predictions:")

with torch.no_grad():
    predictions = model(X)
    print(predictions)

print("\nActual:")
print(y)

#trying with data it never saw suring training
new_data = torch.tensor([
    [5.0, 4.0]
])

with torch.no_grad():
    new_predictions = model(new_data)
    print("\nUnseen Data Predictions:")
    print(new_predictions)