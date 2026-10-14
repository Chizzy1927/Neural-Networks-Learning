import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 3)  # Two inputs and three outputs
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(3, 1)  # Three inputs and one output


    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


model = NeuralNetwork()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
print(model) 

'''
for name, parameter in model.named_parameters():
    print(name)
    print(parameter)
    '''

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

predictions = model(X)

print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y)

loss_function = nn.MSELoss()

loss = loss_function(predictions, y)

print("\nLoss:")
print(loss)

loss.backward()

for name, parameter in model.named_parameters():
    print(name)
    print(parameter.grad)


'''encountered dead RELU PROBLEM with the random initialization of the weights. The weights were initialized in such a way that the output of the first layer was always negative, which caused the ReLU activation function to output zero for all inputs. This resulted in no gradient being propagated back through the network, effectively "killing" the neurons in the first layer and preventing learning.
ayer1.weight
tensor([[ 0.0000,  0.0000],
        [ 0.0000,  0.0000],
        [-9.4584, -6.8659]])
layer1.bias
tensor([ 0.0000,  0.0000, -3.4709])
layer2.weight
tensor([[ 0.0000,  0.0000, -7.6294]])
layer2.bias
tensor([-10.2263])
'''

