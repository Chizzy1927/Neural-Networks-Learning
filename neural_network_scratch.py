import numpy as np

#data
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

#starting parameters
w = 0.5
b = 0.0

#hyperparameters
learning_rate = 0.01
epochs = 1000

#forward pass
for epoch in range(epochs):
    y_hat = w * X + b
    
    #mean squared error loss
    loss = np.mean((y - y_hat) ** 2)

    #deriving the gradients,backpropagation i.e. the partial derivatives of the loss with respect to w and b so we can learn the parameters 
    dw = -2 * np.mean(X * (y - y_hat))
    db = -2 * np.mean(y - y_hat)

    #gradient descent update
    w = w - learning_rate * dw
    b = b - learning_rate * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: "
              f"Loss = {loss: .6f}, "
              f"w = {w:.4f}, "
              f"b = {b:.4f}")
        print("Predictions:", y_hat)