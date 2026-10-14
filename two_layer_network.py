import numpy as np

# One training example
x = np.array([2.0, 4.0])
y = 5.0

# First layer: 2 inputs -> 3 neurons
W1 = np.array([
    [0.2,  0.4],
    [-0.5, 0.3],
    [0.8, -0.1]
])

b1 = np.array([0.1, 0.2, -0.1])

# Second layer: 3 neurons -> 1 output
W2 = np.array([0.5, -1.0, 2.0])

b2 = 0.5

learning_rate = 0.001
epochs =1000

for epoch in range(epochs):
    # Hidden layer
    z1 = W1 @ x + b1

    # ReLU
    h = np.maximum(0, z1)
    
    # Output layer
    y_hat = W2 @ h + b2
    
    # Loss
    loss = (y - y_hat) ** 2

    '''
    print("z1:", z1)
    print("h:", h)
    print("Prediction:", y_hat)
    print("Loss:", loss)
    '''
    # ----- BACKPROPAGATION -----

    # Loss -> prediction
    dy_hat = -2 * (y - y_hat)
    
    # Output layer gradients
    dW2 = dy_hat * h
    db2 = dy_hat
    
    # Send gradient backward into hidden layer
    dh = dy_hat * W2
    
    # Backward through ReLU
    dz1 = dh * (z1 > 0)
    
    # First layer gradients
    dW1 = np.outer(dz1, x)
    db1 = dz1
    
    '''
    print("\ndy_hat:", dy_hat)
    print("dW2:", dW2)
    print("db2:", db2)
    print("dh:", dh)
    print("dz1:", dz1)
    print("dW1:")
    print(dW1)
    print("db1:", db1)
    '''

    #gradient descent update
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    #occasionally print the loss and parameters
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: "
              f"Prediction = {y_hat: .6f}, "
              f"W1 = {W1}, "
              f"b1 = {b1}, "
              f"W2 = {W2}, "
              f"b2 = {b2}"
              f"Loss = {loss: .6f}"
        )