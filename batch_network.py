import numpy as np

# -----------------------
# DATA
# -----------------------

X = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [2.0, 3.0],
    [3.0, 2.0],
    [4.0, 1.0]
])

y = np.array([5.0, 4.0, 8.0, 7.0, 6.0])

n = len(X)

W1 = np.array([
    [0.2,  0.4],
    [-0.5, 0.3],
    [0.8, -0.1]
])

b1 = np.array([0.1, 0.2, -0.1])

W2 = np.array([0.5, -1.0, 2.0])
b2 = 0.5

learning_rate = 0.001
epochs = 5000

for epoch in range(epochs):

    # =====================
    # FORWARD PASS
    # =====================

    z1 = X @ W1.T + b1

    h = np.maximum(0, z1)

    y_hat = h @ W2 + b2

    loss = np.mean((y - y_hat) ** 2)


    # =====================
    # BACKPROPAGATION
    # =====================

    dy_hat = (-2 / n) * (y - y_hat)

    # Output layer
    dW2 = h.T @ dy_hat
    db2 = np.sum(dy_hat)

    # Send gradient into hidden layer
    dh = np.outer(dy_hat, W2)

    # ReLU
    dz1 = dh * (z1 > 0)

    # First layer
    dW1 = dz1.T @ X
    db1 = np.sum(dz1, axis=0)


    # =====================
    # GRADIENT DESCENT
    # =====================

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2


    if epoch % 500 == 0:
        print(
            f"Epoch {epoch}: "
            f"Loss={loss:.6f}"
        )

print("\nFinal predictions:")
print(y_hat)

print("\nActual values:")
print(y)

print("\nFinal loss:")
print(loss)    

new_x = np.array([5.0, 4.0])

new_z1 = W1 @ new_x + b1
new_h = np.maximum(0, new_z1)
new_prediction = W2 @ new_h + b2

print("\nUnseen input:", new_x)
print("Network prediction:", new_prediction)
print("Correct answer: 13")
