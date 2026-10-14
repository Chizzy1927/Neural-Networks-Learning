import torch

w = torch.tensor(3.0, requires_grad=True)

x = torch.tensor(2.0)
y = torch.tensor(10.0)

y_hat = w * x

loss = (y - y_hat) ** 2

print("Prediction:", y_hat)
print("Loss:", loss)

loss.backward()

print("Gradient:", w.grad)

with torch.no_grad():
    w -= 0.01 * w.grad
    