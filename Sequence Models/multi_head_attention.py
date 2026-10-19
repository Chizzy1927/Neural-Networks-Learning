import torch
import torch.nn as nn
import math

torch.manual_seed(42)

# 10 tokens, each represented by 12 numbers
X = torch.randn(10, 12)

d_model = 12
num_heads = 3
d_head = d_model // num_heads
print("X shape:", X.shape)
print("d_head:", d_head)

#three learned transformations for query, key, and value
W_Q = nn.Linear(d_model, d_model, bias=False)
W_K = nn.Linear(d_model, d_model, bias=False)
W_V = nn.Linear(d_model, d_model, bias=False)

Q = W_Q(X)
K = W_K(X)
V = W_V(X)

print("\nBefore splitting:")
print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)

#split the Q, K, and V matrices into multiple heads
Q = Q.reshape(10, num_heads, d_head)
K = K.reshape(10, num_heads, d_head)
V = V.reshape(10, num_heads, d_head)
print("\nAfter splitting:")
print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)

#transpose the Q, K, and V matrices to have the shape (num_heads, tokens, dimensions)
Q = Q.transpose(0, 1)
K = K.transpose(0, 1)
V = V.transpose(0, 1)
print("\nAfter transposing:")
print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)

#
scores = Q @ K.transpose(-2, -1)