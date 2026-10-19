import torch
import torch.nn as nn
import math

torch.manual_seed(42)

# Tiny vocabulary, 4 tokens represented by 6 dimensional embeddings
X = torch.randn(4, 6)
print("X shape:", X.shape)
print(X)

#three learned transformations for query, key, and value
W_Q = nn.Linear(6, 3, bias=False)
W_K = nn.Linear(6, 3, bias=False)
W_V = nn.Linear(6, 3, bias=False)

Q = W_Q(X)
K = W_K(X)
V = W_V(X)

print("\nQ shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

#calculate attention scores by taking the dot product of Q and K
scores = Q @ K.T
print("\nRaw attention scores:")
print(scores)
print("\nScores shape:", scores.shape)

#scale the attention scores by dividing by the square root of the dimension of K
scaled_scores = scores / math.sqrt(K.shape[-1])
print("\nScaled attention scores:")
print(scaled_scores)

#apply softmax to the scaled scores to get attention weights
attention_weights = torch.softmax(scaled_scores, dim=-1)
print("\nAttention weights:")
print(attention_weights)
print("\nEach row sums to:")
print(attention_weights.sum(dim=-1)) #batch dimension

#compute the final attention output by multiplying the attention weights with the value matrix V
attention_output = attention_weights @ V
print("\nV:")
print(V)

print("\nFinal attention output:")
print(attention_output)

print("\nFinal output shape:")
print(attention_output.shape)