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

#calculate attention for all 3 heads simultaneously by taking the dot product of Q and K
scores = Q @ K.transpose(-2, -1)
print("\nRaw attention scores:")
print("Scores shape:", scores.shape)

# Scale the scores
scaled_scores = scores / math.sqrt(d_head)

# Convert scores into attention weights
attention_weights = torch.softmax(scaled_scores, dim=-1)

print("\nAttention weights shape:", attention_weights.shape)

# Use attention weights to combine the Values
attention_output = attention_weights @ V

print("Attention output shape:", attention_output.shape)

# Transpose the attention output back to (tokens, num_heads, dimensions)
attention_output = attention_output.transpose(0, 1)
#then combine
attention_output = attention_output.reshape(10, d_model)
print("\nAfter combining heads:")
print(attention_output.shape)

#add an output projection
W_O = nn.Linear(d_model, d_model, bias=False) #W_O mixes the info from the different heads
final_output = W_O(attention_output)
print("\nFinal multi-head attention output:")
print(final_output.shape)

# Residual connection
residual_output = X + final_output
print("\nAfter residual connection:")
print(residual_output.shape)

# Apply LayerNorm
layer_norm = nn.LayerNorm(d_model)
normalized_output = layer_norm(residual_output)
print("\nAfter LayerNorm:")
print(normalized_output.shape)

# Feed-Forward Network: mixes and processes info across theses representations for each token independently 
ffn = nn.Sequential(
    nn.Linear(d_model, 48),
    nn.ReLU(),
    nn.Linear(48, d_model)
)

ffn_output = ffn(normalized_output)
print("\nFFN output shape:")
print(ffn_output.shape) 

# Second residual connection
ffn_residual = normalized_output + ffn_output

# Second LayerNorm
layer_norm_2 = nn.LayerNorm(d_model)
transformer_output = layer_norm_2(ffn_residual)

print("\nFinal Transformer block output shape:")
print(transformer_output.shape)
