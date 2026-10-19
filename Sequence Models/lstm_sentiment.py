import torch
import torch.nn as nn

torch.manual_seed(42)

# Tiny vocabulary
vocab = {
    "<PAD>": 0,
    "this": 1,
    "movie": 2,
    "was": 3,
    "amazing": 4,
    "terrible": 5,
    "great": 6,
    "awful": 7
}

# Example sentences already converted to token IDs
X = torch.tensor([
    [1, 2, 3, 4],   # this movie was amazing
    [1, 2, 3, 5],   # this movie was terrible
    [1, 2, 3, 6],   # this movie was great
    [1, 2, 3, 7]    # this movie was awful
])

# 1 = positive
# 0 = negative
y = torch.tensor([
    1,
    0,
    1,
    0
])

print("X shape:", X.shape)
print("y shape:", y.shape)

#class for embedding, lstm, and linear layer
class SentimentLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=8,
            embedding_dim=5
        )

        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=8,
            batch_first=True
        )

        self.classifier = nn.Linear(8, 2)

    def forward(self, x):

        # (batch, sequence)
        x = self.embedding(x)

        # (batch, sequence, 5)
        output, (h_n, c_n) = self.lstm(x)

        # h_n = (1, batch, 8)
        final_hidden = h_n[-1]

        # (batch, 8) -> (batch, 2)
        logits = self.classifier(final_hidden)

        return logits


#train the model
model = SentimentLSTM()

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 300

for epoch in range(epochs):

    # Forward pass
    logits = model(X)

    # Calculate loss
    loss = loss_function(logits, y)

    # Clear old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    if epoch % 50 == 0:
        predictions = torch.argmax(logits, dim=1)

        accuracy = (
            (predictions == y)
            .float()
            .mean()
        )

        print(
            f"Epoch {epoch}: "
            f"Loss = {loss.item():.4f}, "
            f"Accuracy = {accuracy.item():.4f}"
        )

model.eval()

with torch.no_grad():
    logits = model(X)

    probabilities = torch.softmax(logits, dim=1)

    predictions = torch.argmax(logits, dim=1)

print("\nFinal predictions:")

for i in range(len(X)):
    print(
        f"Sentence {i + 1}: "
        f"Actual = {y[i].item()}, "
        f"Predicted = {predictions[i].item()}, "
        f"Probabilities = {probabilities[i]}"
    )