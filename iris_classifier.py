import torch
import torch.nn as nn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

epochs = 500

X = iris.data
y = iris.target

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst flower:")
print(X[0])

print("\nFirst label:")
print(y[0])

print("\nClasses:")
print(iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining X:", X_train.shape)
print("Testing X:", X_test.shape)

print("Training y:", y_train.shape)
print("Testing y:", y_test.shape)

#--------BUILDING THE CLASSIFIER------
#Convert numpy arrays to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
#long because we are doing classification, not regression. they are class id's not measurements.
y_train = torch.tensor(y_train, dtype=torch.long) 
y_test = torch.tensor(y_test, dtype=torch.long)

#the actual network
class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 16) #4 input features, 16 hidden neurons
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 3) #16 hidden units, 3 output class scores

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


torch.manual_seed(42)

model = IrisClassifier()
print("\nMODEL:")
print(model)

'''torch.no_grad() is used to disable gradient calculation, which reduces memory consumption
and speeds up computations. It is useful when you are sure that you will not backpropagate. 
we are simply looking at predictions here, not training the model.
'''

with torch.no_grad():
    logits = model(X_test)

print("\nLogits for first 5 test flowers:")
print(logits[:5])

predictions = torch.argmax(logits, dim=1)
print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y_test)

#checking accuracy
accuracy = (predictions == y_test).float().mean()

print("\nUntrained accuracy:")
print(accuracy.item())

#training the model
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(epochs):
    #forward pass
    logits = model(X_train)
    #calculate loss
    loss = loss_fn(logits, y_train)
    #clear old gradients
    optimizer.zero_grad()
     #backpropagation
    loss.backward()
    #update weights
    optimizer.step()

    if epoch % 50 == 0:
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y_train).float().mean()
        print(f"Epoch {epoch}: Loss = {loss.item()}, Accuracy = {accuracy.item()}")

#the model is trained, let's check the accuracy on the test set
with torch.no_grad():

    test_logits = model(X_test)

    test_predictions = torch.argmax(
        test_logits,
        dim=1
    )

    test_accuracy = (
        (test_predictions == y_test)
        .float()
        .mean()
    )

print("\nFINAL TEST RESULTS")

print("Predictions:")
print(test_predictions)

print("\nActual:")
print(y_test)

print("\nTest Accuracy:")
print(test_accuracy.item())

'''
note: it is important to understand that the model is not perfect.
having a 100% accuracy does not mean that the model will classify every Iris flower on earth.
it means that on this particular dataset, the model correctly classifies all 30 observations.'''

with torch.no_grad():
    test_logits = model(X_test)

    probabilities = torch.softmax(
        test_logits,
        dim=1
    )

print("\nFirst 5 logits:")
print(test_logits[:5])

print("\nFirst 5 probabilities:")
print(probabilities[:5])

confidence, predicted_class = torch.max(
    probabilities,
    dim=1
)

least_confident_index = torch.argmin(confidence)

print("\nLEAST CONFIDENT PREDICTION")

print("Flower measurements:")
print(X_test[least_confident_index])

print("Probabilities:")
print(probabilities[least_confident_index])

print("Predicted class:")
print(predicted_class[least_confident_index].item())

print("Actual class:")
print(y_test[least_confident_index].item())

print("Confidence:")
print(confidence[least_confident_index].item())