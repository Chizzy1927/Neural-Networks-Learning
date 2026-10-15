import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


torch.manual_seed(42) #reproducibility

#make sure to download the images and transform them to tensors
transform = transforms.ToTensor()

'''train and test datasets are separate, so we need to create two datasets. We can use the MNIST dataset from torchvision.datasets,
which is a built-in dataset that contains 60,000 training images and 10,000 test images of handwritten digits (0-9).
The images are 28x28 pixels and are grayscale (1 channel).
'''
train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)


print("Training images:", len(train_dataset))
print("Test images:", len(test_dataset))

image, label = train_dataset[0] 

print("\nFirst image shape:")
print(image.shape)

print("\nFirst image label:")
print(label)

print("\nPixel range:")
print(image.min().item(), image.max().item())

#we can use a DataLoader to load the data in batches, which is more efficient than loading one image at a time. We can also shuffle the data to ensure that the model sees different examples in each epoch.
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True  #shuffling the training data is important to prevent the model from learning the order of the data and to improve generalization. i.ewe want the model to see training observations in varying orders rather than repeatedly seeing the exact same sequence.
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False #not learning from the test set, so we don't need to shuffle it.
)

images, labels = next(iter(train_loader))

print("\nBatch image shape:")
print(images.shape)

print("\nBatch label shape:")
print(labels.shape)

#now we can define a convolutional layer using nn.Conv2d. The parameters are:
conv1 = nn.Conv2d(
    in_channels=1,
    out_channels=32,
    kernel_size=3,
    padding=1
)

conv_output = conv1(images)
"""
print("\nBefore convolution:")
print(images.shape)

print("\nAfter convolution:")
print(conv_output.shape)

print("\nConv1 weight shape:")
print(conv1.weight.shape)

print("\nConv1 bias shape:")
print(conv1.bias.shape)
"""
relu = nn.ReLU()
pool = nn.MaxPool2d(kernel_size=2)

relu_output = relu(conv_output)
pooled_output = pool(relu_output)

print("\nAfter ReLU:")
print(relu_output.shape)

print("\nAfter MaxPool:")
print(pooled_output.shape)

'''neural networks are built by stacking layers on top of each other.
We can define a simple CNN architecture using nn.Module.
The architecture consists of two convolutional layers followed by two fully connected layers. 
The first convolutional layer has 32 filters, the second has 64 filters, and the fully connected layers have 128 and 10 neurons, respectively. 
The output layer has 10 neurons, one for each class (0-9).
'''
class MNISTCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)  # flatten the tensor,ONLY the first dimension (batch size) is preserved, and the rest are flattened into a single dimension.
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MNISTCNN()
logits = model(images)

#training the CNN involves defining a loss function and an optimizer. The loss function measures how well the model is performing, and the optimizer updates the model's parameters based on the gradients computed during backpropagation. In this case, we can use CrossEntropyLoss as the loss function and Adam as the optimizer.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#the training loop
epochs = 5
for epoch in range(epochs):
    model.train()  #set the model to training mode
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad() #clear old gradients fromm the previous batch.
        loss.backward()
        optimizer.step() #update the model's parameters based on the gradients computed during backpropagation.
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


#TESTING THE MODEL
model.eval()  #set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad(): #do not compute gradients during evaluation, as we are not training the model and this saves memory and computation.
    for images, labels in test_loader:
        logits = model(images)
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total
print(f"\nTest Accuracy: {test_accuracy:.4f}")