import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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

#104 images were misclassified out of 10,000 test images, which is a test accuracy of 98.96%. inspecting.
wrong_images = []
wrong_predictions = []
wrong_labels = []
wrong_confidences = []

model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        logits = model(images)

        probabilities = torch.softmax(logits, dim=1)

        confidence, predictions = torch.max(
            probabilities,
            dim=1
        )

        wrong_mask = predictions != labels

        wrong_images.append(images[wrong_mask])
        wrong_predictions.append(predictions[wrong_mask])
        wrong_labels.append(labels[wrong_mask])
        wrong_confidences.append(confidence[wrong_mask])


wrong_images = torch.cat(wrong_images)
wrong_predictions = torch.cat(wrong_predictions)
wrong_labels = torch.cat(wrong_labels)
wrong_confidences = torch.cat(wrong_confidences)

print("\nNumber of mistakes:", len(wrong_labels))

print("\nFirst 10 mistakes:")

for i in range(min(10, len(wrong_labels))):

    print(
        f"Actual: {wrong_labels[i].item()}, "
        f"Predicted: {wrong_predictions[i].item()}, "
        f"Confidence: {wrong_confidences[i].item():.4f}"
    )

# Plot some of the wrong predictions. some were confidently wrong, needs human inspection. perhaps it makes sense why the system might confuse some handwritten digits.
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):

    image = wrong_images[i].squeeze()

    ax.imshow(image, cmap="gray")

    ax.set_title(
        f"Actual: {wrong_labels[i].item()}\n"
        f"Pred: {wrong_predictions[i].item()}\n"
        f"Conf: {wrong_confidences[i].item():.2%}"
    )

    ax.axis("off")

plt.tight_layout()
plt.show()

#checking out the most confidently wrong predictions. 
sorted_indices = torch.argsort(
    wrong_confidences,
    descending=True
)

print("\nMOST CONFIDENTLY WRONG:")

for i in range(10):

    index = sorted_indices[i]

    print(
        f"Actual: {wrong_labels[index].item()}, "
        f"Predicted: {wrong_predictions[index].item()}, "
        f"Confidence: {wrong_confidences[index].item():.4f}"
    )

#view em    
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):

    index = sorted_indices[i]

    image = wrong_images[index].squeeze()

    ax.imshow(image, cmap="gray")

    ax.set_title(
        f"Actual: {wrong_labels[index].item()}\n"
        f"Pred: {wrong_predictions[index].item()}\n"
        f"Conf: {wrong_confidences[index].item():.2%}"
    )

    ax.axis("off")

plt.tight_layout()
plt.show()

#opening up the model to see what the first convolutional layer is doing. we can visualize the feature maps produced by the first convolutional layer for a sample image from the test dataset. This can help us understand what features the model is learning to detect in the input images.
sample_image, sample_label = test_dataset[0]

print(sample_image.shape)
print(sample_label)
sample_batch = sample_image.unsqueeze(0)
with torch.no_grad():

    conv1_output = model.conv1(sample_batch)
    conv1_activated = model.relu(conv1_output)

print("Original image:", sample_batch.shape)
print("Conv1 feature maps:", conv1_activated.shape)
fig, axes = plt.subplots(4, 8, figsize=(14, 7))

for i, ax in enumerate(axes.flat):

    feature_map = conv1_activated[0, i]

    ax.imshow(feature_map, cmap="gray")
    ax.set_title(f"Filter {i + 1}")
    ax.axis("off")

plt.tight_layout()
plt.show()

#look into the second convolutional layer. 
with torch.no_grad():

    x = model.conv1(sample_batch)
    x = model.relu(x)
    x = model.pool(x)

    conv2_output = model.conv2(x)
    conv2_activated = model.relu(conv2_output)

print("Input to Conv2:", x.shape)
print("Conv2 feature maps:", conv2_activated.shape)
fig, axes = plt.subplots(8, 8, figsize=(12, 12))

for i, ax in enumerate(axes.flat):

    feature_map = conv2_activated[0, i]

    ax.imshow(feature_map, cmap="gray")
    ax.set_title(f"{i + 1}")
    ax.axis("off")

plt.tight_layout()
plt.show()