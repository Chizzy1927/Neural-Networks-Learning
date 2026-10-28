import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.manual_seed(42)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Small balanced dataset for learning the fine-tuning workflow
texts = [
    "I absolutely loved this movie.",
    "The acting was fantastic.",
    "This was one of the best films I have seen.",
    "I really enjoyed the story.",
    "The movie was exciting and entertaining.",
    "I hated this movie.",
    "This was painfully boring.",
    "The acting was terrible.",
    "I regret watching this film.",
    "The story was awful and predictable.",
]

# Positive = 1, negative = 0
labels = [
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
]


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        # Tokenize each review and pad it to a consistent sequence length
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Remove the temporary batch dimension added by the tokenizer
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


dataset = SentimentDataset(texts, labels, tokenizer)

# Shuffle the examples so batches change between epochs
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
)

batch = next(iter(dataloader))

print("Input IDs shape:", batch["input_ids"].shape)
print("Attention mask shape:", batch["attention_mask"].shape)
print("Labels shape:", batch["label"].shape)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
)

# Keep the pretrained language features fixed while we train the classifier
for param in model.distilbert.parameters():
    param.requires_grad = False 

trainable_params = sum(
    param.numel()
    for param in model.parameters()
    if param.requires_grad
)

total_params = sum(
    param.numel()
    for param in model.parameters()
)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

#trainimng the classifier now
optimizer = torch.optim.AdamW(
    filter(lambda param: param.requires_grad, model.parameters()),
    lr=5e-4,
)

epochs = 5

model.train()

for epoch in range(epochs):
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- Loss: {average_loss:.4f}"
    )

    # Evaluate on sentences the classifier did not see during training
test_texts = [
    "I really enjoyed this film.",
    "That was a complete waste of time.",
    "The movie was surprisingly good.",
    "I could barely stay awake.",
]

model.eval()

for text in test_texts:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )

    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0, predicted_class].item()

    label = "POSITIVE" if predicted_class == 1 else "NEGATIVE"

    print(f"\nText: {text}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")