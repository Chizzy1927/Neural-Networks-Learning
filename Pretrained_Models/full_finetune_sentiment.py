import time

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

SEED = 42
torch.manual_seed(SEED)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Load SST-2 sentiment data
dataset = load_dataset("nyu-mll/glue", "sst2")

print("\nOriginal dataset:")
print(dataset)


# Keep 10% of the labeled training data untouched for final testing
split = dataset["train"].train_test_split(
    test_size=0.10,
    seed=SEED,
    stratify_by_column="label",
)

full_train = split["train"]
our_test = split["test"]
validation = dataset["validation"]

print("\nDataset sizes:")
print("Training:", len(full_train))
print("Validation:", len(validation))
print("Our test:", len(our_test))
print("Official test:", len(dataset["test"]))


# Check that our train/test split kept the original class balance
def show_label_distribution(name, data):
    labels = data["label"]

    positive = sum(label == 1 for label in labels)
    negative = sum(label == 0 for label in labels)
    total = len(labels)

    print(f"\n{name}:")
    print(f"Positive: {positive} ({positive / total:.2%})")
    print(f"Negative: {negative} ({negative / total:.2%})")


show_label_distribution("Training", full_train)
show_label_distribution("Our Test", our_test)
show_label_distribution("Validation", validation)


def tokenize_batch(batch):
    # Padding happens later so each batch only pads as much as necessary
    return tokenizer(
        batch["sentence"],
        truncation=True,
        max_length=128,
    )


print("\nTokenizing datasets...")

train_dataset = full_train.map(
    tokenize_batch,
    batched=True,
)

validation_dataset = validation.map(
    tokenize_batch,
    batched=True,
)

# Dynamically pad each batch to its longest sentence
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    return_tensors="pt",
)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"],
)

validation_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"],
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=data_collator,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=data_collator,
)


# Check one real batch before training
batch = next(iter(train_loader))

print("\nReal training batch:")
print("Input IDs:", batch["input_ids"].shape)
print("Attention mask:", batch["attention_mask"].shape)
print("Labels:", batch["labels"].shape)

print("\nNumber of training batches:", len(train_loader))
print("Number of validation batches:", len(validation_loader))


# Load DistilBERT with a new sentiment classification head
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
)

trainable_params = sum(
    param.numel()
    for param in model.parameters()
    if param.requires_grad
)

total_params = sum(
    param.numel()
    for param in model.parameters()
)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(
    f"Trainable percentage: "
    f"{100 * trainable_params / total_params:.2f}%"
)


# Fine-tune all DistilBERT parameters using a small learning rate
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
)


epochs = 3
best_validation_accuracy = 0.0
best_model_path = "best_full_sentiment_model"

start_time = time.time()

for epoch in range(epochs):
    model.train()
    total_train_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = total_train_loss / len(train_loader)


    # Evaluate on data the model did not train on
    model.eval()
    total_validation_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in validation_loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            total_validation_loss += outputs.loss.item()

            predictions = torch.argmax(
                outputs.logits,
                dim=1,
            )

            correct += (
                predictions == batch["labels"]
            ).sum().item()

            total += batch["labels"].size(0)

    average_validation_loss = (
        total_validation_loss / len(validation_loader)
    )

    validation_accuracy = correct / total

    print(
        f"\nEpoch {epoch + 1}/{epochs}"
        f"\nTrain Loss: {average_train_loss:.4f}"
        f"\nValidation Loss: {average_validation_loss:.4f}"
        f"\nValidation Accuracy: {validation_accuracy:.2%}"
    )


    # Keep whichever epoch generalizes best to the validation data
    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy

        model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)

        print("Saved new best model.")


training_time = time.time() - start_time

print("\nTraining complete.")
print(f"Best Validation Accuracy: {best_validation_accuracy:.2%}")
print(f"Training Time: {training_time / 60:.2f} minutes")

# Final evaluation of the best full fine-tuned checkpoint
print("\nPreparing held-out test set...")

test_dataset = our_test.map(
    tokenize_batch,
    batched=True,
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"],
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=data_collator,
)

print("Test examples:", len(test_dataset))
print("Test batches:", len(test_loader))

# Load the checkpoint that achieved the best validation accuracy
best_model = AutoModelForSequenceClassification.from_pretrained(
    best_model_path
)

best_model.eval()
test_loss = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        outputs = best_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        test_loss += outputs.loss.item()

        predictions = torch.argmax(
            outputs.logits,
            dim=1,
        )

        all_predictions.extend(predictions.tolist())
        all_labels.extend(batch["labels"].tolist())

average_test_loss = test_loss / len(test_loader)
# Calculate final test metrics from all 6,735 predictions
accuracy = accuracy_score(
    all_labels,
    all_predictions,
)

precision = precision_score(
    all_labels,
    all_predictions,
)

recall = recall_score(
    all_labels,
    all_predictions,
)

f1 = f1_score(
    all_labels,
    all_predictions,
)

matrix = confusion_matrix(
    all_labels,
    all_predictions,
)

print("\nFinal Full Fine-tuned Model Test Results")
print("--------------------------------")
print(f"Test Loss: {average_test_loss:.4f}")
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")

print("\nConfusion Matrix:")
print(matrix)