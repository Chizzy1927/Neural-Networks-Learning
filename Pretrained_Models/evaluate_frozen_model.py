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

model_path = "best_frozen_sentiment_model"


# Load the tokenizer and best saved model from training
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path
)

model.eval()


# Recreate the exact same train/test split used during training
dataset = load_dataset("nyu-mll/glue", "sst2")

split = dataset["train"].train_test_split(
    test_size=0.10,
    seed=SEED,
    stratify_by_column="label",
)

our_test = split["test"]

print(f"\nHeld-out test examples: {len(our_test)}")


def tokenize_batch(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        max_length=128,
    )


# Tokenize the test data only now that model selection is finished
test_dataset = our_test.map(
    tokenize_batch,
    batched=True,
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"],
)


# Dynamically pad each test batch
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    return_tensors="pt",
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=data_collator,
)

print(f"Test batches: {len(test_loader)}")


test_loss = 0
all_predictions = []
all_labels = []


# No gradients are needed because the model is only being evaluated
with torch.no_grad():
    for batch in test_loader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        test_loss += outputs.loss.item()

        predictions = torch.argmax(
            outputs.logits,
            dim=1,
        )

        all_predictions.extend(
            predictions.tolist()
        )

        all_labels.extend(
            batch["labels"].tolist()
        )


average_test_loss = test_loss / len(test_loader)


# Calculate metrics across all held-out test examples
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


print("\nFinal Frozen Model Test Results")
print("--------------------------------")
print(f"Test Loss: {average_test_loss:.4f}")
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")

print("\nConfusion Matrix:")
print(matrix)