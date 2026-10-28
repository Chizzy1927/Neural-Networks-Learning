import time

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
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

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
MAX_LENGTH = 128

# This is only a short test to make sure LoRA is learning correctly.
EPOCHS = 3
LORA_LEARNING_RATE = 1e-4
BEST_MODEL_PATH = "best_lora_sentiment_model"


# Load the tokenizer used by DistilBERT.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Load SST-2 sentiment data.
dataset = load_dataset("nyu-mll/glue", "sst2")

print("\nOriginal dataset:")
print(dataset)


# Reserve 10% of the labeled training data for our own test set.
# Using the same seed and stratification recreates the exact split
# used in our frozen and full fine-tuning experiments.
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


# Check that positive and negative reviews remain properly distributed.
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


# Convert each review into DistilBERT token IDs.
# Padding is delayed until batching so short reviews do not waste computation.
def tokenize_batch(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        max_length=MAX_LENGTH,
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


# Pad each batch only to the longest sequence inside that batch.
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
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
)


# Inspect one batch before training.
batch = next(iter(train_loader))

print("\nReal training batch:")
print("Input IDs:", batch["input_ids"].shape)
print("Attention mask:", batch["attention_mask"].shape)
print("Labels:", batch["labels"].shape)

print("\nFull training batches:", len(train_loader))
print("Validation batches:", len(validation_loader))


# Start with the original pretrained DistilBERT model.
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)


# LoRA keeps DistilBERT's original weights frozen and adds small trainable
# adapters to the query and value projections in each attention block.
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    bias="none",

    # The sentiment classifier is new, so it must also remain trainable.
    modules_to_save=["pre_classifier", "classifier"],
)


# Inject the LoRA adapters into DistilBERT.
model = get_peft_model(
    base_model,
    lora_config,
)


# Confirm that only a small fraction of the model will be updated.
model.print_trainable_parameters()

total_params = sum(
    param.numel()
    for param in model.parameters()
)

trainable_params = sum(
    param.numel()
    for param in model.parameters()
    if param.requires_grad
)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(
    f"Trainable percentage: "
    f"{100 * trainable_params / total_params:.2f}%"
)


# Only LoRA adapters and the classifier are trainable.
# We can use a larger learning rate than full fine-tuning because the
# original pretrained DistilBERT weights are not being modified.
optimizer = torch.optim.AdamW(
    filter(
        lambda param: param.requires_grad,
        model.parameters(),
    ),
    lr=LORA_LEARNING_RATE,
)


best_validation_accuracy = 0.0
start_time = time.time()


for epoch in range(EPOCHS):

    # Training mode enables dropout and prepares trainable layers for learning.
    model.train()
    total_train_loss = 0.0
    batches_processed = 0

    for batch in train_loader:
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        loss = outputs.loss

        # Calculate gradients for the LoRA adapters and classifier.
        loss.backward()

        # Update only the trainable parameters.
        optimizer.step()

        total_train_loss += loss.item()
        batches_processed += 1


    average_train_loss = total_train_loss / len(train_loader)

    # Validation checks whether LoRA is learning patterns that generalize
    # beyond the reviews it just trained on.
    model.eval()

    total_validation_loss = 0.0
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
        f"\nEpoch {epoch + 1}/{EPOCHS}"
        f"\nTrain Loss: {average_train_loss:.4f}"
        f"\nValidation Loss: {average_validation_loss:.4f}"
        f"\nValidation Accuracy: {validation_accuracy:.2%}"
    )


    # Save the LoRA checkpoint only when validation performance improves.
    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        model.save_pretrained(BEST_MODEL_PATH)
        tokenizer.save_pretrained(BEST_MODEL_PATH)

        print("Saved new best LoRA model.")


training_time = time.time() - start_time

print("\nLoRA training complete.")
print(f"Best Validation Accuracy: {best_validation_accuracy:.2%}")
print(f"Training Time: {training_time / 60:.2f} minutes")


# ------------------------------------------------------------
# TEST EVALUATION
#
# We are evaluating the sanity model here for learning/debugging.
# For the final experiment, model decisions should be based on
# validation performance rather than repeatedly tuning against test.
# ------------------------------------------------------------

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
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
)

print("Test examples:", len(test_dataset))
print("Test batches:", len(test_loader))


# A PEFT checkpoint stores the LoRA adapters rather than another complete
# copy of all the original DistilBERT weights. Recreate the base model first.
test_base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

# Attach our saved LoRA adapters and saved classification layers.
best_model = PeftModel.from_pretrained(
    test_base_model,
    BEST_MODEL_PATH,
)

best_model.eval()

test_loss = 0.0
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

        all_predictions.extend(
            predictions.tolist()
        )

        all_labels.extend(
            batch["labels"].tolist()
        )


average_test_loss = (
    test_loss / len(test_loader)
)


# Calculate classification metrics across all held-out examples.
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


print("\n Final LoRA Model Test Results")
print("--------------------------------")
print(f"Test Loss: {average_test_loss:.4f}")
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")

print("\nConfusion Matrix:")
print(matrix)