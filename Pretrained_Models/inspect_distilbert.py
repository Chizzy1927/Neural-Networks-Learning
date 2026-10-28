from transformers import AutoModelForSequenceClassification

model_name = "distilbert-base-uncased"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
)

# Show the names of the model's linear layers
for name, module in model.named_modules():
    if module.__class__.__name__ == "Linear":
        print(name)