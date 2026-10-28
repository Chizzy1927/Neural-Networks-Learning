from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name)

model.eval()

text = "This movie was absolutely fantastic."

encoded = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**encoded)

logits = outputs.logits

print("Logits:")
print(logits)

print("\nLogits shape:")
print(logits.shape)

probabilities = torch.softmax(logits, dim=1)

print("\nProbabilities:")
print(probabilities)

predicted_class = torch.argmax(probabilities, dim=1)

print("\nPredicted class:")
print(predicted_class)
label = model.config.id2label[predicted_class.item()]

print("\nPrediction:")
print(label)
confidence = probabilities[0, predicted_class.item()].item()
print(f"Confidence: {confidence * 100:.2f}%")

