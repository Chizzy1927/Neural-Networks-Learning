# Import the automatic tokenizer loader from Hugging Face
from transformers import AutoTokenizer,AutoModel
import torch
from transformers import AutoModelForSequenceClassification

# 1.Name of the pretrained model whose tokenizer we want to use
model_name = "distilbert-base-uncased"

# Download/load the tokenizer that belongs to DistilBERT
# After the first download, Hugging Face caches it locally
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. TOKENIZE A NORMAL SENTENCE
text = "Neural networks are incredibly powerful."

# Break the sentence into tokens
tokens = tokenizer.tokenize(text)

# Convert each token into its vocabulary ID
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Original text:")
print(text)

print("\nTokens:")
print(tokens)

print("\nToken IDs:")
print(token_ids)

# 3. sUBWORD TOKENIZATION
weird_text = "The debugger was debugging unsuccessfully."

# Some words may be broken into smaller subword pieces
weird_tokens = tokenizer.tokenize(weird_text)

# Convert those subword tokens into IDs
weird_ids = tokenizer.convert_tokens_to_ids(weird_tokens)

print("\nSecond sentence:")
print(weird_text)

print("\nSubword tokens:")
print(weird_tokens)

print("\nSubword IDs:")
print(weird_ids)

#4.Instead of manually tokenizing and converting to IDs, we can let the tokenizer prepare everything automatically.
encoded = tokenizer(
    text,
    return_tensors="pt"
)
print("\nEncoded model input:")
print(encoded)

# 5. input_ids contains the numerical token IDs that will actually be sent into the Transformer.
input_ids = encoded["input_ids"]

print("\nInput IDs:")
print(input_ids)

print("\nInput IDs shape:")
print(input_ids.shape)


# 6. CONVERT THE MODEL INPUT BACK INTO TOKENS
# This lets us see the special tokens that were automatically added when preparing the sentence for DistilBERT.
model_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
print("\nTokens actually sent to the model:")
print(model_tokens)


# 7. LOOK AT THE ATTENTION MASK
# The attention mask tells the model which positions contain real input and which positions are padding.
# 1 = real token
# 0 = padding token
attention_mask = encoded["attention_mask"]

print("\nAttention mask:")
print(attention_mask)

print("\nAttention mask shape:")
print(attention_mask.shape)

model.eval()

with torch.no_grad():
    outputs = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"]
    )

hidden_states = outputs.last_hidden_state

print("\nTransformer output shape:")
print(hidden_states.shape)

sentence1 = "I deposited money at the bank."
sentence2 = "I sat beside the river bank."

encoded1 = tokenizer(sentence1, return_tensors="pt")
encoded2 = tokenizer(sentence2, return_tensors="pt")

print("\nSentence 1 tokens:")
print(tokenizer.convert_ids_to_tokens(encoded1["input_ids"][0]))

print("\nSentence 2 tokens:")
print(tokenizer.convert_ids_to_tokens(encoded2["input_ids"][0]))

with torch.no_grad():
    output1 = model(
        input_ids=encoded1["input_ids"],
        attention_mask=encoded1["attention_mask"]
    )

    output2 = model(
        input_ids=encoded2["input_ids"],
        attention_mask=encoded2["attention_mask"]
    )

tokens1 = tokenizer.convert_ids_to_tokens(encoded1["input_ids"][0])
tokens2 = tokenizer.convert_ids_to_tokens(encoded2["input_ids"][0])

bank_index1 = tokens1.index("bank")
bank_index2 = tokens2.index("bank")

bank1 = output1.last_hidden_state[0, bank_index1]
bank2 = output2.last_hidden_state[0, bank_index2]

print("\nBank vector shape in sentence 1:")
print(bank1.shape)

print("\nBank vector shape in sentence 2:")
print(bank2.shape)

similarity = torch.nn.functional.cosine_similarity(
    bank1.unsqueeze(0),
    bank2.unsqueeze(0)
)

print("\nCosine similarity:")
print(similarity.item())