'''
tiny GPThas obvious limitations. Im just doing this to understand the mechanism. 
1.tiny dataset,only 247 characters.
2.also a small context. block_size=8.(can ony remember the last 8 characters)
3. it has a character level tokenizer: it's predicting individual letters, not words/subwords.'''

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Step 1: Tiny training dataset
text = """
neural networks learn patterns from data
transformers use attention to understand context
attention allows tokens to communicate with other tokens
neural networks can learn complex relationships
transformers can generate text one token at a time
"""
print(text)

#Step 2: make a character-level tokenizer
# Find every unique character in the dataset
characters = sorted(list(set(text))) #set(text) removes duplicates.
vocab_size = len(characters)
print("Vocabulary:")
print(characters)
print("\nVocabulary size:", vocab_size)

# Step 3: every character gets an ID
# Character → integer
char_to_id = {
    char: i
    for i, char in enumerate(characters)
}
# Integer → character
id_to_char = {
    i: char
    for i, char in enumerate(characters)
}
print("\nCharacter to ID:")
print(char_to_id)

#Step 4: Encode and Decode
def encode(string):
    return [char_to_id[char] for char in string]

def decode(ids):
    return "".join(id_to_char[i] for i in ids)
#testing the functions
example = "neural"
encoded = encode(example)
decoded = decode(encoded)
print("\nOriginal:", example)
print("Encoded:", encoded)
print("Decoded:", decoded)

#Step 5: Encode all of the training dataset
data = torch.tensor(
    encode(text),
    dtype=torch.long #these are integers. IDs not measurements
)
print("\nTotal number of tokens:", len(data))
print("Data shape:", data.shape)
print("\nFirst 50 token IDs:")
print(data[:50])
print("\nDecoded first 50 tokens:")
print(decode(data[:50].tolist()))

#Step 6: Create next-token training examples
block_size = 8 #tiny GPT can look at 8 previous tokens when making a prediction
chunk = data[:block_size + 1]
print("\nChunk:")
print(chunk)
print("Decoded chunk:")
print(decode(chunk.tolist()))

x = chunk[:-1]
y = chunk[1:]

print("\nInput x:")
print(x)
print(decode(x.tolist()))

print("\nTarget y:")
print(y)
print(decode(y.tolist()))

#Step 10:Building an attention head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        self.register_buffer(
            "tril", #lower triangular
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        #QK^T / SQRT(d_k)
        scores = q @ k.transpose(-2, -1)
        scores = scores / (k.shape[-1] ** 0.5)

        #Causal padding: block future tokens
        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        #convert scores into attentipn weights
        attention_weights = F.softmax(scores, dim=-1)

        #weighted mix of values
        out = attention_weights @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size)
            for _ in range(num_heads)
        ])

        self.projection = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        outputs = [
            head(x)
            for head in self.heads
        ]

        out = torch.cat(outputs, dim=-1)

        out = self.projection(out)

        return out

#Step 7: Build Batches
batch_size = 4

def get_batch():
    # Random starting positions
    starts = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in starts
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in starts
    ])

    return x, y

xb, yb = get_batch()

print("\nBatch input shape:", xb.shape)
print("Batch target shape:", yb.shape)

print("\nFirst input sequence:")
print(decode(xb[0].tolist()))

print("First target sequence:")
print(decode(yb[0].tolist()))

#Step 8: give every token a learned vector 
n_embd = 32

token_embedding_table = nn.Embedding(
    vocab_size,
    n_embd
)
token_embeddings = token_embedding_table(xb)
print("\nToken IDs shape:")
print(xb.shape)
print("\nToken embeddings shape:")
print(token_embeddings.shape)

#Step 9:Positional embeddings
position_embedding_table = nn.Embedding(
    block_size,
    n_embd
)
#the positions
positions = torch.arange(block_size)
position_embeddings = position_embedding_table(positions)

print("\nPositions:")
print(positions)
print("\nPosition embeddings shape:")
print(position_embeddings.shape)

#what token am I? + Where am I ?
x = token_embeddings + position_embeddings
print("\nCombined representation shape:")
print(x.shape)

'''
#attntion head
head_size = 8
head = Head(head_size)
attention_output = head(x)

print("\nAttention output shape:")
print(attention_output.shape)
'''
#Step 11: Multiple attention heads
multi_head_attention = MultiHeadAttention(
    num_heads=4,
    head_size=8
)

attention_output = multi_head_attention(x)

print("\nMulti-head attention output shape:")
print(attention_output.shape)

#Step 13: Residual Connection
x = x + multi_head_attention(x)
# The Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.network(x)


#Step 14: LayerNorm + Complete Block
class Block(nn.Module):
    def __init__(self):
        super().__init__()

        head_size = n_embd // 4

        self.attention = MultiHeadAttention(
            num_heads=4,
            head_size=head_size
        )

        self.feed_forward = FeedForward()

        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        x = x + self.attention(
            self.layer_norm1(x)
        )

        x = x + self.feed_forward(
            self.layer_norm2(x)
        )

        return x

block = Block()

block_output = block(x)

print("\nTransformer block output shape:")
print(block_output.shape)

#Step 15: Stack multiple transformer blocks
num_layers = 4

blocks = nn.Sequential(
    *[Block() for _ in range(num_layers)]
)

transformer_output = blocks(x)

print("\nAfter 4 Transformer blocks:")
print(transformer_output.shape)

#Step 16: Building the TinyGPT
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )

        self.blocks = nn.Sequential(
            *[Block() for _ in range(4)]
        )

        self.final_layer_norm = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )

    def forward(self, idx, targets = None):
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)

        position_embeddings = self.position_embedding_table(
            torch.arange(T)
        )

        x = token_embeddings + position_embeddings

        x = self.blocks(x)

        x = self.final_layer_norm(x)

        logits = self.lm_head(x)

        #Step 17:CROSS-ENTROPY LOSS:if we do not have correct  answers , just return the predictions
        if targets is None:
            loss = None

        else:
            B,T,C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)

            loss =  F.cross_entropy(
                logits_flat, targets_flat
            )

        return logits,loss
    
    #Step 19: generate text
    def generate(self, idx,max_new_tokens, temperature = 1.0): #same idea with temp in modern langauge models. 0 more conservative, 1 more adventurous
        for _ in range(max_new_tokens):
            # Only use the most recent block_size tokens
            idx_context = idx[:, -block_size:] #give GPT only the most recent 8 chars

            #ask GPT for predictions
            logits,loss = self(idx_context)

            #only care about the prediction frpom LASt position
            logits= logits[:, -1, :]

            ## Control how random the predictions are
            logits = logits/temperature

            #convert the logits to probabilities with softmax'
            probabilities = F.softmax(logits, dim = -1)

            # Sample the next character
            next_token = torch.multinomial(
                 probabilities, num_samples=1
             )

            #add that predicted character to the sequence
            idx = torch.cat(
                (idx, next_token),
                dim =1
            )

        return idx
        

model = TinyGPT()

logits,loss = model(xb,yb)

print("\nInput shape:")
print(xb.shape)

print("\nLogits shape:")
print(logits.shape)

print("\nInitial loss:")
print(loss.item())

#Train Tiny GPT
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

training_steps = 2000

for step in range(training_steps):

    # Get a new random batch
    xb, yb = get_batch()

    # Forward pass
    logits, loss = model(xb, yb)

    # Clear old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()

    if step % 200 == 0:
        print(
            f"Step {step}: "
            f"Loss = {loss.item():.4f}"
        )

# Start generation with a newline character
start = torch.tensor(
    [[char_to_id["\n"]]],
    dtype=torch.long
)

generated = model.generate(
    start,
    max_new_tokens=200,
    temperature=1.5
)
print("\nGenerated tensor shape:")
print(generated.shape)

print("\nGenerated token IDs:")
print(generated)

generated_text = decode(
    generated[0].tolist()
)

print("\nGenerated text:")
print(generated_text)


