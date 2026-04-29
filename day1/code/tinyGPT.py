import torch
import torch.nn as nn
import torch.nn.functional as F

# load the data from file
with open("data.txt", "r") as f:
    text = f.read()  # read entire content in string

# set objects are unordered and you cannot sort them or index them directly, if not done sorted is anyway returning a sorted list
chars = sorted(list(set(text)))
vocab_size = len(chars)  # The total number of unique tokens your model knows. If vocab_size = 50, you have 50 vectors
# In REAL MODELS: token = subword and vocab_size ≈ 30,000 – 100,000
# The number of “words” (or pieces of words) the model can choose from at each step

# Build the 'string-to-integer' (stoi) dictionary
# enumerate(chars) yields pairs of (index, character) like (0, 'a'), (1, 'b')
# 'ch:i' sets the character as the Key and the index as the Value
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Define a function to convert a string (text) into a list of numbers


def encode(s):
    # This list comprehension loops through every character 'c' in the input string 's'
    # It looks up each character in the 'stoi' dictionary to get its unique number
    return [stoi[c] for c in s]

# Define a function to convert a list of numbers back into a string


def decode(l):
    # 1. [itos[i] for i in l] creates a list of characters by looking up each number 'i'
    # 2. ''.join(...) takes that list of characters and glues them together into one string
    return ''.join([itos[i] for i in l])


# A tensor = multi-dimensional array (like NumPy array)
# Why tensors?
# Fast computation
# GPU support
# Required for neural networks
#  Converts that list into a PyTorch "Tensor" object.
data = torch.tensor(encode(text), dtype=torch.long)

# hyperparameters are settings that define how your GPT model will learn and how big its "brain" will be.
block_size = 8  # This is the context window. It means the model only looks at the last 8 characters to predict the 9th one. If you want it to remember long sentences, you’d eventually increase this.
batch_size = 16  # The model doesn't look at one example at a time; it looks at 16 chunks of text simultaneously in every training step to speed things up
# This is how many numbers the model uses to represent a single character. Instead of just "4", it represents the letter 'h' as a list of 32 different numbers to capture its meaning.
embedding_dim = 32
# This is Multi-Head Attention. It means the model looks at the text through 4 different "sets of eyes" simultaneously (one might look for grammar, another for rhyming, etc.).
num_heads = 4
# This is how many Transformer blocks are stacked on top of each other. More layers usually mean a smarter model but a slower training time.
num_layers = 2
# (0.001): This is how big of a "step" the model takes when correcting its mistakes. Too big, and it crashes; too small, and it takes forever to learn
learning_rate = 1e-3
# This is how many times the model will go through the training cycle. 2,000 rounds of practice.
epochs = 2000


def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # For every random start index (i), we grab a chunk of 8 (block_size) characters
    x = torch.stack([data[i:i+block_size] for i in ix])
    # torch.stack = This takes those 16 separate chunks and "stacks" them on top of each other into a single matrix (a 16x8 grid).
    # This is the "answer key." It grabs the same chunk but shifted one position to the right. Why shift? Because GPT is a predictor. If the input (x) is [h, e, l, l], the target (y) should be [e, l, l, o]. It teaches the model: "When you see 'h', the next letter is 'e'."
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# Self attention implementation
# This is the "Communication Layer." Self-attention is how every character in your block_size (8 characters) looks at the others to figure out which ones are important for predicting the next letter
# 1. The Three Roles (Query, Key, Value)
# In the __init__, we create three linear layers. Every character gets three vectors:
# Query (Q): "What am I looking for?" (The search term).
# Key (K): "What do I contain?" (The profile description).
# Value (V): "If I am important, what information do I share?" (The actual content).


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()  # It tells Python to connect your SelfAttention class to all the built-in features of nn.Module
        # If your "brain" size is 32 and you have 4 heads, each head gets a smaller chunk of 8. Why? It’s more efficient. It’s like splitting a 32-person meeting into 4 specialized teams of 8.
        self.heads = heads
        self.head_dim = embed_size // heads

        # If you used one big layer, the model would only be able to focus on one relationship at a time. By splitting it into heads, you're forcing the model to learn multiple things at once (like grammar AND meaning).

        # These are Trainable Weights.
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        # After you split the work into 4 heads and get 4 different results, you need a way to merge them back together. This layer "mixes" the insights from all heads back into a single 32-dimension representation.
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # This grabs the dimensions of the input data: Batch size (number of samples), Time/Sequence length (number of tokens), and Channels/Embedding size (vector size per token).
        B, T, C = x.shape

        # The input x is passed through linear layers (defined in __init__) to create the Query, Key, and Value matrices.
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # .view(B, T, self.heads, self.head_dim)==> The input embedding dimension (C) is split into smaller, independent "heads"
        # .transpose(1, 2): The dimensions are rearranged to (B, Heads, T, Head_dim).
        # Why? This puts the Heads dimension before Time (T), allowing PyTorch to perform matrix multiplication on all heads independently and simultaneously.

        Q = Q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # Q @ K.transpose(-2, -1): This is matrix multiplication (dot-product) between Queries and Keys, determining the affinity between every token and every other token.
        attn = (Q @ K.transpose(-2,
                                -1)) / (self.head_dim ** 0.5)  # Scaling. The dot products are divided by the square root of the head dimension to keep gradients stable.

        # Converts the raw scores into probabilities (summing to 1) along the last dimension, determining how much focus one token places on others.
        attn = F.softmax(attn, dim=-1)

        # The attention probabilities are multiplied by the Value () matrix.
        # Result: Each token's new representation is a weighted sum of the values of other tokens based on the attention scores
        out = attn @ V

        # Reverses the earlier transpose, moving the heads back next to the Time dimension.
        # .contiguous(): Ensures the tensor is stored continuously in memory, which is necessary after a transpose.
        # .view(B, T, C): Concatenates all the heads back together, reshaping the (B, T, Heads, Head_dim) tensor back to the original (B, T, C) format.
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.fc_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()

        # Plugs in the SelfAttention class you just built. This is the communication phase.
        self.attn = SelfAttention(embed_size, heads)
        # These are like "volume knobs." They normalize the numbers so no single neuron gets too loud or too quiet. This makes training much more stable.
        self.ln1 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            # It expands the data to a larger space (32 becomes 128) to give it more "room" to process complex patterns.
            nn.Linear(embed_size, 4 * embed_size),
            # Rectified Linear Unit=  A simple filter that turns negative numbers to zero. This adds "non-linearity," which is a fancy way of saying it lets the model learn complex logic instead of just simple math.
            nn.ReLU(),
            # It shrinks the data back down to its original size (32).
            nn.Linear(4 * embed_size, embed_size)
        )  # After communicating, every character needs to "think" individually.
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
            # These are called Residual Connections (or Skip Connections).
            # Normalize x. Run it through Attention, Add the result back to the original x. Why? This allows the original information to flow through the network without getting lost or "diluted." It's like having a copy of the original notes while you're adding new highlights.
        x = x + self.attn(self.ln1(x))
            # Normalize again. Run it through the FeedForward "thinking" layers. Add it back again
        x = x + self.ff(self.ln2(x))
        return x

# GPT Implementation
# What this means:
# You are defining a custom neural network
# Anything inside this class becomes part of the model:
# layers
# weights
# forward logic
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()  # Why super().__init__()? ==> It initializes the parent class (nn.Module), which:registers parameters, enables .parameters() for optimizer, enables GPU support (.to(device)) Without this → your model won’t work properly.
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        # A lookup table that converts token IDs → vectors:shape = [vocab_size × embedding_dim]e.g. 50 tokens, each mapped to 32 numbers → matrix = 50 × 32 ==> instead of 'h' = 7 you get 'h' = [0.12, -0.88, 0.45, ..., 32 values]
        # Why needed? Neural networks can’t learn from raw integers meaningfully—vectors allow:
        # similarity learning
        # semantic structure
        
        self.position_embedding = nn.Embedding(block_size, embedding_dim) 
        # Why? Because transformers don’t understand order.
        # Another lookup table, but for positions
        # Shape = (block_size, embedding_dim) e.g. (8, 32) → positions 0–7
        # Adds information like: "This is token at position 0"
        # Transformers have no inherent sense of order so without this "abc" == "cba" wrong so final output becomes token_vector + position_vector

        self.blocks = nn.Sequential(
            *[TransformerBlock(embedding_dim, num_heads) for _ in range(num_layers)] # Creates a list of identical layers e.g. num_layers = 2 → [Block1, Block2]
            # * (unpacking) Converts list into arguments
            # nn.Sequential(...) Chains layers one after another
            # So effectively: input → Block1 → Block2 → ... → output
            # Each block: Looks at relationships between tokens (attention), Processes features (feedforward network), Keeps original info via residual connections
            # Why stack multiple blocks?
            # Each layer builds deeper understanding: Layer 1 → basic patterns, Layer 2 → relationships, Layer N → higher-level structure
        )
        self.ln = nn.LayerNorm(embedding_dim) # layer normalization ==> Normalizes values across features, keeps numbers stable mean ~ 0 and variance ~ 1
        # Why needed? Prevents exploding/vanishing values, Stabilizes training, Helps gradients flow
        # It rescales the values of a vector so they stay well-behaved (centered and not too large/small).
        # =====================================\
        # Imagine your model produces this vector for a token:
        # [100, -200, 50, 300]
        # This is unstable:
        # values are too large
        # inconsistent scale
        # 
        # LayerNorm converts it to something like:
        # [0.5, -1.2, 0.1, 0.6]
        # Now:
        # values are balanced
        # easier for next layers to process
        # Problem without it:
        # As data flows through layers:
        # values can explode (very large)
        # or vanish (very small)
        # This causes:
        # unstable training
        # poor learning
        # It is: A numerical stabilizer that ensures the model can learn effectively.
        #====================================================
        self.fc = nn.Linear(embedding_dim, vocab_size) # A projection from: embedding space → vocabulary space 
        # Input: vector of size embedding_dim
        # Output: vector of size vocab_size

    def forward(self, idx):
        B, T = idx.shape # idx is tensor and shape returns its dimensions for hello => idx.shape = (B,T)=(1,5) = B =batch size and T= sequence length
        token_emb = self.token_embedding(idx) # token_emb => B, T, C-> embedding_dim (eg 32)
        pos_emb = self.position_embedding(torch.arange(T)) # maps positions with verctors => pos_emb → (T, C)
        x = token_emb + pos_emb

        x = self.blocks(x) # applies multiple TransformerBlocks one after anothe
        x = self.ln(x)
        logits = self.fc(x)
        return logits
    
    #  It takes a starting sequence and predicts the next words one-by-one.
    def generate(self, idx, max_new_tokens): 
        for _ in range(max_new_tokens): # This tells the computer: "I want you to generate exactly this many new characters/words." It repeats the steps below for every single new token.
            idx_cond = idx[:, -block_size:] # Slicing - [start:stop]=From every row, take the last block_size columns: The model has a "memory limit" called block_size.If your block_size is 10, but you’ve already generated 50 words, the model only looks at the last 10 to decide what comes next.
            logits = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
        
# Create the model You are creating your neural network
# It contains:embeddings, attention layers, linear layer
model = TinyGPT()

# Optimize the model , It updates the model’s weights to reduce mistakes
# model.parameters() → all learnable values, lr (learning rate) → how big changes should be, its “Teacher who adjusts the brain after each mistake”
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for step in range(epochs): # Repeat learning many times (2000 steps)
    x, y = get_batch() # get data 
    logits = model(x) # Pass input through model, Get predictions
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)) # Compare predicted characters vs actual characters

    optimizer.zero_grad() # Clear old gradients, PyTorch accumulates gradients, You must reset before next step
    loss.backward() # Backpropagation= Find out what caused the mistake, Model computes: which weights were wrong, how much to adjust
    optimizer.step() # Update weights, Fix the model slightly

    if step % 200 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")

# inference the model
context = torch.tensor([encode('data leads to ')], dtype=torch.long)
output = model.generate(context, max_new_tokens=100)
print("\n Generated Text: ")
print(decode(output[0].tolist()))

# ==========================================================================
#TRAINING:
# data → model → prediction → error → fix → repeat

# INFERENCE:
# start text → model → next char → next char → next char
