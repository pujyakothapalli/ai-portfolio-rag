import tiktoken

# Tokenizer used by GPT-4
enc = tiktoken.get_encoding("cl100k_base")

# Tokenize some strings
texts = [
    "Hello world",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog",
    "AI is transforming healthcare and technology",
    "supercalifragilisticexpialidocious",
    "ChatGPT",
    "1234567890",
]

print("=" * 50)
print(f"{'Text':<40} {'Tokens':>6}")
print("=" * 50)

for text in texts:
    tokens = enc.encode(text)
    print(f"{text:<40} {len(tokens):>6}")
    print(f"  Token IDs: {tokens}")
    print()