from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def call_claude(system_prompt, user_message, temperature=0):
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    return response.content[0].text

# ── TEST TEXT ──────────────────────────────────────────
text = "The new transformer model achieved state-of-the-art results \
on multiple benchmarks but requires significant computational resources \
and the training process was unstable at higher learning rates."

print("=" * 60)
print("EXPERIMENT 1: Zero-shot vs Few-shot vs Chain-of-thought")
print("=" * 60)

# 1. Zero-shot
zero_shot = call_claude(
    system_prompt="You are a helpful assistant.",
    user_message=f"Classify the sentiment of this text as POSITIVE, NEGATIVE, or MIXED:\n\n{text}"
)
print(f"\n1. Zero-shot:\n{zero_shot}")

# 2. Few-shot
few_shot = call_claude(
    system_prompt="You are a helpful assistant.",
    user_message=f"""Classify sentiment as POSITIVE, NEGATIVE, or MIXED.

Examples:
Text: "The model is fast and accurate" → POSITIVE
Text: "The model is slow and breaks often" → NEGATIVE  
Text: "The model is accurate but very slow" → MIXED

Now classify:
Text: "{text}" →"""
)
print(f"\n2. Few-shot:\n{few_shot}")

# 3. Chain-of-thought
cot = call_claude(
    system_prompt="You are a helpful assistant.",
    user_message=f"""Classify the sentiment of this text as POSITIVE, NEGATIVE, or MIXED.
Think step by step before giving your answer.

Text: "{text}"

Step 1 - identify positive aspects:
Step 2 - identify negative aspects:
Step 3 - final classification:"""
)
print(f"\n3. Chain-of-thought:\n{cot}")