from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def call_claude(prompt, temperature, max_tokens=150):
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# ── EXPERIMENT 1: Temperature ──────────────────────────
print("=" * 60)
print("EXPERIMENT 1: Temperature effect")
print("=" * 60)

prompt = "Explain what an embedding is in 2 sentences."
temperatures = [0, 0.5, 1.0]

for temp in temperatures:
    print(f"\n── Temperature: {temp} ──")
    # Call 3 times at each temperature to see variance
    for i in range(3):
        response = call_claude(prompt, temperature=temp)
        print(f"Run {i+1}: {response[:120]}...")

# ── EXPERIMENT 2: Determinism at temperature 0 ────────
print("\n" + "=" * 60)
print("EXPERIMENT 2: Is temperature=0 truly deterministic?")
print("=" * 60)

prompt = "Give me a random number between 1 and 100."
print("\nTemperature=0 (should be same every time):")
for i in range(10):
    response = call_claude(prompt, temperature=0, max_tokens=10)
    print(f"  Run {i+1}: {response.strip()}")

print("\nTemperature=1 (should vary):")
for i in range(10):
    response = call_claude(prompt, temperature=1, max_tokens=10)
    print(f"  Run {i+1}: {response.strip()}")


# ── EXPERIMENT 3: Temperature on creative tasks ────────
print("\n" + "=" * 60)
print("EXPERIMENT 3: Temperature matters on creative tasks")
print("=" * 60)

creative_prompt = "Write a one-sentence tagline for an AI startup."

print("\nTemperature=0 (should be identical every time):")
for i in range(3):
    response = call_claude(creative_prompt, temperature=0, max_tokens=50)
    print(f"  Run {i+1}: {response.strip()}")

print("\nTemperature=1 (should vary significantly):")
for i in range(3):
    response = call_claude(creative_prompt, temperature=1, max_tokens=50)
    print(f"  Run {i+1}: {response.strip()}")