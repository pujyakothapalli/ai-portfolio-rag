from anthropic import Anthropic
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def call_claude(system_prompt, user_message):
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    return response.content[0].text

def extract_json(response_text):
    """Strip markdown code fences if present, then parse JSON."""
    text = response_text.strip()
    if text.startswith("```"):
        # Remove ```json or ``` from start, and ``` from end
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())

# ── EXPERIMENT 2: Structured Output ───────────────────
print("=" * 60)
print("EXPERIMENT 2: Getting reliable JSON from an LLM")
print("=" * 60)

texts = [
    "The new RAG pipeline reduced hallucinations by 40% but increased latency",
    "Our model completely failed on out-of-distribution data",
    "Incredible results — 99% accuracy on the test set with zero fine-tuning",
]

# Bad prompt — unpredictable output
print("\n❌ BAD PROMPT (no structure enforced):")
bad_response = call_claude(
    system_prompt="You are a helpful assistant.",
    user_message=f"Analyze this text: '{texts[0]}'"
)
print(bad_response)

# Good prompt — structured JSON output
print("\n✅ GOOD PROMPT (structured JSON enforced):")
for text in texts:
    good_response = call_claude(
        system_prompt="""You are a text analysis assistant. 
You ALWAYS respond with valid JSON only. 
No explanation, no markdown, no extra text. Just the JSON object.""",
        user_message=f"""Analyze this text and return a JSON object with exactly these fields:
{{
    "sentiment": "POSITIVE" | "NEGATIVE" | "MIXED",
    "confidence": 0.0 to 1.0,
    "positive_aspects": ["list", "of", "positives"],
    "negative_aspects": ["list", "of", "negatives"],
    "one_line_summary": "brief summary"
}}

Text: "{text}"

JSON:"""
    )
    
    # Try to parse it — this is how your RAG pipeline will use it
    try:
        parsed = extract_json(good_response)
        print(f"\nText: {text[:50]}...")
        print(f"  Sentiment:  {parsed['sentiment']} (confidence: {parsed['confidence']})")
        print(f"  Positives:  {parsed['positive_aspects']}")
        print(f"  Negatives:  {parsed['negative_aspects']}")
        print(f"  Summary:    {parsed['one_line_summary']}")
    except json.JSONDecodeError:
        print(f"  ❌ Failed to parse JSON: {good_response}")