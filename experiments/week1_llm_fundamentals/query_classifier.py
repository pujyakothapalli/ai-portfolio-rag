from anthropic import Anthropic
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def extract_json(response_text):
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())

def classify_query(query):
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        temperature=0,
        system="""You are a query classification assistant.
You ALWAYS respond with valid JSON only. No markdown, no explanation.""",
        messages=[{"role": "user", "content": f"""Classify this query into exactly one type:

- FACTUAL: asks for a specific fact, date, number, or definition
- ANALYTICAL: asks to compare, evaluate, or analyze something  
- CONVERSATIONAL: casual chat, greetings, opinions
- SUMMARIZATION: asks to summarize or condense content

Return JSON with exactly these fields:
{{
    "query_type": "FACTUAL" | "ANALYTICAL" | "CONVERSATIONAL" | "SUMMARIZATION",
    "confidence": 0.0 to 1.0,
    "reasoning": "one sentence explanation"
}}

Query: "{query}"

JSON:"""}]
    )
    return extract_json(response.content[0].text)

# Test queries
queries = [
    "What is the capital of France?",
    "Compare RAG vs fine-tuning for domain adaptation",
    "Hey, how are you doing today?",
    "Summarize the key points of this document",
    "What were the main causes of the 2008 financial crisis?",
    "Which embedding model performs better for medical text?",
    "Can you give me a quick overview of this paper?",
    "What is gradient descent?",
]

print("=" * 65)
print(f"{'Query':<45} {'Type':<15} {'Conf'}")
print("=" * 65)

results = []
for query in queries:
    result = classify_query(query)
    results.append((query, result))
    print(f"{query[:44]:<45} {result['query_type']:<15} {result['confidence']:.2f}")

print("\n--- Reasoning ---")
for query, result in results:
    print(f"\nQ: {query}")
    print(f"   {result['reasoning']}")