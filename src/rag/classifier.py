import json
import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())

def classify_query(query: str) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        temperature=0,
        system="You are a query classifier. Respond with JSON only. No markdown.",
        messages=[{"role": "user", "content": f"""Classify this query:

{{
    "query_type": "FACTUAL" | "ANALYTICAL" | "SUMMARIZATION" | "CONVERSATIONAL",
    "retrieval_strategy": "dense" | "hybrid" | "hybrid",
    "top_k": 3 | 5 | 7,
    "reasoning": "one sentence"
}}

Rules:
- FACTUAL → dense, top_k 3 (specific fact lookup)
- ANALYTICAL → hybrid, top_k 7 (needs broad context)  
- SUMMARIZATION → hybrid, top_k 7 (needs full coverage)
- CONVERSATIONAL → dense, top_k 3 (simple questions)

Query: "{query}"

JSON:"""}]
    )
    return extract_json(response.content[0].text)