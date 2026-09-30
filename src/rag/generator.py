import os
from anthropic import Anthropic
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def generate_answer(query: str, chunks: List[Dict],
                    max_retries: int = 2) -> Dict:
    """Generate answer with Self-RAG confidence loop."""

    for attempt in range(max_retries + 1):
        context = "\n\n".join([
            f"[Source: {c['source']}]\n{c['content']}"
            for c in chunks
        ])

        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=600,
            temperature=0,
            system="""You are a precise assistant. Answer using ONLY the provided context.
If the answer isn't in the context, say "I don't have enough information."
Always cite sources. Be concise and accurate.""",
            messages=[{"role": "user", "content": f"""Context:
{context}

Question: {query}

Answer (cite sources, be precise):"""}]
        )

        answer = response.content[0].text

        # Self-RAG: score confidence
        confidence_response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=100,
            temperature=0,
            system="You are an evaluator. Respond with JSON only.",
            messages=[{"role": "user", "content": f"""Rate this answer:

Question: {query}
Answer: {answer}

{{"confidence": 0.0-1.0, "is_grounded": true/false, "needs_more_context": true/false}}

JSON:"""}]
        )

        try:
            conf_text = confidence_response.content[0].text.strip()
            if conf_text.startswith("```"):
                conf_text = conf_text.split("```")[1]
                if conf_text.startswith("json"):
                    conf_text = conf_text[4:]
            import json
            confidence = json.loads(conf_text.strip())
        except:
            confidence = {"confidence": 0.8,
                         "is_grounded": True,
                         "needs_more_context": False}

        # If confident enough or last attempt — return
        if confidence.get("confidence", 0) >= 0.7 or attempt == max_retries:
            return {
                "answer": answer,
                "confidence": confidence.get("confidence", 0.8),
                "is_grounded": confidence.get("is_grounded", True),
                "attempts": attempt + 1,
                "chunks_used": len(chunks)
            }

    return {"answer": answer, "confidence": 0.5,
            "is_grounded": False, "attempts": max_retries + 1,
            "chunks_used": len(chunks)}