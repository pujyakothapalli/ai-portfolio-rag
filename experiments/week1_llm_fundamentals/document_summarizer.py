from anthropic import Anthropic
from dotenv import load_dotenv
import os
import json
import sys

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

#  Helper Functions

def extract_json(response_text):
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())

def call_claude(system_prompt, user_message, temperature=0, max_tokens=500):
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    return response.content[0].text

# ── Step 1: Chunking ───────────────────────────────────

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # overlap between chunks
        
    return chunks

# ── Step 2: Classify document ─────────────────────────

def classify_document(text_sample):
    """Classify what kind of document this is."""
    response = call_claude(
        system_prompt="You are a document classifier. Respond with JSON only. No markdown.",
        user_message=f"""Classify this document and return JSON with exactly these fields:
{{
    "document_type": "research_paper" | "news_article" | "technical_doc" | "business_report" | "other",
    "domain": "one word domain e.g. technology, healthcare, finance, science",
    "complexity": "simple" | "moderate" | "complex",
    "estimated_read_time_minutes": number
}}

Document sample:
{text_sample[:1000]}

JSON:"""
    )
    return extract_json(response)

# ── Step 3: Summarize each chunk ──────────────────────

def summarize_chunk(chunk, chunk_num, total_chunks):
    """Summarize a single chunk."""
    return call_claude(
        system_prompt="You are a precise summarizer. Extract only key information.",
        user_message=f"""Summarize this section (part {chunk_num} of {total_chunks}).
Keep it to 2-3 sentences. Focus on key facts and insights.

Text:
{chunk}

Summary:""",
        max_tokens=150
    )

# ── Step 4: Final synthesis ───────────────────────────

def synthesize_summaries(summaries, doc_type):
    """Combine chunk summaries into one final summary."""
    combined = "\n\n".join([f"Section {i+1}: {s}" 
                             for i, s in enumerate(summaries)])
    
    return call_claude(
        system_prompt="You are an expert at synthesizing information clearly.",
        user_message=f"""You have summaries of different sections of a {doc_type}.
Combine them into one coherent final summary.

Section summaries:
{combined}

Write a final summary with:
1. Main topic (1 sentence)
2. Key findings or points (3-5 bullet points)
3. Conclusion (1 sentence)""",
        max_tokens=400
    )

# ── Step 5: Extract key insights ─────────────────────

def extract_insights(full_text_sample):
    """Extract actionable insights from the document."""
    response = call_claude(
        system_prompt="You are an analyst. Respond with JSON only. No markdown.",
        user_message=f"""Analyze this document and return JSON:
{{
    "key_themes": ["theme1", "theme2", "theme3"],
    "important_entities": ["entity1", "entity2"],
    "sentiment": "positive" | "negative" | "neutral" | "mixed",
    "actionable_insights": ["insight1", "insight2"],
    "one_line_summary": "single sentence capturing the essence"
}}

Document:
{full_text_sample[:2000]}

JSON:""",
        max_tokens=600
    )
    return extract_json(response)

# ── Main pipeline ─────────────────────────────────────

def summarize_document(text):
    print("\n" + "=" * 60)
    print("DOCUMENT SUMMARIZER")
    print("=" * 60)
    
    # Word count
    word_count = len(text.split())
    print(f"\n📄 Document loaded: {word_count} words")
    
    # Step 1: Classify
    print("\n🔍 Classifying document...")
    doc_info = classify_document(text)
    print(f"   Type:       {doc_info['document_type']}")
    print(f"   Domain:     {doc_info['domain']}")
    print(f"   Complexity: {doc_info['complexity']}")
    print(f"   Read time:  ~{doc_info['estimated_read_time_minutes']} minutes")
    
    # Step 2: Chunk
    print("\n✂️  Chunking document...")
    chunks = chunk_text(text, chunk_size=300, overlap=30)
    print(f"   Created {len(chunks)} chunks")
    
    # Step 3: Summarize chunks
    print("\n📝 Summarizing chunks...")
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"   Processing chunk {i+1}/{len(chunks)}...", end="\r")
        summary = summarize_chunk(chunk, i+1, len(chunks))
        chunk_summaries.append(summary)
    print(f"   ✅ All {len(chunks)} chunks summarized")
    
    # Step 4: Synthesize
    print("\n🔗 Synthesizing final summary...")
    final_summary = synthesize_summaries(chunk_summaries, doc_info['document_type'])
    
    # Step 5: Extract insights
    print("\n💡 Extracting insights...")
    insights = extract_insights(text)
    
    # ── Final output ──────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n📌 ONE LINE: {insights['one_line_summary']}")
    
    print(f"\n📊 FINAL SUMMARY:")
    print(final_summary)
    
    print(f"\n🏷️  KEY THEMES: {', '.join(insights['key_themes'])}")
    print(f"🎯 SENTIMENT: {insights['sentiment']}")
    
    print(f"\n✅ ACTIONABLE INSIGHTS:")
    for insight in insights['actionable_insights']:
        print(f"   • {insight}")
    
    # Save results to JSON
    output = {
        "document_info": doc_info,
        "word_count": word_count,
        "chunks_processed": len(chunks),
        "final_summary": final_summary,
        "insights": insights
    }
    
    with open("summary_output.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Full results saved to summary_output.json")
    print("=" * 60)

# ── Sample text to test with ──────────────────────────

sample_text = """
Retrieval-Augmented Generation (RAG) represents a significant advancement 
in natural language processing, combining the strengths of retrieval-based 
and generative approaches to create more accurate and reliable AI systems.

Traditional language models rely solely on knowledge encoded during training,
which can lead to outdated information, hallucinations, and lack of 
domain-specific knowledge. RAG addresses these limitations by dynamically 
retrieving relevant information from external knowledge bases at inference time.

The RAG architecture consists of two main components: a retriever and a generator.
The retriever searches a document corpus to find relevant passages based on 
the input query, typically using dense vector representations and approximate 
nearest neighbor search. The generator, usually a large language model, then 
conditions its output on both the original query and the retrieved passages.

Recent advances in RAG include more sophisticated retrieval strategies such as 
hybrid search combining sparse and dense retrieval, iterative retrieval where 
the model performs multiple rounds of retrieval, and self-RAG where the model 
learns to decide when and what to retrieve.

Evaluation of RAG systems presents unique challenges since performance depends 
on both retrieval quality and generation quality. Metrics such as faithfulness,
answer relevance, and context precision have been developed to assess these 
systems comprehensively. Frameworks like RAGAS provide automated evaluation 
pipelines that can measure these metrics at scale.

Applications of RAG span numerous domains including question answering, 
document summarization, code generation, and customer support. In healthcare,
RAG systems can provide clinicians with relevant medical literature while 
generating patient-specific recommendations. In finance, they can retrieve 
regulatory documents while generating compliance reports.

Despite its advantages, RAG faces challenges including retrieval latency,
context length limitations, and the difficulty of handling conflicting 
information across retrieved documents. Future research directions include
more efficient retrieval mechanisms, better integration of structured and 
unstructured data, and improved methods for handling uncertainty.
"""

summarize_document(sample_text)