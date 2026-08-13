from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def count_tokens(messages, system=""):
    """Estimate token count for a conversation."""
    response = client.messages.count_tokens(
        model="claude-sonnet-4-5",
        system=system,
        messages=messages
    )
    return response.input_tokens

def summarize_conversation(history):
    """Compress old conversation into a summary."""
    history_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}" 
        for msg in history
    ])
    
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": f"""Summarize this conversation 
in 3-4 sentences, preserving all key facts and context discussed:

{history_text}

Summary:"""}]
    )
    return response.content[0].text

def smart_chat(user_input, conversation_history, system_prompt, token_limit=400):
    """Chat with automatic memory compression when context gets long."""
    
    # Add new message
    conversation_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Check token count
    current_tokens = count_tokens(conversation_history, system_prompt)
    print(f"[Token count: {current_tokens}]")
    
    # If approaching limit — compress old messages
    if current_tokens > token_limit and len(conversation_history) > 4:
        print("[⚡ Context getting long — compressing memory...]")
        
        # Keep last 2 exchanges, summarize everything before
        recent = conversation_history[-4:]
        older = conversation_history[:-4]
        
        summary = summarize_conversation(older)
        
        # Replace old history with summary
        conversation_history = [
            {"role": "user", "content": f"[Previous conversation summary: {summary}]"},
            {"role": "assistant", "content": "Understood, I have context from our previous discussion."}
        ] + recent
        
        print(f"[Compressed to {count_tokens(conversation_history, system_prompt)} tokens]")
    
    # Get response
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=150,
        temperature=0.7,
        system=system_prompt,
        messages=conversation_history
    )
    
    assistant_message = response.content[0].text
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    return assistant_message, conversation_history

# ── Run it ─────────────────────────────────────────────
system_prompt = "You are a concise ML tutor. Keep answers to 2-3 sentences."
history = []

test_conversation = [
    "What is a neural network?",
    "What is backpropagation?",
    "How does gradient descent work?",
    "What is a learning rate?",
    "What happens if learning rate is too high?",
    "What is batch normalization?",
    "What is dropout?",
    "What did we first talk about?",  # Tests if memory survived compression
]

print("=" * 60)
print("Smart conversation with memory compression")
print("=" * 60)

for user_msg in test_conversation:
    print(f"\nYou: {user_msg}")
    response, history = smart_chat(user_msg, history, system_prompt)
    print(f"Claude: {response}")