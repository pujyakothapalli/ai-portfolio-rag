from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── EXPERIMENT 4: Conversation memory ─────────────────
print("=" * 60)
print("EXPERIMENT 4: Multi-turn conversation with memory")
print("=" * 60)

def chat_with_memory():
    """A conversation loop that remembers everything said."""
    conversation_history = []
    
    system_prompt = """You are a helpful AI tutor teaching concepts 
about machine learning. Be concise — 2-3 sentences max per response."""
    
    print("\nChat started. Type 'quit' to exit, 'history' to see memory.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print(f"\nConversation ended. Total turns: {len(conversation_history) // 2}")
            break
            
        if user_input.lower() == 'history':
            print("\n── Conversation History ──")
            for msg in conversation_history:
                print(f"  {msg['role'].upper()}: {msg['content'][:80]}...")
            print()
            continue
        
        if not user_input:
            continue
        
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Send FULL history every time — this is how LLMs remember
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=150,
            temperature=0.7,
            system=system_prompt,
            messages=conversation_history  # entire history sent each time
        )
        
        assistant_message = response.content[0].text
        
        # Add assistant response to history
        conversation_history.append({
            "role": "assistant", 
            "content": assistant_message
        })
        
        # Show token usage — watch this grow
        print(f"\nClaude: {assistant_message}")
        print(f"[Tokens used this call: {response.usage.input_tokens} in, "
              f"{response.usage.output_tokens} out | "
              f"History length: {len(conversation_history)} messages]\n")

chat_with_memory()