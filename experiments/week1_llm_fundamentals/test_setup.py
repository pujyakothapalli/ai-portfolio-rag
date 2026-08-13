from dotenv import load_dotenv
import os
import anthropic

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=100,
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)

print("✅ Setup working!")
print(message.content[0].text)