import os
from dotenv import load_dotenv

load_dotenv()

# Reads from .env — supports both key name spellings
api_key = os.getenv("SARVAM_API_KEY") or os.getenv("sarvamai_apy_key")

if not api_key:
    print("❌ No API key found.")
    print("   Open your .env file and add:  SARVAM_API_KEY=your_key_here")
    exit(1)

from sarvamai import SarvamAI

client = SarvamAI(api_subscription_key=api_key)

response = client.chat.completions(
    model="sarvam-105b-conversations",
    messages=[
        {"role": "user", "content": "Hello! Tell me one farming tip for Indian farmers."}
    ]
)

print("✅ Sarvam AI reply:")
print(response.choices[0].message.content)
