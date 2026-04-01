from google import genai
from config.api_keys import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)
for m in client.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(m.name)
