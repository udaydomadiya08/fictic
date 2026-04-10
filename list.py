

from google import genai

# Replace with your actual API key
api_key = "AIzaSyCgrOfRMJiEVgNIWowe05zResbqzqjjNdY"
from google import genai




client = genai.Client(api_key=api_key)

# List all available models
models = client.models.list()  # For the current SDK, 'models.list()' is the correct call

print("✅ Available Models:")
for m in models:
    print(m.name, "-", getattr(m, "description", "No description"))
