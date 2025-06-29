from google import genai
from dotenv import load_dotenv
import os

# Load the API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Check if the key is loaded correctly
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables!")

# Create the client using the environment variable
client = genai.Client(api_key=api_key)

# Function to get medicine suggestions
def get_medicine_suggestion(disease_name: str,animal: str) -> str:
    prompt = f"""
You are a veterinary assistant.

Provide the following for the disease: {disease_name}, in the animal: {animal}:

- If the disease is 'Healthy', 'Health', or 'Normal Skin', return general well-being and good care messages only.
- Otherwise, first suggest commonly prescribed medicines for the disease, then give general treatment and care advice.

Do not include any introductory or concluding statements.
"""
    print("Sending prompt to Gemini API...")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    print("Received response from Gemini API.")
    return response.text

# Run the function if executed directly
if __name__ == "__main__":
    disease = input("Enter the disease name: ")
    suggestion = get_medicine_suggestion(disease)
    print("\n💊 Suggested Treatment:\n")
    print(suggestion)
