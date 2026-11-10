import os
from dotenv import load_dotenv
from google import genai
import time

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

for attempt in range(3):
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents="""
            Explain in 3 simple sentences how extreme heat
            affects electricity demand in Texas.
            """
        )

        print("\nGEMINI RESPONSE:\n")
        print(response.text)
        break

    except Exception as e:
        print(f"Attempt {attempt + 1} failed: {e}")

        if attempt < 2:
            time.sleep(10)