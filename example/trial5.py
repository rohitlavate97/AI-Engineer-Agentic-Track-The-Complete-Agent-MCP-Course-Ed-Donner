from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv(override=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

requirement = """
User should login with valid credentials
"""

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": f"""
            Generate:
            1. Test Cases
            2. Appium Java Code by using UIAutomator2 for Android, set UIAutomator2Options
            and use page object model design pattern. Don't forget to include necessary imports and setup code.
            3. Integrate automated test cases with TestRail test cases
            Requirement:
            {requirement}
            """
        }
    ]
)

print(response.choices[0].message.content)