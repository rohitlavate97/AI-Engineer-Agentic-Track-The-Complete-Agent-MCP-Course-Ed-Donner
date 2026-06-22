from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)
client = OpenAI()

response = client.responses.create(
    model="gpt-5.5",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)