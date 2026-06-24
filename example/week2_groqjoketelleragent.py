from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)

class Agent:
    def __init__(self, name, instructions, model):
        self.name = name
        self.instructions = instructions
        self.model = model

client = Groq()

agent = Agent(
    name="Jokester",
    instructions="You are a joke teller",
    model="llama-3.3-70b-versatile"
)

response = client.chat.completions.create(
    model=agent.model,
    messages=[
        {"role": "system", "content": agent.instructions},
        {"role": "user", "content": "Tell a joke about Autonomous AI Agents"}
    ]
)

print(response.choices[0].message.content)