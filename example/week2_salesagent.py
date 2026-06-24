"""
sales_agents_groq.py

PURPOSE
-------
This file shows BOTH:

1. OpenAI Agents SDK approach (commented)
2. Groq equivalent (working code)

This allows you to learn:
- What OpenAI Agent SDK does
- How to replace it with Groq
- How parallel agent execution works

Author: Rohit Learning Agentic AI
"""

# ============================================================
# IMPORTS
# ============================================================

from dotenv import load_dotenv
from groq import Groq
import asyncio
import os

# ------------------------------------------------------------
# OPENAI AGENTS SDK IMPORTS
# ------------------------------------------------------------

# from agents import Agent
# from agents import Runner
# from agents import trace

# Why commented?
#
# OpenAI Agents SDK requires:
#
# - OpenAI API Credits
# - OpenAI Responses API
# - OpenAI Tracing
#
# We are replacing it with Groq.


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(override=True)

# ============================================================
# CREATE GROQ CLIENT
# ============================================================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ============================================================
# AGENT CLASS
# ============================================================

# OPENAI VERSION
#
# agent = Agent(
#     name="Jokester",
#     instructions="You are funny",
#     model="gpt-4o-mini"
# )

# GROQ VERSION
#
# Groq does not provide Agent objects.
# We create a lightweight Agent class.

class Agent:

    def __init__(self, name, instructions, model):

        self.name = name
        self.instructions = instructions
        self.model = model


# ============================================================
# SALES AGENT INSTRUCTIONS
# ============================================================

instructions1 = """
You are a professional sales representative.

You work for ComplAI.

Write professional cold sales emails.
"""

instructions2 = """
You are an engaging sales representative.

You work for ComplAI.

Write witty cold sales emails.

Your goal is to maximize response rate.
"""

instructions3 = """
You are a busy sales representative.

You work for ComplAI.

Write concise cold sales emails.
"""

# ============================================================
# AGENT CREATION
# ============================================================

# OPENAI VERSION
#
# sales_agent1 = Agent(
#     name="Professional Sales Agent",
#     instructions=instructions1,
#     model="gpt-4o-mini"
# )

# GROQ VERSION

sales_agent1 = Agent(
    name="Professional Sales Agent",
    instructions=instructions1,
    model="llama-3.3-70b-versatile"
)

sales_agent2 = Agent(
    name="Engaging Sales Agent",
    instructions=instructions2,
    model="llama-3.3-70b-versatile"
)

sales_agent3 = Agent(
    name="Busy Sales Agent",
    instructions=instructions3,
    model="llama-3.3-70b-versatile"
)

# ============================================================
# RUN AGENT
# ============================================================

# OPENAI VERSION
#
# result = await Runner.run(
#     sales_agent1,
#     "Write a cold sales email"
# )
#
# print(result.final_output)

# GROQ VERSION
#
# Why?
#
# Runner.run() is an OpenAI SDK feature.
#
# We manually call Groq Chat API.


async def run_agent(agent, prompt):

    print(f"\nRunning Agent: {agent.name}")

    response = client.chat.completions.create(

        model=agent.model,

        messages=[

            # Agent personality
            {
                "role": "system",
                "content": agent.instructions
            },

            # User task
            {
                "role": "user",
                "content": prompt
            }

        ]
    )

    return response.choices[0].message.content


# ============================================================
# SALES MANAGER
# ============================================================

# OPENAI VERSION
#
# sales_picker = Agent(
#     name="sales_picker",
#     instructions="Pick best email",
#     model="gpt-4o-mini"
# )
#
# best = await Runner.run(
#     sales_picker,
#     emails
# )

# GROQ VERSION
#
# For simplicity:
# Pick the longest email.
#
# Later you can replace this with another LLM call.

def select_best_email(emails):

    print("\nSelecting Best Email...")

    best_email = max(emails, key=len)

    return best_email


# ============================================================
# PARALLEL EXECUTION
# ============================================================

# OPENAI VERSION
#
# with trace("Parallel cold emails"):
#
#     results = await asyncio.gather(
#
#         Runner.run(sales_agent1, message),
#         Runner.run(sales_agent2, message),
#         Runner.run(sales_agent3, message)
#
#     )

# GROQ VERSION
#
# asyncio.gather() still works.
#
# Only Runner.run() changes.


async def generate_emails():

    message = """
    Write a cold sales email for ComplAI
    targeted at startup CTOs.
    """

    print("\n")
    print("=" * 80)
    print("GENERATING EMAILS IN PARALLEL")
    print("=" * 80)

    results = await asyncio.gather(

        run_agent(
            sales_agent1,
            message
        ),

        run_agent(
            sales_agent2,
            message
        ),

        run_agent(
            sales_agent3,
            message
        )
    )

    print("\n")
    print("=" * 80)
    print("PROFESSIONAL EMAIL")
    print("=" * 80)

    print(results[0])

    print("\n")
    print("=" * 80)
    print("ENGAGING EMAIL")
    print("=" * 80)

    print(results[1])

    print("\n")
    print("=" * 80)
    print("BUSY EMAIL")
    print("=" * 80)

    print(results[2])

    return results


# ============================================================
# ORCHESTRATOR
# ============================================================

async def sales_manager():

    emails = await generate_emails()

    best_email = select_best_email(emails)

    print("\n")
    print("=" * 80)
    print("SELECTED EMAIL")
    print("=" * 80)

    print(best_email)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        sales_manager()
    )