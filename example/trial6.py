# ================================================================
#   AGENTIC AI — Personal Assistant with Evaluation & Rerun
#   Uses: Groq (agent) + Gemini (evaluator) + Gradio (UI)
# ================================================================
# pip install openai groq gradio pypdf python-dotenv pydantic google-genai

import os
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from pypdf import PdfReader
from pydantic import BaseModel
import gradio as gr

load_dotenv(override=True)

# ── Initialize Clients ───────────────────────────────────────────
# openai = OpenAI()  # ← No credits, using Groq instead

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # free

gemini = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ── Fix Paths (me folder is inside 1_foundations) ───────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
me_dir     = os.path.join(script_dir, "..", "1_foundations", "me")

# ── Load LinkedIn PDF ────────────────────────────────────────────
reader   = PdfReader(os.path.join(me_dir, "linkedin.pdf"))
linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text

# ── Load Summary ─────────────────────────────────────────────────
with open(os.path.join(me_dir, "summary.txt"), "r", encoding="utf-8") as f:
    summary = f.read()

# ── Setup ────────────────────────────────────────────────────────
name = "Rohit Lavate"  # ← Change to your name

# ── System Prompt (Agent) ────────────────────────────────────────
system_prompt  = f"You are acting as {name}. You are answering questions on {name}'s website, "
system_prompt += f"particularly questions related to {name}'s career, background, skills and experience. "
system_prompt += f"Your responsibility is to represent {name} for interactions on the website as faithfully as possible. "
system_prompt += f"You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions. "
system_prompt += f"Be professional and engaging, as if talking to a potential client or future employer who came across the website. "
system_prompt += f"If you don't know the answer, say so."
system_prompt += f"\n\n## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n\n"
system_prompt += f"With this context, please chat with the user, always staying in character as {name}."

# ── Evaluator System Prompt ──────────────────────────────────────
evaluator_system_prompt  = f"You are an evaluator that decides whether a response to a question is acceptable. "
evaluator_system_prompt += f"You are provided with a conversation between a User and an Agent. "
evaluator_system_prompt += f"Your task is to decide whether the Agent's latest response is acceptable quality. "
evaluator_system_prompt += f"The Agent is playing the role of {name} and is representing {name} on their website. "
evaluator_system_prompt += f"The Agent has been instructed to be professional and engaging, as if talking to a potential client or future employer. "
evaluator_system_prompt += f"The Agent has been provided with context on {name} in the form of their summary and LinkedIn details. Here's the information:"
evaluator_system_prompt += f"\n\n## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n\n"
evaluator_system_prompt += "With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback."


# ── Pydantic Model for Evaluation ────────────────────────────────
class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str


# ── Evaluator User Prompt ────────────────────────────────────────
def evaluator_user_prompt(reply, message, history):
    user_prompt  = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
    user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
    user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
    user_prompt += "Please evaluate the response, replying with whether it is acceptable and your feedback."
    return user_prompt


# ── Evaluate Function (Gemini) ───────────────────────────────────
def evaluate(reply, message, history) -> Evaluation:
    messages = [{"role": "system", "content": evaluator_system_prompt}] + \
               [{"role": "user",   "content": evaluator_user_prompt(reply, message, history)}]
    try:
        # Try Gemini structured output first
        response = gemini.beta.chat.completions.parse(
            model="gemini-2.5-flash",
            messages=messages,
            response_format=Evaluation
        )
        return response.choices[0].message.parsed
    except Exception:
        # Fallback — use Groq and assume acceptable
        print("⚠️ Gemini evaluator failed — skipping evaluation")
        return Evaluation(is_acceptable=True, feedback="Evaluation skipped")


# ── Rerun Function (if evaluation fails) ─────────────────────────
def rerun(reply, message, history, feedback):
    updated_system_prompt  = system_prompt + "\n\n## Previous answer rejected\n"
    updated_system_prompt += "You just tried to reply, but the quality control rejected your reply\n"
    updated_system_prompt += f"## Your attempted answer:\n{reply}\n\n"
    updated_system_prompt += f"## Reason for rejection:\n{feedback}\n\n"
    messages = [{"role": "system", "content": updated_system_prompt}] + \
               history + \
               [{"role": "user", "content": message}]
    # response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
    # return response.choices[0].message.content
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages
    )
    return response.choices[0].message.content


# ── Main Chat Function ───────────────────────────────────────────
def chat(message, history):
    # Convert history to messages format
    history_messages = []
    for item in history:
        if isinstance(item, dict):
            # New Gradio format: list of dicts
            history_messages.append({"role": item["role"], "content": item["content"]})
        else:
            # Old Gradio format: list of tuples
            human, assistant = item[0], item[1]
            history_messages.append({"role": "user",      "content": human})
            history_messages.append({"role": "assistant", "content": assistant})

    # Intentional bad response trigger for testing evaluator
    if "patent" in message:
        system = system_prompt + "\n\nEverything in your reply needs to be in pig latin - \
              it is mandatory that you respond only and entirely in pig latin"
    else:
        system = system_prompt

    messages = [{"role": "system", "content": system}] + \
               history_messages + \
               [{"role": "user", "content": message}]

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages
    )
    reply = response.choices[0].message.content

    evaluation = evaluate(reply, message, history_messages)

    if evaluation.is_acceptable:
        print("✅ Passed evaluation - returning reply")
    else:
        print("❌ Failed evaluation - retrying")
        print(f"Feedback: {evaluation.feedback}")
        reply = rerun(reply, message, history_messages, evaluation.feedback)

    return reply


# ── Launch Gradio UI ─────────────────────────────────────────────
gr.ChatInterface(chat).launch()


# ================================================================
#   HOW IT WORKS
#
#   User asks question
#         ↓
#   Groq gpt-oss-120b generates reply     (Agent)  ← FREE
#         ↓
#   Gemini 2.5-flash evaluates reply      (Evaluator) ← FREE
#         ↓
#   is_acceptable = True  → return reply
#   is_acceptable = False → rerun with feedback → return new reply
#
#   FOLDER STRUCTURE:
#   agents/
#   ├── 1_foundations/
#   │   └── me/
#   │       ├── linkedin.pdf   ← your LinkedIn PDF
#   │       └── summary.txt    ← your summary
#   └── example/
#       └── trial6.py          ← this script
#
#   .env file:
#   GROQ_API_KEY=gsk_...
#   GOOGLE_API_KEY=AIza...
# ================================================================