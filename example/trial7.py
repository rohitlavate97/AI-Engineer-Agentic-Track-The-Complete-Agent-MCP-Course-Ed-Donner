# ================================================================
#   AGENTIC AI — Personal Assistant with Tool Calling
#   Uses: Groq (agent) + Pushover (notifications) + Gradio (UI)
# ================================================================
# pip install openai groq gradio pypdf python-dotenv requests

import os
import json
import requests
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
import gradio as gr

load_dotenv(override=True)

# ── Initialize Clients ───────────────────────────────────────────
# openai = OpenAI()  # ← No credits, using Groq instead
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Pushover Setup ───────────────────────────────────────────────
pushover_user  = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url   = "https://api.pushover.net/1/messages.json"

if pushover_user:
    print(f"Pushover user found and starts with {pushover_user[0]}")
else:
    print("Pushover user not found")

if pushover_token:
    print(f"Pushover token found and starts with {pushover_token[0]}")
else:
    print("Pushover token not found")


# ── Pushover Function ────────────────────────────────────────────
def push(message):
    print(f"Push: {message}")
    if pushover_user and pushover_token:
        payload = {"user": pushover_user, "token": pushover_token, "message": message}
        requests.post(pushover_url, data=payload)


# ── Tools (actual Python functions) ─────────────────────────────
def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording interest from {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question} asked that I couldn't answer")
    return {"recorded": "ok"}


# ── Tools JSON (for LLM) ─────────────────────────────────────────
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            },
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            }
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json}
]


# ── Handle Tool Calls (elegant — no IF statement) ────────────────
def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}", flush=True)
        tool   = globals().get(tool_name)
        result = tool(**arguments) if tool else {}
        results.append({
            "role":         "tool",
            "content":      json.dumps(result),
            "tool_call_id": tool_call.id
        })
    return results


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

# ── System Prompt ────────────────────────────────────────────────
system_prompt  = f"You are acting as {name}. You are answering questions on {name}'s website, "
system_prompt += f"particularly questions related to {name}'s career, background, skills and experience. "
system_prompt += f"Your responsibility is to represent {name} for interactions on the website as faithfully as possible. "
system_prompt += f"You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions. "
system_prompt += f"Be professional and engaging, as if talking to a potential client or future employer who came across the website. "
system_prompt += f"If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. "
system_prompt += f"If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "
system_prompt += f"\n\n## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n\n"
system_prompt += f"With this context, please chat with the user, always staying in character as {name}."


# ── Main Chat Function ───────────────────────────────────────────
def chat(message, history):
    # Convert history to messages format
    history_messages = []
    for item in history:
        if isinstance(item, dict):
            history_messages.append({"role": item["role"], "content": item["content"]})
        else:
            human, assistant = item[0], item[1]
            history_messages.append({"role": "user",      "content": human})
            history_messages.append({"role": "assistant", "content": assistant})

    messages = [{"role": "system", "content": system_prompt}] + \
               history_messages + \
               [{"role": "user", "content": message}]

    done = False
    while not done:
        # Call LLM with tools
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",  # free on Groq
            messages=messages,
            tools=tools
        )

        finish_reason = response.choices[0].finish_reason

        # If LLM wants to call a tool
        if finish_reason == "tool_calls":
            msg        = response.choices[0].message
            tool_calls = msg.tool_calls
            results    = handle_tool_calls(tool_calls)
            messages.append(msg)
            messages.extend(results)
        else:
            done = True

    return response.choices[0].message.content


# ── Launch Gradio UI ─────────────────────────────────────────────
gr.ChatInterface(chat).launch()


# ================================================================
#   HOW IT WORKS
#
#   User chats
#         ↓
#   Groq generates reply OR requests tool call
#         ↓
#   Tool call? → handle_tool_calls() executes it
#              → sends Pushover notification to phone
#              → result sent back to LLM
#              → LLM generates final reply
#         ↓
#   No tool call? → return reply directly
#
#   TOOLS:
#   record_user_details   → saves email + sends push notification
#   record_unknown_question → saves question + sends push notification
#
#   FOLDER STRUCTURE:
#   agents/
#   ├── 1_foundations/
#   │   └── me/
#   │       ├── linkedin.pdf   ← your LinkedIn PDF
#   │       └── summary.txt    ← your summary
#   └── example/
#       └── trial7.py          ← this script
#
#   .env file:
#   GROQ_API_KEY=gsk_...
#   PUSHOVER_USER=u...
#   PUSHOVER_TOKEN=a...
# ================================================================