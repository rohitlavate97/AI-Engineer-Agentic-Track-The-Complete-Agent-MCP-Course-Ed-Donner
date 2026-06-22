import os
import json
from dotenv import load_dotenv
from groq import Groq
# import ollama  # Uncomment after: pip install ollama
import google.genai as genai
from IPython.display import Markdown, display

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()
groq_api_key   = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GEMINI_API_KEY")

print(f"Groq API Key:   {groq_api_key[:4] if groq_api_key else 'Not set'}")
print(f"Gemini API Key: {google_api_key[:4] if google_api_key else 'Not set'}")

# ── Initialize clients ───────────────────────────────────────────────────────
groq_client   = Groq(api_key=groq_api_key)
gemini_client = genai.Client(api_key=google_api_key)

# ── Helper: print for both terminal and notebook ─────────────────────────────
def show(text):
    print(text)           # terminal
    display(Markdown(text))  # notebook

# ── Step 1: Generate a challenging question ──────────────────────────────────
request  = "Please come up with a challenging, nuanced question that I can ask a number of LLMs to evaluate their intelligence. "
request += "Answer only with the question, no explanation."
messages = [{"role": "user", "content": request}]

response = groq_client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=messages,
)
question = response.choices[0].message.content
print("Question:", question)

# ── Step 2: Each competitor answers the question ─────────────────────────────
competitors = []
answers     = []
messages    = [{"role": "user", "content": question}]

# --- Groq models (cloud, free) — all current as of June 2026 ---
groq_models = [
    "openai/gpt-oss-120b",              # OpenAI OSS — most capable on Groq
    "openai/gpt-oss-20b",               # OpenAI OSS — faster/lighter
    "qwen/qwen3.6-27b",                 # Qwen — strong reasoning
    "meta-llama/llama-3.1-8b-instant",  # Llama — fast
]

for model_name in groq_models:
    try:
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        answer = response.choices[0].message.content
        competitors.append(f"groq/{model_name}")
        answers.append(answer)
        print(f"\n{'='*50}")
        print(f"Model: {model_name} (Groq)")
        print(f"{'='*50}")
        show(answer)
    except Exception as e:
        print(f"Groq model '{model_name}' failed: {e}")

# --- Gemini (cloud, free tier) ---
try:
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=question,
    )
    answer = response.text
    competitors.append("google/gemini-2.0-flash")
    answers.append(answer)
    print(f"\n{'='*50}")
    print(f"Model: gemini-2.0-flash (Google)")
    print(f"{'='*50}")
    show(answer)
except Exception as e:
    print(f"Gemini failed (rate limit — wait 1 min and retry): {e}")

# --- Ollama models (local, fully free) ---
# Uncomment after: pip install ollama
# ollama_models = ["llama3.2"]
# for model_name in ollama_models:
#     try:
#         response = ollama.chat(model=model_name, messages=messages)
#         answer = response["message"]["content"]
#         competitors.append(f"ollama/{model_name}")
#         answers.append(answer)
#         print(f"\n{'='*50}")
#         print(f"Model: {model_name} (Ollama - Local)")
#         print(f"{'='*50}")
#         show(answer)
#     except Exception as e:
#         print(f"Ollama model '{model_name}' failed: {e}")

# ── Step 3: Judge the answers ────────────────────────────────────────────────
judge_prompt  = "You are an expert judge evaluating LLM responses.\n\n"
judge_prompt += f"The question was: {question}\n\n"

for i, (competitor, answer) in enumerate(zip(competitors, answers)):
    judge_prompt += f"## Answer {i+1} — {competitor}\n{answer}\n\n"

judge_prompt += "Please rank these answers from best to worst, explaining your reasoning."

judge_messages = [{"role": "user", "content": judge_prompt}]

response = groq_client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=judge_messages,
)
results = response.choices[0].message.content
print("\n\n===== JUDGEMENT =====")
show(results)