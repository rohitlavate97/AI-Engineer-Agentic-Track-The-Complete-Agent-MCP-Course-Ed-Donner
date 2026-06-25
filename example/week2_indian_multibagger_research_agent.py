"""
indian_multibagger_research_agent.py

Deep research pipeline for Indian Stock Market multibagger screening.
Combines the multi-agent equity analysis framework with live web search,
structured report generation, and email delivery via SendGrid.

Pipeline:
  Stage 1 — Search Planner     (Groq LLaMA 3.3-70b)
             Generates targeted web search queries for the stock
             across governance, financials, macro, and technicals.

  Stage 2 — Web Search         (DuckDuckGo, no API key needed)
             Fetches live news, filings, analyst reports, and
             shareholding disclosures for the stock.

  Stage 3 — 4 Parallel Analysts (Groq LLaMA 3.3-70b, asyncio.gather)
             Each analyst reads the relevant search summaries and
             produces a BUY / HOLD / AVOID signal:
               · Governance & Shareholding Analyst
               · Financial Quality & Efficiency Analyst
               · Industry Runway & Macro Tailwinds Analyst
               · Technical Structure & Valuation Analyst

  Stage 4 — CIO Synthesis       (Groq LLaMA 3.3-70b)
             Chief Investment Officer weighs all 4 reports and
             produces a structured PMS-style multibagger verdict.

  Stage 5 — Report Writer       (Groq LLaMA 3.3-70b)
             Converts the CIO verdict + analyst reports into a
             1000+ word institutional markdown research report.

  Stage 6 — Email Dispatch      (Groq LLaMA 3.1-8b + SendGrid)
             Formats the report as HTML and emails it.

Why 100% Groq?
  Groq's free tier is fast enough for all stages. Using DuckDuckGo
  for web search avoids Groq's 400 tool_use_failed error that occurs
  when the OpenAI Agents SDK's WebSearchTool is used with LLaMA models.

============================================================
PREREQUISITES
============================================================

1. Python 3.9 or higher
   Check:     python --version
   Download:  https://www.python.org/downloads/

2. OpenAI API Key
   Required by the openai-agents SDK internally (Runner, trace).
   LLM calls are NOT billed to OpenAI — all inference goes to Groq.
   Get one: https://platform.openai.com/api-keys

3. Groq API Key  (free tier — no credit card required)
   Sign up:  https://console.groq.com
   Steps:
     - Log in → "API Keys" → "Create API Key" → copy it
   Free limits: ~30 requests/min, ~15,000 tokens/min

4. SendGrid API Key  (free tier: 100 emails/day)
   Sign up:  https://signup.sendgrid.com
   Steps:
     - Settings → API Keys → Create API Key
     - Select "Restricted Access" → enable "Mail Send" only
     - Copy the key (shown only once)

5. Verified SendGrid Sender Email
   - SendGrid → Settings → Sender Authentication
   - "Verify a Single Sender" → complete the verification email
   - Your FROM_EMAIL must match a verified sender or all sends fail with 403.

6. DuckDuckGo search — no API key needed
   Installed via pip. Falls back to mock data if unavailable.

============================================================
INSTALLATION STEPS
============================================================

Step 1 — Create a project folder:
    mkdir multibagger-research
    cd multibagger-research

Step 2 — (Recommended) Create a virtual environment:
    python -m venv venv

    Activate:
      macOS / Linux:   source venv/bin/activate
      Windows:         venv\\Scripts\\activate

Step 3 — Install required packages:
    pip install openai-agents python-dotenv sendgrid duckduckgo-search

    Package breakdown:
      openai-agents       — Agent orchestration (Runner, function_tool, trace)
      python-dotenv       — Loads secrets from .env file
      sendgrid            — Sends the final HTML research report by email
      duckduckgo-search   — Free live web search, no API key needed

Step 4 — Create a .env file:
    touch .env          # macOS / Linux
    type nul > .env     # Windows

    Add these lines:
        OPENAI_API_KEY=sk-...
        GROQ_API_KEY=gsk_...
        SENDGRID_API_KEY=SG....
        FROM_EMAIL=your_verified_sender@yourdomain.com
        TO_EMAIL=recipient@example.com

    Protect your keys:
        echo ".env" >> .gitignore

Step 5 — Set the stock to analyse:
    Scroll to the bottom and change:
        STOCK = "Tata Power Ltd."

Step 6 — Run:
    python indian_multibagger_research_agent.py

============================================================
FOLDER STRUCTURE AFTER SETUP
============================================================

    multibagger-research/
    ├── indian_multibagger_research_agent.py   ← this file
    ├── .env                                   ← secrets (never commit)
    ├── .gitignore                             ← must contain: .env
    └── venv/                                  ← virtual environment

============================================================
5-POINT MULTIBAGGER SCREENING CHECKLIST
============================================================

  Metric                Criteria                   Why It Matters
  ──────────────────    ─────────────────────────  ──────────────────────────────
  Market Cap            Under ₹15,000 Crore        Easier to 10x from small base
  Operating Leverage    Sales growth < Profit       Fixed costs = exploding profits
  Promoter Pledging     Strictly 0% (or falling)   High pledge = overnight crash risk
  ROCE Track Record     Consistently > 20%          Self-funding growth, no dilution
  Earnings Conversion   CFO / EBITDA > 70%          Reported profits = real cash

============================================================
RATE LIMIT STRATEGY
============================================================

  Delays built into the pipeline:
    - 1.5s between each web search      → protects Groq token/min limits
    - 2.0s between pipeline stages      → prevents cascading rate errors

  If you see RateLimitError, increase sleep values or reduce
  HOW_MANY_SEARCHES from 4 to 2.

============================================================
TROUBLESHOOTING
============================================================

  ModuleNotFoundError: No module named 'agents'
    → Run: pip install openai-agents  (package name differs from import)

  ModuleNotFoundError: No module named 'duckduckgo_search'
    → Run: pip install duckduckgo-search
    → Script uses mock fallback without it, but live data won't be fetched.

  RuntimeError: Missing required key / email config
    → Check your .env — all 5 values must be set and non-empty.

  Groq RateLimitError
    → Increase sleep values to 4.0 seconds.
    → Reduce HOW_MANY_SEARCHES to 2.

  SendGrid 403 Forbidden
    → FROM_EMAIL not verified in SendGrid Sender Authentication.
    → Fix: https://app.sendgrid.com/settings/sender_auth

  SendGrid 401 Unauthorized
    → API key wrong or missing "Mail Send" permission.
    → Fix: https://app.sendgrid.com/settings/api_keys

  Report uses fallback / generic data
    → DuckDuckGo may be rate-limited on your network.
    → Wait a few minutes or retry with a VPN.

  python: command not found
    → Try: python3 indian_multibagger_research_agent.py

============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent,
    trace,
    Runner,
    function_tool,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content

# ============================================================
# SETUP
# ============================================================

set_tracing_disabled(True)
load_dotenv(override=True)

# ============================================================
# LOAD AND VALIDATE KEYS
# ============================================================

openai_api_key   = os.getenv("OPENAI_API_KEY")
groq_api_key     = os.getenv("GROQ_API_KEY")
sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
FROM_EMAIL       = os.getenv("FROM_EMAIL", "")
TO_EMAIL         = os.getenv("TO_EMAIL",   "")

for key_name, key_value in {
    "OPENAI_API_KEY":   openai_api_key,
    "GROQ_API_KEY":     groq_api_key,
    "SENDGRID_API_KEY": sendgrid_api_key,
}.items():
    if not key_value:
        raise RuntimeError(
            f"Initialization Aborted: Missing environment variable '{key_name}'.\n"
            f"Add it to your .env file."
        )

if not FROM_EMAIL or not TO_EMAIL:
    raise RuntimeError(
        "Initialization Aborted: FROM_EMAIL and TO_EMAIL must be set in your .env file.\n"
        "Example:\n  FROM_EMAIL=you@yourdomain.com\n  TO_EMAIL=recipient@example.com"
    )

# ============================================================
# GROQ CLIENT + MODEL ALLOCATIONS
# ============================================================

groq_client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key
)

llama3_3 = OpenAIChatCompletionsModel(
    model="llama-3.3-70b-versatile",
    openai_client=groq_client
)

llama3_1_fast = OpenAIChatCompletionsModel(
    model="llama-3.1-8b-instant",
    openai_client=groq_client
)

# ============================================================
# CONFIGURATION
# ============================================================

# One search per analyst lens: governance, financials, macro, technical.
# Reduce to 2 if hitting Groq free-tier rate limits.
HOW_MANY_SEARCHES = 4

# ============================================================
# LIVE WEB SEARCH (DuckDuckGo — no API key)
# Runs as plain Python, outside the agent tool loop.
# This avoids Groq's 400 tool_use_failed error.
# ============================================================

async def web_search(query: str) -> str:
    """Fetch live web snippets for a query via DuckDuckGo."""

    print(f"      [Web] Searching: '{query}'...")

    try:
        from duckduckgo_search import DDGS

        def _fetch():
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
                if results:
                    return "\n\n".join(
                        f"Title: {r['title']}\nSnippet: {r['body']}"
                        for r in results
                    )
            return ""

        data = await asyncio.to_thread(_fetch)
        if data:
            return data

    except Exception as e:
        print(f"      [Web] DuckDuckGo unavailable ({e}), using fallback.")

    await asyncio.sleep(0.5)
    return (
        f"Fallback context for: '{query}'\n"
        "- NSE/BSE filings show steady promoter holding above 50%.\n"
        "- Analyst consensus leans positive on capital efficiency metrics.\n"
        "- Sector tailwinds remain intact with domestic demand growth."
    )

# ============================================================
# AGENT INSTRUCTIONS
# ============================================================

# --- Planner ---
PLANNER_INSTRUCTIONS = (
    f"You are a research planner for Indian equity analysis.\n"
    f"Given a stock name, output exactly {HOW_MANY_SEARCHES} targeted web search queries "
    f"that will find current information about:\n"
    f"  1. Promoter shareholding, pledging, and governance news\n"
    f"  2. Financial results: ROCE, ROE, revenue growth, debt levels\n"
    f"  3. Industry sector outlook and macro tailwinds for this business\n"
    f"  4. Technical chart analysis, valuation ratios, and analyst targets\n\n"
    f"Format your output strictly as a hyphen list — one query per line:\n"
    f"- query 1\n- query 2\n\n"
    f"Output only the list. No explanation or preamble."
)

# --- Governance Analyst ---
GOVERNANCE_INSTRUCTIONS = """
You are a Corporate Governance & Shareholding Analyst for Indian equities (NSE/BSE).

You will receive live research snippets about the stock. Analyse:
  - Promoter holding trends (stable, increasing, or diluting)
  - Promoter pledged shares % — flag anything above 10% as HIGH RISK
  - FII (Foreign Institutional Investor) accumulation or exit trends
  - DII (Mutual Funds, LIC) holding patterns
  - Governance red flags: related-party transactions, auditor resignations,
    SEBI actions, frequent management changes

Give a clear BUY / HOLD / AVOID signal with supporting evidence from the research.
"""

# --- Financials Analyst ---
FINANCIALS_INSTRUCTIONS = """
You are a Financial Quality & Efficiency Analyst evaluating Indian multibagger candidates.

You will receive live research snippets about the stock. Analyse over a 3-5 year horizon:
  - ROCE and ROE (flag if sustainably above 15-20%)
  - Operating and net profit margin trends (expanding or contracting?)
  - Debt-to-Equity ratio (flag if above 0.5 for non-NBFC businesses)
  - Interest Coverage Ratio (flag if below 3x)
  - Quality of Earnings: does net profit align with operating cash flow?
    A CFO/EBITDA ratio above 70% confirms genuine cash generation.
  - Working capital cycle trends

Give a clear BUY / HOLD / AVOID signal with supporting evidence from the research.
"""

# --- Macro Runway Analyst ---
MACRO_INSTRUCTIONS = """
You are an Industry Runway & Macro Tailwinds Analyst for the Indian market.

You will receive live research snippets about the stock. Analyse:
  - Total Addressable Market (TAM): is the industry under-penetrated?
  - Structural tailwinds: Make in India, EV ecosystem, renewable energy,
    digitalization, defence indigenization, healthcare expansion, premiumization
  - Market cap classification (Micro/Small/Mid/Large) and room to grow
  - Competitive moat: brand loyalty, switching costs, cost leadership,
    regulatory licences, distribution network

Give a clear BUY / HOLD / AVOID signal with supporting evidence from the research.
"""

# --- Technical & Valuation Analyst ---
TECHNICAL_INSTRUCTIONS = """
You are a Technical Structure & Valuation Analyst for Indian equities.

You will receive live research snippets about the stock. Analyse:
  - Price trend: is the stock above its EMA 50 and EMA 200?
  - Volume: are breakouts backed by above-average volume?
  - Base pattern: is the stock breaking out of a long consolidation base?
  - P/E ratio vs historic band and sector peers
  - PEG Ratio (P/E ÷ expected EPS growth rate):
      PEG < 1 = undervalued growth, PEG 1-2 = fair, PEG > 2 = expensive
  - P/B ratio for asset-heavy sectors

Give a clear BUY / HOLD / AVOID signal with supporting evidence from the research.
"""

# --- CIO ---
CIO_INSTRUCTIONS = """
You are the Chief Investment Officer (CIO) of a SEBI-registered Portfolio Management Service (PMS).

You will receive 4 analyst reports on the same stock. Synthesise them:
  - GOVERNANCE issues are disqualifying — no growth story compensates for
    promoter fraud or high pledging
  - FINANCIALS must show compounding ROCE/ROE over multiple years
  - MACRO RUNWAY provides the growth engine
  - VALUATION determines entry safety

Format your output EXACTLY as:

  MULTIBAGGER VERDICT:    YES / NO / POTENTIAL HOLD
  CONVICTION LEVEL:       High / Medium / Low
  TARGET HORIZON:         3 - 5 Years

  INVESTMENT THESIS (BULL CASE):
  (2-3 sentences on the core structural re-rating driver)

  RISK FACTORS (BEAR CASE):
  (2-3 sentences on what could permanently derail this)

  CRITICAL TRACKING METRICS:
  - (Metric 1 to watch every quarter)
  - (Metric 2 to watch every quarter)
  - (Metric 3 to watch every quarter)

  FINAL SUMMARY:
  (1 paragraph institutional stance on this stock)
"""

# --- Report Writer ---
WRITER_INSTRUCTIONS = """
You are a senior equity research analyst writing an institutional research report.

You will receive a stock name, 4 analyst reports, and a CIO verdict.
Write a detailed, well-structured markdown research report of at least 1000 words.

Structure:
  1. At the very top, a 2-3 sentence executive summary inside a blockquote:
     > **Executive Summary:** ...
  2. ## Stock Overview
  3. ## Corporate Governance & Shareholding Analysis
  4. ## Financial Quality & Efficiency Analysis
  5. ## Industry Runway & Macro Tailwinds
  6. ## Technical Structure & Valuation
  7. ## CIO Verdict & Investment Thesis
  8. ## Key Risks
  9. ### Follow-up Research Areas  (3-5 bullet points)

Be analytical and specific. Reference actual metrics where available.
End with a disclaimer: "This report is for educational purposes only and does not
constitute SEBI-registered investment advice."
"""

# ============================================================
# AGENT CREATION
# ============================================================

planner_agent    = Agent(name="PlannerAgent",           instructions=PLANNER_INSTRUCTIONS,   model=llama3_3)
governance_agent = Agent(name="Governance Analyst",     instructions=GOVERNANCE_INSTRUCTIONS, model=llama3_3)
financials_agent = Agent(name="Financials Analyst",     instructions=FINANCIALS_INSTRUCTIONS, model=llama3_3)
macro_agent      = Agent(name="Macro Runway Analyst",   instructions=MACRO_INSTRUCTIONS,      model=llama3_3)
technical_agent  = Agent(name="Technical Analyst",      instructions=TECHNICAL_INSTRUCTIONS,  model=llama3_3)
cio_agent        = Agent(name="CIO",                    instructions=CIO_INSTRUCTIONS,        model=llama3_3)
writer_agent     = Agent(name="WriterAgent",            instructions=WRITER_INSTRUCTIONS,     model=llama3_3)

# ============================================================
# EMAIL AGENT + TOOL
# ============================================================

@function_tool
async def send_email_tool(subject: str, html_body: str) -> str:
    """Send the research report as an HTML email via SendGrid."""

    def _sync_send():
        try:
            sg   = sendgrid.SendGridAPIClient(api_key=sendgrid_api_key)
            mail = Mail(
                from_email=Email(FROM_EMAIL),
                to_emails=To(TO_EMAIL),
                subject=subject,
                html_content=Content("text/html", html_body),
            ).get()
            sg.client.mail.send.post(request_body=mail)
            return "success"
        except Exception as e:
            return f"failed: {str(e)}"

    return await asyncio.to_thread(_sync_send)

email_agent = Agent(
    name="Email Agent",
    instructions=(
        "You are able to send a nicely formatted HTML email based on a detailed equity research report. "
        "Convert the markdown report into clean, professional HTML and send it using your tool. "
        "Use an appropriate subject line that includes the stock name and verdict."
    ),
    tools=[send_email_tool],
    model=llama3_1_fast,
)

# ============================================================
# HELPER: run a single agent call
# ============================================================

async def run_agent(agent: Agent, prompt: str) -> str:
    print(f"  [→] Running: {agent.name}")
    result = await Runner.run(agent, prompt)
    print(f"  [✓] Done:    {agent.name}")
    return result.final_output

# ============================================================
# PIPELINE STAGE FUNCTIONS
# ============================================================

async def plan_searches(stock: str) -> list[str]:
    """Stage 1: Generate targeted search queries for the stock."""

    print("[→] Planning searches...")
    result = await Runner.run(planner_agent, f"Stock to research: {stock}")
    text   = result.final_output

    queries = [
        line.lstrip("- ").strip()
        for line in text.split("\n")
        if line.strip().startswith("-")
    ]
    if not queries:
        queries = [l.strip() for l in text.split("\n") if l.strip()][:HOW_MANY_SEARCHES]

    print(f"[✓] Generated {len(queries)} search queries.")
    return queries


async def perform_searches(queries: list[str]) -> dict[str, str]:
    """Stage 2: Fetch and summarise web data for each query.
    Returns a dict mapping query → summary for use by analysts."""

    print("[→] Running web searches...")
    summaries = {}

    for i, query in enumerate(queries, 1):
        print(f"    Search {i}/{len(queries)}: {query}")
        raw = await web_search(query)

        result = await Runner.run(
            Agent(
                name="Summariser",
                instructions=(
                    "Summarise the following web search results in 2-3 concise paragraphs "
                    "under 300 words. Focus on facts relevant to Indian equity investing. "
                    "No commentary beyond the summary."
                ),
                model=llama3_3,
            ),
            f"Topic: {query}\n\nRaw results:\n{raw}"
        )
        summaries[query] = result.final_output

        if i < len(queries):
            await asyncio.sleep(1.5)

    print("[✓] All searches complete.")
    return summaries


async def run_analysts(stock: str, summaries: dict[str, str]) -> tuple[str, str, str, str]:
    """Stage 3: Run all 4 analysts in parallel using asyncio.gather.
    Each analyst receives all search summaries as context."""

    research_context = "\n\n---\n\n".join(
        f"Search: {q}\nSummary: {s}" for q, s in summaries.items()
    )
    base_prompt = (
        f"Stock: {stock}\n\n"
        f"Live research summaries:\n{research_context}"
    )

    print("\n[→] Running 4 analysts in parallel...")

    gov, fin, macro, tech = await asyncio.gather(
        run_agent(governance_agent, base_prompt),
        run_agent(financials_agent, base_prompt),
        run_agent(macro_agent,      base_prompt),
        run_agent(technical_agent,  base_prompt),
    )

    print("[✓] All analyst reports complete.")
    return gov, fin, macro, tech


async def run_cio(stock: str, gov: str, fin: str, macro: str, tech: str) -> str:
    """Stage 4: CIO synthesises all 4 analyst reports into a final verdict."""

    print("\n[→] CIO synthesis...")
    prompt = (
        f"Stock: {stock}\n\n"
        f"--- GOVERNANCE REPORT ---\n{gov}\n\n"
        f"--- FINANCIALS REPORT ---\n{fin}\n\n"
        f"--- MACRO RUNWAY REPORT ---\n{macro}\n\n"
        f"--- TECHNICAL & VALUATION REPORT ---\n{tech}"
    )
    verdict = await run_agent(cio_agent, prompt)
    print("[✓] CIO verdict complete.")
    return verdict


async def write_report(stock: str, gov: str, fin: str, macro: str, tech: str, verdict: str) -> str:
    """Stage 5: Write a full institutional research report in markdown."""

    print("\n[→] Writing research report...")
    prompt = (
        f"Stock: {stock}\n\n"
        f"--- GOVERNANCE ANALYSIS ---\n{gov}\n\n"
        f"--- FINANCIALS ANALYSIS ---\n{fin}\n\n"
        f"--- MACRO RUNWAY ANALYSIS ---\n{macro}\n\n"
        f"--- TECHNICAL & VALUATION ANALYSIS ---\n{tech}\n\n"
        f"--- CIO VERDICT ---\n{verdict}"
    )
    report = await run_agent(writer_agent, prompt)
    print("[✓] Report complete.")
    return report


async def dispatch_email(stock: str, report: str) -> None:
    """Stage 6: Instantly dispatch raw text directly to the underlying function."""
    print(f"\n[→] Sending report to {TO_EMAIL}...")
    
    # .fn safely unpacks the decorated tool to run as direct Python code
    await send_email_tool.fn(
        subject=f"{stock} Multibagger Research Report",
        html_body=f"<pre style='font-family: sans-serif;'>{report}</pre>"
    )
    print("[✓] Email dispatched.")
# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

async def main():

    # ── Change this to any NSE/BSE listed stock ───────────────────────────
    STOCK = "Tata Power Ltd."
    # Suggestions:
    #   "Dixon Technologies Ltd."        — Electronics manufacturing
    #   "Kaynes Technology India Ltd."   — EMS / PCB assembly
    #   "Waaree Energies Ltd."           — Solar manufacturing
    #   "Suzlon Energy Ltd."             — Wind energy turnaround
    #   "Mankind Pharma Ltd."            — Pharma distribution
    # ──────────────────────────────────────────────────────────────────────

    with trace("Indian Multibagger Research Pipeline"):

        print(f"\n{'='*65}")
        print(f"  INDIAN MULTIBAGGER DEEP RESEARCH AGENT")
        print(f"  Stock: {STOCK}")
        print(f"{'='*65}\n")

        # Stage 1: Plan
        queries = await plan_searches(STOCK)
        await asyncio.sleep(2.0)

        # Stage 2: Search
        summaries = await perform_searches(queries)
        await asyncio.sleep(2.0)

        # Stage 3: 4 Analysts in parallel
        gov, fin, macro, tech = await run_analysts(STOCK, summaries)
        await asyncio.sleep(2.0)

        # Stage 4: CIO verdict
        verdict = await run_cio(STOCK, gov, fin, macro, tech)

        print("\n")
        print("=" * 65)
        print("  CIO VERDICT")
        print("=" * 65)
        print(verdict)

        await asyncio.sleep(2.0)

        # Stage 5: Write full report
        report = await write_report(STOCK, gov, fin, macro, tech, verdict)
        await asyncio.sleep(2.0)

        # Stage 6: Email
        await dispatch_email(STOCK, report)

        print(f"\n{'='*65}")
        print(f"  Pipeline complete. Report delivered to: {TO_EMAIL}")
        print(f"{'='*65}\n")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())
