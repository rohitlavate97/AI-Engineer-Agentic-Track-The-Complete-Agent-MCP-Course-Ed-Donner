"""
stock_agents_groq.py

Multi-agent stock market analyzer using Groq.

Architecture:
  - 3 parallel analyst agents (technical, fundamental, sentiment)
  - 1 portfolio manager agent that synthesizes all 3 reports
  - asyncio.gather() for parallel execution (same as sales_agents_groq.py)

Usage:
    python stock_agents_groq.py
    or change TICKER below to analyse any stock.
"""

# ============================================================
# IMPORTS
# ============================================================

from dotenv import load_dotenv
from groq import Groq
import asyncio
import os

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
# AGENT CLASS (same lightweight pattern as sales agents)
# ============================================================

class Agent:

    def __init__(self, name, instructions, model):
        self.name = name
        self.instructions = instructions
        self.model = model


# ============================================================
# ANALYST INSTRUCTIONS
# ============================================================

technical_instructions = """
You are a professional technical analyst at a hedge fund.

Analyse the given stock from a technical perspective only.

Cover: price action, trend direction, RSI, MACD,
moving averages, key support/resistance levels,
and volume patterns.

Give a clear BUY / HOLD / SELL signal with reasoning.
"""

fundamental_instructions = """
You are a professional fundamental analyst at a hedge fund.

Analyse the given stock from a fundamental perspective only.

Cover: valuation (P/E, P/S, EV/EBITDA), revenue growth,
profit margins, balance sheet health, competitive moat,
and management quality.

Give a clear BUY / HOLD / SELL signal with reasoning.
Note any significant risks.
"""

sentiment_instructions = """
You are a professional market sentiment analyst at a hedge fund.

Analyse the given stock from a sentiment perspective only.

Cover: recent news headlines, analyst upgrades/downgrades,
insider buying/selling, institutional ownership changes,
social media buzz, and overall market mood.

Give a clear BUY / HOLD / SELL signal with reasoning.
"""

portfolio_manager_instructions = """
You are a senior portfolio manager at a hedge fund.

You will receive three analyst reports on the same stock:
  1. Technical analysis
  2. Fundamental analysis
  3. Sentiment analysis

Your job:
  - Weigh each report
  - Identify where they agree or conflict
  - Produce a final investment recommendation

Format your output as:
  FINAL VERDICT: BUY / HOLD / SELL
  CONVICTION: High / Medium / Low
  SUMMARY: (2-3 sentences)
  KEY RISKS: (bullet points)
"""

# ============================================================
# AGENT CREATION
# ============================================================

technical_agent = Agent(
    name="Technical Analyst",
    instructions=technical_instructions,
    model="llama-3.3-70b-versatile"
)

fundamental_agent = Agent(
    name="Fundamental Analyst",
    instructions=fundamental_instructions,
    model="llama-3.3-70b-versatile"
)

sentiment_agent = Agent(
    name="Sentiment Analyst",
    instructions=sentiment_instructions,
    model="llama-3.3-70b-versatile"
)

portfolio_manager = Agent(
    name="Portfolio Manager",
    instructions=portfolio_manager_instructions,
    model="llama-3.3-70b-versatile"
)

# ============================================================
# RUN SINGLE AGENT
# ============================================================

async def run_agent(agent, prompt):

    print(f"\n[→] Running: {agent.name}")

    response = client.chat.completions.create(

        model=agent.model,

        messages=[
            {
                "role": "system",
                "content": agent.instructions
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = response.choices[0].message.content
    print(f"[✓] Done:    {agent.name}")

    return result


# ============================================================
# PARALLEL ANALYST EXECUTION
# ============================================================

async def run_analysts(ticker):

    prompt = f"Analyse {ticker} stock."

    print("\n")
    print("=" * 70)
    print(f"  RUNNING 3 ANALYSTS IN PARALLEL — {ticker}")
    print("=" * 70)

    # All 3 analysts run simultaneously via asyncio.gather()
    technical_report, fundamental_report, sentiment_report = await asyncio.gather(
        run_agent(technical_agent, prompt),
        run_agent(fundamental_agent, prompt),
        run_agent(sentiment_agent, prompt)
    )

    print("\n")
    print("=" * 70)
    print("  TECHNICAL ANALYSIS")
    print("=" * 70)
    print(technical_report)

    print("\n")
    print("=" * 70)
    print("  FUNDAMENTAL ANALYSIS")
    print("=" * 70)
    print(fundamental_report)

    print("\n")
    print("=" * 70)
    print("  SENTIMENT ANALYSIS")
    print("=" * 70)
    print(sentiment_report)

    return technical_report, fundamental_report, sentiment_report


# ============================================================
# PORTFOLIO MANAGER — SYNTHESIZES ALL 3 REPORTS
# ============================================================

async def run_portfolio_manager(ticker, technical, fundamental, sentiment):

    synthesis_prompt = f"""
Stock: {ticker}

--- TECHNICAL ANALYSIS ---
{technical}

--- FUNDAMENTAL ANALYSIS ---
{fundamental}

--- SENTIMENT ANALYSIS ---
{sentiment}

Based on all three reports, give your final investment recommendation.
"""

    print("\n")
    print("=" * 70)
    print("  PORTFOLIO MANAGER — SYNTHESISING REPORTS")
    print("=" * 70)

    recommendation = await run_agent(portfolio_manager, synthesis_prompt)

    print("\n")
    print("=" * 70)
    print("  FINAL RECOMMENDATION")
    print("=" * 70)
    print(recommendation)

    return recommendation


# ============================================================
# ORCHESTRATOR
# ============================================================

async def analyse_stock(ticker="AAPL"):

    # Step 1: Run 3 analysts in parallel
    technical, fundamental, sentiment = await run_analysts(ticker)

    # Step 2: Portfolio manager synthesises
    recommendation = await run_portfolio_manager(
        ticker, technical, fundamental, sentiment
    )

    return recommendation


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Change this to any stock ticker you want to analyse
    TICKER = "NVDA"

    asyncio.run(analyse_stock(TICKER))