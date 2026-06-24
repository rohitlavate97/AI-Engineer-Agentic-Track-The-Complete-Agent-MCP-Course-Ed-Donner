"""
crypto_agents_groq.py

Multi-agent crypto market analyzer using Groq.

Architecture:
  - 4 parallel analyst agents (on-chain, tokenomics, sentiment/narrative, technical/market structure)
  - 1 crypto fund manager agent that synthesizes all 4 reports
  - asyncio.gather() for parallel execution

Why different agents vs stocks?
  Crypto has unique signals that don't exist in equities:
    - On-chain: wallet flows, exchange reserves, whale movements
    - Tokenomics: supply schedule, vesting cliffs, inflation rate
    - Narrative: community momentum, developer activity, hype cycles

============================================================
PREREQUISITES
============================================================

1. Python 3.8 or higher
   Check your version:
       python --version
   Download from: https://www.python.org/downloads/

2. A Groq API key (free tier available)
   Sign up at: https://console.groq.com
   - Go to "API Keys" in the dashboard
   - Click "Create API Key"
   - Copy and save your key

3. pip (Python package manager — comes with Python)
   Check it works:
       pip --version

============================================================
INSTALLATION STEPS
============================================================

Step 1 — Clone or download this file into a project folder:
    mkdir crypto-agents
    cd crypto-agents

Step 2 — (Recommended) Create a virtual environment:
    python -m venv venv

    Activate it:
      macOS / Linux:   source venv/bin/activate
      Windows:         venv\\Scripts\\activate

Step 3 — Install required packages:
    pip install groq python-dotenv

    What each package does:
      groq          — official Groq Python SDK (calls the LLM API)
      python-dotenv — loads your API key from a .env file safely

Step 4 — Create a .env file in the same folder as this script:
    touch .env           # macOS / Linux
    type nul > .env      # Windows

    Open it in any text editor and add this line:
        GROQ_API_KEY=your_actual_api_key_here

    Example .env contents:
        GROQ_API_KEY=gsk_abc123xyz...

    IMPORTANT: Never commit your .env file to Git.
    Add it to .gitignore:
        echo ".env" >> .gitignore

Step 5 — Run the script:
    python crypto_agents_groq.py

    To analyse a different coin, edit line at the bottom:
        COIN = "Ethereum (ETH)"

============================================================
FOLDER STRUCTURE AFTER SETUP
============================================================

    crypto-agents/
    ├── crypto_agents_groq.py   ← this file
    ├── .env                    ← your API key (never share this)
    ├── .gitignore              ← should list .env
    └── venv/                   ← virtual environment (optional)

============================================================
TROUBLESHOOTING
============================================================

  ModuleNotFoundError: No module named 'groq'
    → You forgot Step 3, or your venv is not activated.
    → Run: pip install groq python-dotenv

  AuthenticationError / 401
    → Your GROQ_API_KEY in .env is missing or incorrect.
    → Double-check there are no spaces around the = sign.

  RateLimitError
    → Groq free tier has rate limits. Wait a few seconds and retry.
    → Or upgrade your Groq plan at console.groq.com

  python: command not found
    → Try python3 instead: python3 crypto_agents_groq.py

============================================================
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
# AGENT CLASS
# ============================================================

class Agent:

    def __init__(self, name, instructions, model):
        self.name = name
        self.instructions = instructions
        self.model = model


# ============================================================
# ANALYST INSTRUCTIONS
# ============================================================

# --- Agent 1: On-Chain Analyst ---
# Crypto-specific: looks at wallet-level data that has no stock equivalent

onchain_instructions = """
You are a professional on-chain analyst at a crypto hedge fund.

Analyse the given cryptocurrency from an on-chain perspective only.

Cover:
  - Exchange reserve flows (is BTC/ETH being withdrawn to cold wallets = bullish?)
  - Whale wallet accumulation or distribution patterns
  - Active addresses trend (growing = healthy network)
  - Network hash rate or validator count (security & confidence)
  - HODL waves / coin age distribution (long-term holders vs short-term)
  - Any notable large transactions or wallet movements

Give a clear BUY / HOLD / SELL signal with your on-chain reasoning.
Be specific about which metrics are most important for this asset.
"""

# --- Agent 2: Tokenomics Analyst ---
# Crypto-specific: supply dynamics are critical — no equivalent in stocks

tokenomics_instructions = """
You are a professional tokenomics analyst at a crypto hedge fund.

Analyse the given cryptocurrency from a tokenomics perspective only.

Cover:
  - Circulating supply vs max supply (how much inflation is left?)
  - Upcoming token unlock / vesting cliff events (potential sell pressure)
  - Token emission schedule (how fast new tokens are minted)
  - Token utility and demand drivers (is there real demand for the token?)
  - Staking or locking mechanisms (how much supply is locked up?)
  - Treasury and team allocation (are insiders well-incentivised?)
  - Any recent or upcoming protocol changes affecting supply/demand

Give a clear BUY / HOLD / SELL signal based purely on tokenomics.
Flag any major unlock events as high-priority risks.
"""

# --- Agent 3: Narrative & Sentiment Analyst ---
# Combines market sentiment with crypto-specific narrative cycles

narrative_instructions = """
You are a professional narrative and sentiment analyst at a crypto hedge fund.

Analyse the given cryptocurrency from a narrative and sentiment perspective only.

Cover:
  - Current market narrative (is this coin part of a hot sector: AI, DeFi, L2, RWA?)
  - Community strength (Reddit, Discord, X/Twitter engagement)
  - Developer activity (GitHub commits, ecosystem growth)
  - Influencer and KOL (Key Opinion Leader) sentiment
  - Recent exchange listings or delistings
  - Regulatory news affecting this specific coin
  - Macro crypto sentiment (Bitcoin dominance, risk-on vs risk-off)
  - Hype cycle position (early adoption, peak hype, or capitulation?)

Give a clear BUY / HOLD / SELL signal based on narrative and sentiment.
Note if the narrative is driven by fundamentals or pure speculation.
"""

# --- Agent 4: Technical & Market Structure Analyst ---
# Price action, momentum indicators, and derivatives metrics

technical_instructions = """
You are a professional technical and market structure analyst at a crypto hedge fund.

Analyse the given cryptocurrency from a technical and derivatives perspective only.

Cover:
  - Price Action & Trend: Key support/resistance levels, moving averages (EMA 20/50/200), trend direction (bullish, bearish, sideways).
  - Momentum Indicators: RSI (Relative Strength Index) for overbought/oversold levels, MACD (Moving Average Convergence Divergence) for trend momentum.
  - Volume Analysis: Volume trends, VWAP, or volume profiles to validate price moves.
  - Derivatives Market Structure: Funding Rates (are traders over-leveraging long or short?) and Open Interest (is leverage entering or leaving the market?).
  - Liquidity & Volatility: Bollinger Bands, historical volatility, and order book depth if relevant.

Give a clear BUY / HOLD / SELL signal based on technicals and derivatives data.
Indicate specific price areas or conditions needed to trigger the signal.
"""

# --- Agent 5: Crypto Fund Manager ---
# Synthesizes all 4, aware of crypto-specific risk factors

fund_manager_instructions = """
You are a senior portfolio manager at a crypto-native hedge fund.

You will receive four analyst reports on the same cryptocurrency:
  1. On-chain analysis
  2. Tokenomics analysis
  3. Narrative & sentiment analysis
  4. Technical & market structure analysis

Your job:
  - Weigh each report, knowing that technical signals help with entry/exit timing, while on-chain and tokenomics guide medium-to-long term viability.
  - Identify where the analysts agree or conflict
  - Call out any major near-term catalysts (unlocks, halvings, upgrades, technical resistance/support zones)
  - Account for crypto-specific risks: smart contract exploits,
    regulatory crackdowns, liquidity risk on smaller caps, and derivatives liquidation cascades

Format your output EXACTLY as:

  FINAL VERDICT:  BUY / HOLD / SELL
  CONVICTION:     High / Medium / Low
  TIME HORIZON:   Short (days-weeks) / Medium (1-3 months) / Long (6m+)

  BULL CASE:
  (2-3 sentences on why this works)

  BEAR CASE:
  (2-3 sentences on the main downside scenario)

  KEY RISKS:
  - (risk 1)
  - (risk 2)
  - (risk 3)

  SUMMARY:
  (1 paragraph final take)
"""

# ============================================================
# AGENT CREATION
# ============================================================

onchain_agent = Agent(
    name="On-Chain Analyst",
    instructions=onchain_instructions,
    model="llama-3.3-70b-versatile"
)

tokenomics_agent = Agent(
    name="Tokenomics Analyst",
    instructions=tokenomics_instructions,
    model="llama-3.3-70b-versatile"
)

narrative_agent = Agent(
    name="Narrative & Sentiment Analyst",
    instructions=narrative_instructions,
    model="llama-3.3-70b-versatile"
)

technical_agent = Agent(
    name="Technical Analyst",
    instructions=technical_instructions,
    model="llama-3.3-70b-versatile"
)

fund_manager = Agent(
    name="Crypto Fund Manager",
    instructions=fund_manager_instructions,
    model="llama-3.3-70b-versatile"
)

# ============================================================
# RUN SINGLE AGENT
# ============================================================

async def run_agent(agent, prompt):

    print(f"\n  [->] Running: {agent.name}")

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
    print(f"  [OK] Done:    {agent.name}")

    return result


# ============================================================
# PARALLEL ANALYST EXECUTION
# ============================================================

async def run_analysts(coin):

    prompt = f"Analyse {coin} cryptocurrency."

    print("\n")
    print("=" * 70)
    print(f"  RUNNING 4 ANALYSTS IN PARALLEL - {coin}")
    print("=" * 70)

    # All 4 analysts run simultaneously
    onchain_report, tokenomics_report, narrative_report, technical_report = await asyncio.gather(
        run_agent(onchain_agent, prompt),
        run_agent(tokenomics_agent, prompt),
        run_agent(narrative_agent, prompt),
        run_agent(technical_agent, prompt)
    )

    print("\n")
    print("=" * 70)
    print("  ON-CHAIN ANALYSIS")
    print("=" * 70)
    print(onchain_report)

    print("\n")
    print("=" * 70)
    print("  TOKENOMICS ANALYSIS")
    print("=" * 70)
    print(tokenomics_report)

    print("\n")
    print("=" * 70)
    print("  NARRATIVE & SENTIMENT ANALYSIS")
    print("=" * 70)
    print(narrative_report)

    print("\n")
    print("=" * 70)
    print("  TECHNICAL & MARKET STRUCTURE ANALYSIS")
    print("=" * 70)
    print(technical_report)

    return onchain_report, tokenomics_report, narrative_report, technical_report


# ============================================================
# FUND MANAGER - SYNTHESIZES ALL 4 REPORTS
# ============================================================

async def run_fund_manager(coin, onchain, tokenomics, narrative, technical):

    synthesis_prompt = f"""
Cryptocurrency: {coin}

--- ON-CHAIN ANALYSIS ---
{onchain}

--- TOKENOMICS ANALYSIS ---
{tokenomics}

--- NARRATIVE & SENTIMENT ANALYSIS ---
{narrative}

--- TECHNICAL & MARKET STRUCTURE ANALYSIS ---
{technical}

Based on all four reports, give your final investment recommendation.
"""

    print("\n")
    print("=" * 70)
    print("  CRYPTO FUND MANAGER - SYNTHESISING REPORTS")
    print("=" * 70)

    recommendation = await run_agent(fund_manager, synthesis_prompt)

    print("\n")
    print("=" * 70)
    print("  FINAL RECOMMENDATION")
    print("=" * 70)
    print(recommendation)

    return recommendation


# ============================================================
# ORCHESTRATOR
# ============================================================

async def analyse_crypto(coin="Bitcoin (BTC)"):

    # Step 1: Run 4 analysts in parallel
    onchain, tokenomics, narrative, technical = await run_analysts(coin)

    # Step 2: Fund manager synthesises all 4 into a final call
    recommendation = await run_fund_manager(
        coin, onchain, tokenomics, narrative, technical
    )

    return recommendation


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Change to any coin: "Ethereum (ETH)", "Solana (SOL)", "Arbitrum (ARB)", etc.
    COIN = "Hyperliquid (HYPE)"

    asyncio.run(analyse_crypto(COIN))