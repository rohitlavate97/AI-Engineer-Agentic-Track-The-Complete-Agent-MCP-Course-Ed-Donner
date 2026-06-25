"""
indian_market_multibagger.py

Multi-agent Indian Stock Market analyzer using Groq.
Screens NSE/BSE equities for potential MULTIBAGGER stocks
(companies capable of 2x, 5x, or 10x returns over 3-5 years).

Architecture:
  - 4 parallel analyst agents running simultaneously via asyncio.gather()
  - 1 CIO (Chief Investment Officer) agent that synthesizes all 4 reports
  - Final output structured as a PMS (Portfolio Management Service) disclosure

Why 4 agents instead of 3 (unlike crypto/stock versions)?
  Indian equities require one extra lens:
    - Governance Analyst    → Promoter pledging, FII/DII flows, auditor integrity
    - Financials Analyst    → ROE, ROCE, FCF quality, debt health
    - Macro Runway Analyst  → TAM, sector tailwinds, Make in India, EV, etc.
    - Technical & Valuation → EMA, PEG ratio, consolidation breakouts

============================================================
PREREQUISITES
============================================================

1. Python 3.8 or higher
   Check your version:
       python --version
   Download from: https://www.python.org/downloads/

2. A Groq API key (free tier available)
   Sign up at: https://console.groq.com
   Steps:
     - Register / log in
     - Go to "API Keys" in the left sidebar
     - Click "Create API Key"
     - Copy and store your key safely

3. pip (Python package manager — ships with Python by default)
   Verify:
       pip --version

============================================================
INSTALLATION STEPS
============================================================

Step 1 — Create a project folder and navigate into it:
    mkdir indian-multibagger
    cd indian-multibagger

Step 2 — (Recommended) Create a virtual environment:
    python -m venv venv

    Activate it:
      macOS / Linux:   source venv/bin/activate
      Windows:         venv\\Scripts\\activate

    You should see (venv) prefix in your terminal.

Step 3 — Install required packages:
    pip install groq python-dotenv

    Package breakdown:
      groq            — Official Groq Python SDK. Handles API calls to
                        LLaMA 3.3-70b running on Groq's fast inference.
      python-dotenv   — Loads your GROQ_API_KEY from a .env file so you
                        never hardcode secrets in your source code.

Step 4 — Create a .env file in the same folder as this script:
    macOS / Linux:   touch .env
    Windows:         type nul > .env

    Open .env in any text editor and add exactly this line:
        GROQ_API_KEY=your_actual_key_here

    Example:
        GROQ_API_KEY=gsk_Abc123XYZ...

    IMPORTANT — Keep your key private:
        echo ".env" >> .gitignore

Step 5 — Place this file in the same folder, then run:
    python indian_market_multibagger.py

    To screen a different stock, scroll to the bottom and change:
        STOCK = "Tata Power Ltd."
    to any NSE/BSE listed company, e.g.:
        STOCK = "Dixon Technologies Ltd."
        STOCK = "Kaynes Technology India Ltd."
        STOCK = "Suzlon Energy Ltd."

============================================================
FOLDER STRUCTURE AFTER SETUP
============================================================

    indian-multibagger/
    ├── indian_market_multibagger.py   ← this file
    ├── .env                           ← your API key (never share/commit)
    ├── .gitignore                     ← should contain: .env
    └── venv/                          ← virtual environment (optional)

============================================================
5-POINT MULTIBAGGER SCREENING CHECKLIST
============================================================

When reviewing the final output, look for ALL of the following:

  Metric                Multibagger Criteria       Why It Matters
  ─────────────────     ────────────────────────   ──────────────────────────────────
  Market Cap            Under ₹15,000 Crore        Easier to 10x from ₹2,000 Cr than
                                                   for a mega-cap to double
  Operating Leverage    Sales growth < Profit      Fixed costs stay flat while profits
                        growth                     explode with revenue scale
  Promoter Pledging     Strictly 0% (or falling)   High pledging = margin call risk,
                                                   overnight crashes
  ROCE Track Record     Consistently > 20%         Proves self-funding growth without
                                                   equity dilution
  Earnings Conversion   CFO / EBITDA > 70%         Reported profits = real cash in bank

============================================================
TROUBLESHOOTING
============================================================

  ModuleNotFoundError: No module named 'groq'
    → Run: pip install groq python-dotenv
    → Make sure your venv is activated first.

  AuthenticationError / 401 Unauthorized
    → Your GROQ_API_KEY in .env is wrong or missing.
    → Check for extra spaces: KEY=value  (no spaces around =)

  RateLimitError
    → Groq free tier has per-minute token limits.
    → Wait 10-20 seconds and run again.
    → Or reduce parallel agents temporarily.

  python: command not found
    → Try: python3 indian_market_multibagger.py

  Results feel generic / not stock-specific
    → The LLM uses training knowledge, not live market data.
    → For real-time accuracy, inject live data before the prompt:
        * Screener.in API or web scrape for financials
        * NSEpy / jugaad-trader for price/volume data
        * Trendlyne or Tickertape for shareholding data

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

# --- Agent 1: Corporate Governance & Shareholding Analyst ---
# Replaces the On-Chain Analyst from crypto version.
# In India, exchange-disclosed ownership data is the equivalent of
# on-chain transparency — it reveals insider confidence and traps.

governance_instructions = """
You are an expert Corporate Governance & Shareholding Analyst
specializing in Indian equities listed on NSE and BSE.

Analyze the given company focusing on ownership stability and
corporate trust. Cover:

  - Promoter holding trends (stable, increasing, or alarming dilution)
  - Promoters' Pledged Shares % — flag anything above 10% as HIGH RISK
  - FII (Foreign Institutional Investors) holding patterns
  - DII (Domestic: Mutual Funds, LIC, etc.) accumulation or exit trends
  - Corporate governance red flags:
      * Related-party transactions benefiting promoters at company's cost
      * Sudden auditor resignations or qualifications in audit reports
      * Frequent CFO or MD changes without clear explanation
      * Regulatory actions by SEBI, stock exchanges, or MCA

Provide a clear BUY / HOLD / AVOID signal based purely on
corporate integrity and insider alignment with minority shareholders.
"""

# --- Agent 2: Financial Quality & Efficiency Analyst ---
# Replaces the Tokenomics Analyst from crypto version.
# Indian multibaggers are built on compounding internal capital,
# NOT token inflation or deflation mechanics.

financials_instructions = """
You are a Financial Quality & Efficiency Analyst evaluating
Indian multibagger candidates listed on NSE/BSE.

Analyze the business fundamentals strictly over a 3-to-5-year
trailing horizon. Cover:

  - Return on Capital Employed (ROCE) — is it sustainably > 15-20%?
  - Return on Equity (ROE) — same benchmark, watch for leverage distortion
  - Operating Profit Margin (OPM) and Net Profit Margin trends
    (are they expanding or contracting over years?)
  - Balance Sheet health:
      * Debt-to-Equity ratio (ideally < 0.5 for non-NBFC/infra)
      * Interest Coverage Ratio (> 3x is comfortable)
  - Quality of Earnings: Does Net Profit match Operating Cash Flow (CFO)?
    A CFO/EBITDA ratio > 70% confirms real cash generation.
    Low ratio = profits trapped in receivables or inflated by accounting.
  - Working Capital cycle (is it improving or deteriorating?)
  - Capital Allocation history: acquisitions, capex, dividends

Provide a clear BUY / HOLD / AVOID signal based purely on the
company's ability to compound capital internally.
"""

# --- Agent 3: Industry Runway & Macro Tailwinds Analyst ---
# Replaces the Narrative Analyst from crypto version.
# In India, a massive under-penetrated domestic market is the fuel
# that converts a good business into a 10x compounder.

macro_runway_instructions = """
You are an Industry Runway & Macro Tailwinds Analyst looking for
high-growth sector opportunities in the Indian economy.

Evaluate the long-term structural expansion potential. Cover:

  - Total Addressable Market (TAM): Is the industry under-penetrated?
  - Structural tailwinds — does this company ride one of India's
    mega-trends?
      * Make in India / China+1 supply chain shift
      * Digitalization of financial services or retail
      * EV ecosystem and renewable energy transition
      * Premiumization of consumer goods
      * Defence indigenization (Atmanirbhar Bharat)
      * Healthcare infrastructure expansion
  - Market Cap classification:
      * Micro-cap (< ₹5,000 Cr): highest upside, highest risk
      * Small-cap (₹5,000–₹15,000 Cr): sweet spot for multibaggers
      * Mid-cap (₹15,000–₹50,000 Cr): slower but safer compounding
      * Large-cap (> ₹50,000 Cr): unlikely to deliver 5-10x
  - Competitive Moat:
      * Brand loyalty, switching costs, cost leadership,
        regulatory licenses, distribution network advantages

Provide a clear BUY / HOLD / AVOID signal based on market
runway and structural tailwinds available to this company.
"""

# --- Agent 4: Technical Structure & Valuation Analyst ---
# Adapts the Technical Analyst from the stock version but adds
# Indian-specific valuation metrics critical for entry timing.

technical_valuation_instructions = """
You are a Technical Structure & Valuation Analyst focused on
entry timing and price reasonability for Indian equities.

Analyze both the chart structure and fundamental valuation. Cover:

  - Price trend: EMA 50 and EMA 200 — is the stock in a structural
    uptrend (price > EMA50 > EMA200)?
  - Volume profile: Is a breakout accompanied by above-average volume?
    Low-volume breakouts in Indian small-caps are frequent fakeouts.
  - Base pattern: Is the stock breaking out of a multi-month or
    multi-year consolidation base? (Most reliable setup for multibaggers)
  - P/E ratio vs historic band and sector peers
    (Screener.in data is the standard reference in India)
  - P/B (Price-to-Book) for asset-heavy sectors like banking, infra
  - PEG Ratio (P/E ÷ Expected EPS Growth Rate):
      * PEG < 1 = potentially undervalued growth
      * PEG 1-2 = fairly valued
      * PEG > 2 = expensive relative to growth
  - Avoid stocks at extended all-time highs without a consolidation phase

Provide a clear BUY / HOLD / AVOID signal based on the
risk-reward ratio of entering at current market prices.
"""

# --- Agent 5: Chief Investment Officer (CIO) ---
# Synthesizes all 4 reports and delivers a PMS-style
# institutional investment disclosure on multibagger potential.

cio_instructions = """
You are the Chief Investment Officer (CIO) of a SEBI-registered
Portfolio Management Service (PMS) in India.

You will receive 4 institutional analyst reports on the same stock:
  1. Corporate Governance & Shareholding
  2. Financial Quality & Efficiency
  3. Industry Runway & Macro Tailwinds
  4. Technical Structure & Valuation

Your job:
  - Weigh all 4 reports, knowing that for Indian multibaggers:
      * GOVERNANCE issues are disqualifying — no amount of growth
        compensates for promoter fraud or pledging risk
      * FINANCIALS must show compounding ROCE/ROE over 3-5 years
      * MACRO RUNWAY is the growth engine — size of opportunity matters
      * VALUATION determines entry safety, not whether to invest
  - Identify where analysts agree or conflict
  - Call out the single most critical metric to track going forward
  - Consider SEBI regulations, circuit breaker risks for small-caps,
    and liquidity risk (low float stocks in India can be illiquid)

Format your output EXACTLY as:

  MULTIBAGGER VERDICT:    YES / NO / POTENTIAL HOLD
  CONVICTION LEVEL:       High / Medium / Low
  TARGET HORIZON:         3 - 5 Years

  INVESTMENT THESIS (BULL CASE):
  (2-3 sentences on the core structural driver for re-rating)

  RISK FACTORS (BEAR CASE):
  (2-3 sentences on what could permanently derail the business)

  CRITICAL TRACKING METRICS:
  - (Metric 1 to watch every quarter — e.g., ROCE trend)
  - (Metric 2 to watch every quarter — e.g., Promoter pledge %)
  - (Metric 3 to watch every quarter — e.g., Order book growth)

  FINAL SUMMARY:
  (1 paragraph: your institutional stance on this stock)
"""

# ============================================================
# AGENT CREATION
# ============================================================

MODEL = "llama-3.3-70b-versatile"

governance_agent = Agent(
    name="Governance Analyst",
    instructions=governance_instructions,
    model=MODEL
)

financials_agent = Agent(
    name="Financials Analyst",
    instructions=financials_instructions,
    model=MODEL
)

macro_agent = Agent(
    name="Macro Runway Analyst",
    instructions=macro_runway_instructions,
    model=MODEL
)

technical_agent = Agent(
    name="Technical & Valuation Analyst",
    instructions=technical_valuation_instructions,
    model=MODEL
)

cio_agent = Agent(
    name="Chief Investment Officer",
    instructions=cio_instructions,
    model=MODEL
)

# ============================================================
# RUN SINGLE AGENT
# ============================================================

async def run_agent(agent, prompt):

    print(f"  [→] Running: {agent.name}")

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
    print(f"  [✓] Done:    {agent.name}")

    return result


# ============================================================
# PARALLEL ANALYST EXECUTION (all 4 run simultaneously)
# ============================================================

async def run_analysts(stock):

    prompt = f"Analyse the Indian equity: {stock} for long-term multi-fold returns."

    print("\n")
    print("=" * 70)
    print(f"  PARALLEL EQUITY EVALUATION — {stock}")
    print("=" * 70)

    gov, fin, macro, tech = await asyncio.gather(
        run_agent(governance_agent, prompt),
        run_agent(financials_agent, prompt),
        run_agent(macro_agent, prompt),
        run_agent(technical_agent, prompt)
    )

    print("\n")
    print("=" * 70)
    print("  GOVERNANCE & SHAREHOLDING REPORT")
    print("=" * 70)
    print(gov)

    print("\n")
    print("=" * 70)
    print("  FINANCIAL QUALITY & EFFICIENCY REPORT")
    print("=" * 70)
    print(fin)

    print("\n")
    print("=" * 70)
    print("  INDUSTRY RUNWAY & MACRO TAILWINDS REPORT")
    print("=" * 70)
    print(macro)

    print("\n")
    print("=" * 70)
    print("  TECHNICAL STRUCTURE & VALUATION REPORT")
    print("=" * 70)
    print(tech)

    return gov, fin, macro, tech


# ============================================================
# CIO SYNTHESIS — Combines all 4 reports into final verdict
# ============================================================

async def run_cio(stock, gov, fin, macro, tech):

    synthesis_prompt = f"""
Stock: {stock}

--- 1. GOVERNANCE & SHAREHOLDING ANALYSIS ---
{gov}

--- 2. FINANCIAL QUALITY & EFFICIENCY ANALYSIS ---
{fin}

--- 3. INDUSTRY RUNWAY & MACRO TAILWINDS ANALYSIS ---
{macro}

--- 4. TECHNICAL STRUCTURE & VALUATION ANALYSIS ---
{tech}

Based on all four reports, provide your final multibagger assessment.
"""

    print("\n")
    print("=" * 70)
    print("  CIO OFFICE — SYNTHESISING MULTIBAGGER POTENTIAL")
    print("=" * 70)

    verdict = await run_agent(cio_agent, synthesis_prompt)

    print("\n")
    print("=" * 70)
    print("  FINAL PMS DISCLOSURE")
    print("=" * 70)
    print(verdict)

    return verdict


# ============================================================
# ORCHESTRATOR
# ============================================================

async def analyze_indian_stock(stock):

    # Step 1: Run all 4 analysts in parallel
    gov, fin, macro, tech = await run_analysts(stock)

    # Step 2: CIO synthesises all 4 reports into a final verdict
    verdict = await run_cio(stock, gov, fin, macro, tech)

    return verdict


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Change this to any NSE/BSE listed company you want to screen.
    # Good candidates to try:
    #   "Dixon Technologies Ltd."        — Electronics manufacturing
    #   "Kaynes Technology India Ltd."   — EMS / electronics
    #   "Suzlon Energy Ltd."             — Wind energy turnaround
    #   "Waaree Energies Ltd."           — Solar manufacturing
    #   "Tata Power Ltd."                — Renewable energy pivot
    #   "Mankind Pharma Ltd."            — Pharma distribution

    STOCK = "Tata Power Ltd."

    asyncio.run(analyze_indian_stock(STOCK))
