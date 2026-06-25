"""
appium_test_generator.py

Multi-agent pipeline that reads an Appium XML/JSON UI dump and
auto-generates a full Java + TestNG test suite.

Pipeline:
  Stage 1 — UI Analyser Agent     (Groq LLaMA 3.3-70b)
             Parses the raw UI dump and extracts every interactive
             element: locators, element type, text, resource-id,
             content-desc, bounds. Outputs a structured element map.

  Stage 2 — 4 Parallel Test Writer Agents  (Groq, asyncio.gather)
             Each agent receives the element map and writes one
             category of Java TestNG tests:
               · Happy Path Writer      — positive end-to-end flows
               · Negative Case Writer   — invalid inputs, error states
               · Boundary Case Writer   — empty fields, max length, special chars
               · Accessibility Writer   — content-desc, touch targets, contrast hints

  Stage 3 — Page Object Model Writer  (Groq LLaMA 3.3-70b)
             Generates the Java Page Object class for the screen
             so the test classes stay clean and maintainable.

  Stage 4 — Test Plan Writer      (Groq LLaMA 3.3-70b)
             Writes a full markdown test plan report: scope, test
             cases table, traceability, risks, and run instructions.

  Stage 5 — File Saver
             Writes all .java files and the test plan to disk under
             output/<ScreenName>/.

  Stage 6 — Email Dispatch        (Groq LLaMA 3.1-8b + SendGrid)
             Emails the test plan report (HTML) with a summary of
             generated files.

============================================================
PREREQUISITES
============================================================

1. Python 3.9 or higher
   Check:     python --version
   Download:  https://www.python.org/downloads/

2. OpenAI API Key
   Used by the openai-agents SDK internally (Runner, trace).
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
     - "Restricted Access" → enable "Mail Send" only
     - Copy the key (shown only once)

5. A verified SendGrid Sender Email
   - SendGrid → Settings → Sender Authentication
   - "Verify a Single Sender" → complete the email verification
   - FROM_EMAIL below must match a verified sender

6. Your Appium UI dump file
   Export from any of:
     - uiautomatorviewer (Android)  → saves as XML
     - Appium Inspector              → File → Save Source (XML)
     - adb shell:
         adb shell uiautomator dump /sdcard/ui.xml
         adb pull /sdcard/ui.xml ./ui_dump.xml
     - Python/Java during a session:
         driver.getPageSource()  → save output to a .xml file

============================================================
INSTALLATION STEPS
============================================================

Step 1 — Create a project folder:
    mkdir appium-test-generator
    cd appium-test-generator

Step 2 — (Recommended) Create a virtual environment:
    python -m venv venv

    Activate:
      macOS / Linux:   source venv/bin/activate
      Windows:         venv\\Scripts\\activate

Step 3 — Install required packages:
    pip install openai-agents python-dotenv sendgrid

    Package breakdown:
      openai-agents   — Agent orchestration (Runner, function_tool, trace)
      python-dotenv   — Loads secrets from .env file
      sendgrid        — Sends the final test plan report by email

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

Step 5 — Place your UI dump file in the project folder:
    cp /path/to/your/ui_dump.xml ./ui_dump.xml
    (or ui_dump.json — both are supported)

Step 6 — Configure and run:
    Edit the CONFIG section at the bottom of this file:
        UI_DUMP_FILE = "ui_dump.xml"    ← your file name
        SCREEN_NAME  = "LoginScreen"    ← used for class/file names
        APP_PACKAGE  = "com.example.app" ← your app package

    Then run:
        python appium_test_generator.py

============================================================
OUTPUT STRUCTURE
============================================================

    output/
    └── LoginScreen/
        ├── LoginScreenPage.java          ← Page Object Model class
        ├── LoginScreenHappyPathTest.java ← Positive flow tests
        ├── LoginScreenNegativeTest.java  ← Invalid input / error tests
        ├── LoginScreenBoundaryTest.java  ← Edge case tests
        ├── LoginScreenAccessibilityTest.java ← Accessibility checks
        └── TestPlan_LoginScreen.md       ← Full test plan report

============================================================
GENERATED CODE CONVENTIONS
============================================================

  - All tests use TestNG annotations (@Test, @BeforeMethod, @AfterMethod)
  - Page Object Model (POM) pattern: locators in Page class, logic in tests
  - Locators use By.id (resource-id), By.xpath, or By.accessibilityId
  - AppiumDriver setup uses desired capabilities template — fill in your
    device details before running
  - Tests extend a base AppiumBaseTest class (template included in output)

============================================================
TROUBLESHOOTING
============================================================

  ModuleNotFoundError: No module named 'agents'
    → Run: pip install openai-agents

  RuntimeError: Missing required key
    → Check .env — all 5 values must be present and non-empty.

  FileNotFoundError: UI dump file not found
    → Check UI_DUMP_FILE path in the CONFIG section below.
    → Make sure the file is in the same folder as this script,
      or provide an absolute path.

  Groq RateLimitError
    → Increase asyncio.sleep() values to 4.0 seconds.
    → The 4 test writers run in parallel — if rate-limited,
      set PARALLEL_WRITERS = False to run them sequentially.

  SendGrid 403 Forbidden
    → FROM_EMAIL not verified in SendGrid Sender Authentication.
    → Fix: https://app.sendgrid.com/settings/sender_auth

  Generated Java code has syntax errors
    → The LLM occasionally makes small errors in long outputs.
    → Re-run the script — outputs are non-deterministic and
      the next run usually produces clean code.
    → Or set SCREEN_NAME to something simpler to reduce complexity.

  python: command not found
    → Try: python3 appium_test_generator.py

============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import asyncio
from pathlib import Path
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

# Main workhorse — used for all analysis and code generation
llama3_3 = OpenAIChatCompletionsModel(
    model="llama-3.3-70b-versatile",
    openai_client=groq_client
)

# Fast model — used only for email formatting
llama3_1_fast = OpenAIChatCompletionsModel(
    model="llama-3.1-8b-instant",
    openai_client=groq_client
)

# ============================================================
# CONFIG — EDIT THESE BEFORE RUNNING
# ============================================================

UI_DUMP_FILE     = "ui_dump.xml"       # Path to your Appium UI dump (XML or JSON)
SCREEN_NAME      = "LoginScreen"       # Used for Java class names and output folder
APP_PACKAGE      = "com.example.app"   # Your Android app package name
PARALLEL_WRITERS = True                # Set False if hitting Groq rate limits

OUTPUT_DIR = Path("output") / SCREEN_NAME

# ============================================================
# AGENT INSTRUCTIONS
# ============================================================

UI_ANALYSER_INSTRUCTIONS = """
You are an expert Appium test automation engineer.

You will receive a raw UI dump (XML or JSON) from an Android app screen captured
via uiautomatorviewer or Appium Inspector.

Your job is to extract and structure every interactive UI element on the screen.

For each element, extract:
  - element_name: a clean camelCase name describing the element (e.g. emailInputField)
  - element_type: EditText / Button / TextView / CheckBox / ImageView / etc.
  - resource_id: the resource-id attribute (e.g. com.example.app:id/email_input)
  - content_desc: the content-description attribute (for accessibility)
  - text: any visible text or hint text
  - bounds: the bounds attribute [x1,y1][x2,y2]
  - locator_strategy: recommend the best Appium locator:
      * By.id if resource-id is present
      * By.accessibilityId if content-desc is present but no resource-id
      * By.xpath as fallback
  - locator_value: the actual value to use with that strategy
  - is_interactive: true if the element can be tapped, typed into, or scrolled

Output a clean structured summary like this:

SCREEN: <screen name>
ELEMENTS:
  1. emailInputField
     Type: EditText
     Locator: By.id("com.example.app:id/email_input")
     ContentDesc: "Email address input"
     Text/Hint: "Enter your email"
     Interactive: true

  2. loginButton
     Type: Button
     Locator: By.id("com.example.app:id/login_btn")
     ContentDesc: "Login button"
     Text: "Login"
     Interactive: true

After the element list, add:
SCREEN SUMMARY: (1-2 sentences describing what this screen does)
POSSIBLE USER FLOWS: (bullet list of likely user actions on this screen)
"""

HAPPY_PATH_INSTRUCTIONS = """
You are a senior Appium test automation engineer writing Java TestNG tests.

You will receive a structured UI element map for an Android screen.
Write a complete Java TestNG test class for HAPPY PATH (positive) test cases only.

Rules:
  - Class name: {screen_name}HappyPathTest
  - Extend AppiumBaseTest (assume it exists and provides `driver`)
  - Use Page Object Model: reference {screen_name}Page for all locators
  - Use @Test annotation for each test method
  - Use @BeforeMethod to launch/reset the screen
  - Use @AfterMethod for teardown
  - Cover all positive flows: successful form submissions, valid navigation,
    correct data display, successful state changes
  - Add clear TestNG @Test(description="...") for each test
  - Use Assert.assertEquals, Assert.assertTrue from TestNG
  - Add meaningful comments explaining each test's purpose

Output ONLY the complete Java class code. No explanation outside the code.
"""

NEGATIVE_INSTRUCTIONS = """
You are a senior Appium test automation engineer writing Java TestNG tests.

You will receive a structured UI element map for an Android screen.
Write a complete Java TestNG test class for NEGATIVE test cases only.

Rules:
  - Class name: {screen_name}NegativeTest
  - Extend AppiumBaseTest (assume it exists and provides `driver`)
  - Use Page Object Model: reference {screen_name}Page for all locators
  - Cover all negative scenarios:
      * Invalid inputs (wrong email format, incorrect password, etc.)
      * Missing required fields (submit with empty fields)
      * Error message verification (assert the right error text appears)
      * Invalid state transitions (e.g. navigating back unexpectedly)
      * Network error simulation hints (add TODO comments where applicable)
  - Add @Test(description="...") for each test
  - Verify error messages using Assert.assertEquals on error text elements
  - Add comments explaining what negative scenario each test covers

Output ONLY the complete Java class code. No explanation outside the code.
"""

BOUNDARY_INSTRUCTIONS = """
You are a senior Appium test automation engineer writing Java TestNG tests.

You will receive a structured UI element map for an Android screen.
Write a complete Java TestNG test class for BOUNDARY and EDGE CASE tests only.

Rules:
  - Class name: {screen_name}BoundaryTest
  - Extend AppiumBaseTest (assume it exists and provides `driver`)
  - Use Page Object Model: reference {screen_name}Page for all locators
  - Cover all boundary scenarios:
      * Empty string inputs on all EditText fields
      * Maximum length inputs (255+ characters) on text fields
      * Special characters: !@#$%^&*()_+-=[]{}|;':",.<>?/\\
      * Whitespace-only inputs (spaces, tabs)
      * Unicode and emoji inputs
      * Very long single-word inputs (no spaces)
      * Numeric inputs in text fields and vice versa
      * SQL injection patterns (for security awareness)
      * XSS patterns (for security awareness)
  - Use @DataProvider for parameterised boundary tests where applicable
  - Add @Test(description="...") and comments

Output ONLY the complete Java class code. No explanation outside the code.
"""

ACCESSIBILITY_INSTRUCTIONS = """
You are a senior Appium test automation engineer writing Java TestNG tests
focused on accessibility compliance (WCAG 2.1 AA for mobile).

You will receive a structured UI element map for an Android screen.
Write a complete Java TestNG test class for ACCESSIBILITY tests only.

Rules:
  - Class name: {screen_name}AccessibilityTest
  - Extend AppiumBaseTest (assume it exists and provides `driver`)
  - Use Page Object Model: reference {screen_name}Page for all locators
  - Cover all accessibility checks:
      * Content description present on all interactive elements
        (assert element.getAttribute("content-desc") is not null/empty)
      * Touch target size >= 48x48dp (check bounds from element attributes)
      * No element relies on colour alone to convey information
        (add TODO verification comment)
      * All images have non-empty content descriptions
      * Form fields have associated labels (hint text or content-desc)
      * Focus order is logical (use AccessibilityNodeInfo checks)
      * Elements are reachable via TalkBack (add comment on manual verification)
  - Add @Test(description="...") with WCAG criterion reference where applicable
    e.g. @Test(description="WCAG 1.1.1 - Non-text content has text alternative")

Output ONLY the complete Java class code. No explanation outside the code.
"""

PAGE_OBJECT_INSTRUCTIONS = """
You are a senior Appium test automation engineer.

You will receive a structured UI element map for an Android screen.
Write a complete Java Page Object Model (POM) class for this screen.

Rules:
  - Class name: {screen_name}Page
  - Declare all locators as private static final By fields at the top
  - Use By.id for resource-id locators
  - Use By.xpath for fallback locators
  - Use AppiumBy.accessibilityId for content-desc locators
  - Include a constructor: public {screen_name}Page(AppiumDriver driver)
  - Write one public method per user action, e.g.:
      * public void enterEmail(String email)
      * public void tapLoginButton()
      * public String getErrorMessage()
      * public boolean isLoginButtonEnabled()
  - Add Javadoc comments on each method
  - Import: io.appium.java_client.AppiumDriver, AppiumBy,
            org.openqa.selenium.By, org.openqa.selenium.support.ui.WebDriverWait

Also output a separate AppiumBaseTest.java class that provides:
  - @BeforeSuite: AppiumDriver setup with desired capabilities template
  - @AfterSuite: driver.quit()
  - protected AppiumDriver driver field
  - TODO comments for deviceName, platformVersion, appPath

Output ONLY the complete Java code for both classes. No explanation outside the code.
Separate the two classes with a clear comment: // ===== AppiumBaseTest.java =====
"""

TEST_PLAN_INSTRUCTIONS = """
You are a QA lead writing a formal test plan document.

You will receive:
  - A UI element map for an Android screen
  - Four categories of generated test cases (happy path, negative, boundary, accessibility)

Write a comprehensive markdown test plan report. Structure:

  # Test Plan: {screen_name}
  **Version:** 1.0  |  **Date:** {date}  |  **Tool:** Appium + TestNG + Java

  ## 1. Scope
  (What screen is being tested, what is in/out of scope)

  ## 2. Test Environment
  (Appium version, Java version, TestNG version, Android API level requirements,
   how to set up desired capabilities)

  ## 3. Test Cases Summary Table
  | Test ID | Category | Test Name | Description | Expected Result | Priority |
  (Fill with all generated test cases — at least 20 rows)

  ## 4. Traceability Matrix
  | UI Element | Happy Path | Negative | Boundary | Accessibility |
  (Map each UI element to the tests that cover it)

  ## 5. Test Data
  (List all test data needed: valid credentials, invalid inputs,
   boundary values, special character sets)

  ## 6. Risks & Mitigations
  (List automation risks specific to mobile: flaky locators, device fragmentation,
   OS version differences, timing issues)

  ## 7. How to Run
  (Step-by-step: clone, configure capabilities, run via Maven/Gradle,
   view TestNG HTML report)

  ## 8. Generated Files
  (List all .java files generated with a one-line description each)
"""

# ============================================================
# AGENT CREATION
# ============================================================

def make_agent(name, instructions):
    return Agent(name=name, instructions=instructions, model=llama3_3)

ui_analyser_agent  = make_agent("UI Analyser",         UI_ANALYSER_INSTRUCTIONS)
happy_path_agent   = make_agent("Happy Path Writer",    HAPPY_PATH_INSTRUCTIONS)
negative_agent     = make_agent("Negative Case Writer", NEGATIVE_INSTRUCTIONS)
boundary_agent     = make_agent("Boundary Case Writer", BOUNDARY_INSTRUCTIONS)
accessibility_agent= make_agent("Accessibility Writer", ACCESSIBILITY_INSTRUCTIONS)
page_object_agent  = make_agent("Page Object Writer",   PAGE_OBJECT_INSTRUCTIONS)
test_plan_agent    = make_agent("Test Plan Writer",     TEST_PLAN_INSTRUCTIONS)

# ============================================================
# EMAIL AGENT
# ============================================================

@function_tool
async def send_email_tool(subject: str, html_body: str) -> str:
    """Send the test plan report as an HTML email via SendGrid."""
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
        "You are able to send a nicely formatted HTML email containing a QA test plan report. "
        "Convert the markdown test plan into clean, professional HTML — use a table for test cases. "
        "Subject line should be: 'Appium Test Suite Generated: <screen name>'. "
        "Send using your tool."
    ),
    tools=[send_email_tool],
    model=llama3_1_fast,
)

# ============================================================
# HELPERS
# ============================================================

async def run_agent(agent: Agent, prompt: str) -> str:
    print(f"  [→] {agent.name}")
    result = await Runner.run(agent, prompt)
    print(f"  [✓] {agent.name} done")
    return result.final_output


def save_file(folder: Path, filename: str, content: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    path.write_text(content, encoding="utf-8")
    print(f"  [💾] Saved: {path}")
    return path


def load_ui_dump(filepath: str) -> str:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"UI dump file not found: {filepath}\n"
            f"Export it from uiautomatorviewer or Appium Inspector and place it "
            f"in the same folder as this script."
        )
    return path.read_text(encoding="utf-8")


def inject_screen_name(instructions: str, screen_name: str) -> str:
    """Replace {screen_name} placeholder in agent instructions."""
    return instructions.replace("{screen_name}", screen_name)


# ============================================================
# PIPELINE STAGES
# ============================================================

async def stage_analyse_ui(ui_dump: str) -> str:
    """Stage 1: Parse UI dump and extract structured element map."""
    print("\n[Stage 1] Analysing UI dump...")
    element_map = await run_agent(
        ui_analyser_agent,
        f"Screen name: {SCREEN_NAME}\n\nUI Dump:\n{ui_dump}"
    )
    print("[✓] Element map extracted.")
    return element_map


async def stage_generate_tests(element_map: str) -> tuple[str, str, str, str]:
    """Stage 2: Run all 4 test writers in parallel."""

    print("\n[Stage 2] Generating test classes in parallel...")

    base_prompt = (
        f"Screen name: {SCREEN_NAME}\n"
        f"App package: {APP_PACKAGE}\n\n"
        f"UI Element Map:\n{element_map}"
    )

    # Inject screen name into each agent's instructions
    hp_agent = Agent(
        name="Happy Path Writer",
        instructions=inject_screen_name(HAPPY_PATH_INSTRUCTIONS, SCREEN_NAME),
        model=llama3_3
    )
    neg_agent = Agent(
        name="Negative Case Writer",
        instructions=inject_screen_name(NEGATIVE_INSTRUCTIONS, SCREEN_NAME),
        model=llama3_3
    )
    bnd_agent = Agent(
        name="Boundary Case Writer",
        instructions=inject_screen_name(BOUNDARY_INSTRUCTIONS, SCREEN_NAME),
        model=llama3_3
    )
    acc_agent = Agent(
        name="Accessibility Writer",
        instructions=inject_screen_name(ACCESSIBILITY_INSTRUCTIONS, SCREEN_NAME),
        model=llama3_3
    )

    if PARALLEL_WRITERS:
        happy, negative, boundary, accessibility = await asyncio.gather(
            run_agent(hp_agent,  base_prompt),
            run_agent(neg_agent, base_prompt),
            run_agent(bnd_agent, base_prompt),
            run_agent(acc_agent, base_prompt),
        )
    else:
        # Sequential mode — use if hitting Groq rate limits
        happy        = await run_agent(hp_agent,  base_prompt); await asyncio.sleep(2)
        negative     = await run_agent(neg_agent, base_prompt); await asyncio.sleep(2)
        boundary     = await run_agent(bnd_agent, base_prompt); await asyncio.sleep(2)
        accessibility= await run_agent(acc_agent, base_prompt)

    print("[✓] All test classes generated.")
    return happy, negative, boundary, accessibility


async def stage_generate_page_object(element_map: str) -> str:
    """Stage 3: Generate the Page Object Model class."""
    print("\n[Stage 3] Generating Page Object Model...")
    pom_agent = Agent(
        name="Page Object Writer",
        instructions=inject_screen_name(PAGE_OBJECT_INSTRUCTIONS, SCREEN_NAME),
        model=llama3_3
    )
    pom = await run_agent(
        pom_agent,
        f"Screen name: {SCREEN_NAME}\nApp package: {APP_PACKAGE}\n\nUI Element Map:\n{element_map}"
    )
    print("[✓] Page Object class generated.")
    return pom


async def stage_write_test_plan(element_map: str, happy: str, negative: str,
                                 boundary: str, accessibility: str) -> str:
    """Stage 4: Write the full test plan markdown report."""
    from datetime import date
    print("\n[Stage 4] Writing test plan report...")
    plan_agent = Agent(
        name="Test Plan Writer",
        instructions=inject_screen_name(TEST_PLAN_INSTRUCTIONS, SCREEN_NAME)
                     .replace("{date}", str(date.today())),
        model=llama3_3
    )
    prompt = (
        f"Screen: {SCREEN_NAME}\n\n"
        f"--- UI ELEMENT MAP ---\n{element_map}\n\n"
        f"--- HAPPY PATH TESTS ---\n{happy}\n\n"
        f"--- NEGATIVE TESTS ---\n{negative}\n\n"
        f"--- BOUNDARY TESTS ---\n{boundary}\n\n"
        f"--- ACCESSIBILITY TESTS ---\n{accessibility}"
    )
    plan = await run_agent(plan_agent, prompt)
    print("[✓] Test plan written.")
    return plan


def stage_save_files(happy: str, negative: str, boundary: str,
                     accessibility: str, pom: str, test_plan: str) -> list[Path]:
    """Stage 5: Save all generated files to disk."""
    print(f"\n[Stage 5] Saving files to {OUTPUT_DIR}/...")

    saved = []

    # Split POM output into two files if both classes are present
    if "AppiumBaseTest.java" in pom:
        parts = pom.split("// ===== AppiumBaseTest.java =====")
        saved.append(save_file(OUTPUT_DIR, f"{SCREEN_NAME}Page.java",    parts[0].strip()))
        saved.append(save_file(OUTPUT_DIR, "AppiumBaseTest.java",         parts[1].strip() if len(parts) > 1 else ""))
    else:
        saved.append(save_file(OUTPUT_DIR, f"{SCREEN_NAME}Page.java", pom))

    saved.append(save_file(OUTPUT_DIR, f"{SCREEN_NAME}HappyPathTest.java",    happy))
    saved.append(save_file(OUTPUT_DIR, f"{SCREEN_NAME}NegativeTest.java",     negative))
    saved.append(save_file(OUTPUT_DIR, f"{SCREEN_NAME}BoundaryTest.java",     boundary))
    saved.append(save_file(OUTPUT_DIR, f"{SCREEN_NAME}AccessibilityTest.java",accessibility))
    saved.append(save_file(OUTPUT_DIR, f"TestPlan_{SCREEN_NAME}.md",          test_plan))

    print(f"[✓] {len(saved)} files saved to {OUTPUT_DIR}/")
    return saved


async def stage_send_email(test_plan: str, saved_files: list[Path]) -> None:
    """Stage 6: Email the test plan report with file summary."""
    print(f"\n[Stage 6] Sending report to {TO_EMAIL}...")
    file_list = "\n".join(f"  - {f.name}" for f in saved_files)
    full_report = (
        f"# Appium Test Suite Generated: {SCREEN_NAME}\n\n"
        f"## Generated Files\n```\n{file_list}\n```\n\n"
        f"---\n\n{test_plan}"
    )
    await Runner.run(email_agent, f"Screen: {SCREEN_NAME}\n\nReport:\n{full_report}")
    print("[✓] Email dispatched.")

# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

async def main():

    with trace("Appium Test Generator Pipeline"):

        print(f"\n{'='*65}")
        print(f"  APPIUM TEST GENERATOR")
        print(f"  Screen:  {SCREEN_NAME}")
        print(f"  Package: {APP_PACKAGE}")
        print(f"  Input:   {UI_DUMP_FILE}")
        print(f"{'='*65}\n")

        # Load UI dump from disk
        ui_dump = load_ui_dump(UI_DUMP_FILE)
        print(f"[✓] Loaded UI dump ({len(ui_dump)} chars)")

        # Stage 1: Analyse UI
        element_map = await stage_analyse_ui(ui_dump)
        await asyncio.sleep(2.0)

        # Stage 2: 4 test writers in parallel
        happy, negative, boundary, accessibility = await stage_generate_tests(element_map)
        await asyncio.sleep(2.0)

        # Stage 3: Page Object Model
        pom = await stage_generate_page_object(element_map)
        await asyncio.sleep(2.0)

        # Stage 4: Test plan report
        test_plan = await stage_write_test_plan(
            element_map, happy, negative, boundary, accessibility
        )
        await asyncio.sleep(2.0)

        # Stage 5: Save all files
        saved_files = stage_save_files(
            happy, negative, boundary, accessibility, pom, test_plan
        )

        # Stage 6: Email report
        await stage_send_email(test_plan, saved_files)

        print(f"\n{'='*65}")
        print(f"  Pipeline complete!")
        print(f"  Files saved to: {OUTPUT_DIR}/")
        print(f"  Report emailed to: {TO_EMAIL}")
        print(f"{'='*65}\n")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())
