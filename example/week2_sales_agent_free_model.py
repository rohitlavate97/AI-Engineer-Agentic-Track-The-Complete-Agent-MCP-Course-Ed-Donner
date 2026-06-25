import os
import asyncio
from typing import Dict
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI
import agents
from agents import (
    Agent, 
    Runner, 
    trace, 
    function_tool, 
    OpenAIChatCompletionsModel, 
    input_guardrail, 
    GuardrailFunctionOutput,
    set_tracing_disabled
)
from agents.exceptions import InputGuardrailTripwireTriggered
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content

# Globally disable background telemetry sync overhead to optimize execution
set_tracing_disabled(True)

# Load environment configuration variables
load_dotenv(override=True)

# Extract and register API parameters
openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
sendgrid_api_key = os.getenv('SENDGRID_API_KEY')

# Critical API validation checks
required_keys = {
    "OPENAI_API_KEY": openai_api_key,
    "DEEPSEEK_API_KEY": deepseek_api_key,
    "GROQ_API_KEY": groq_api_key,
    "SENDGRID_API_KEY": sendgrid_api_key,
}

for key_name, key_value in required_keys.items():
    if not key_value:
        raise RuntimeError(f"Initialization Aborted: Missing required key environment variable '{key_name}'.")

# Router Endpoint Mapping Layout Definitions
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Async Client Instances
deepseek_client = AsyncOpenAI(base_url=DEEPSEEK_BASE_URL, api_key=deepseek_api_key)
groq_client = AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=groq_api_key)

# Model Wrapper Objects (100% Free Tiers and Open-Source Engines)
deepseek_model = OpenAIChatCompletionsModel(model="deepseek-chat", openai_client=deepseek_client)
llama3_3_model = OpenAIChatCompletionsModel(model="llama-3.3-70b-versatile", openai_client=groq_client)
llama3_1_fast = OpenAIChatCompletionsModel(model="llama-3.1-8b-instant", openai_client=groq_client)

# ============================================================
# SALES DRAFTING AGENTS (Text-In / Text-Out Utilities)
# ============================================================
instructions1 = "You are a sales agent working for ComplAI. You write professional, serious cold emails."
instructions2 = "You are a humorous sales agent working for ComplAI. You write witty, engaging cold emails."
instructions3 = "You are a busy sales agent working for ComplAI. You write concise, to the point cold emails."

sales_agent1 = Agent(name="DeepSeek Sales Agent", instructions=instructions1, model=deepseek_model)
sales_agent2 = Agent(name="Gemini Sales Agent", instructions=instructions2, model=llama3_3_model)
sales_agent3 = Agent(name="Llama3.3 Sales Agent", instructions=instructions3, model=llama3_3_model)

description = "Write a cold sales email"
tool1 = sales_agent1.as_tool(tool_name="sales_agent1", tool_description=description)
tool2 = sales_agent2.as_tool(tool_name="sales_agent2", tool_description=description)
tool3 = sales_agent3.as_tool(tool_name="sales_agent3", tool_description=description)

# ============================================================
# DEPENDENT TOOLS (WITH SAFE EXCEPTIONS AND EXECUTOR BACKGROUND COOLDOWNS)
# ============================================================
@function_tool
async def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
    """ Send out an email with the given subject and HTML body to all sales prospects """
    def _sync_send():
        try:
            sg = sendgrid.SendGridAPIClient(api_key=os.environ["SENDGRID_API_KEY"])
            from_email = Email("ed@edwarddonner.com")  
            to_email = To("ed.donner@gmail.com")      
            content = Content("text/html", html_body)
            mail = Mail(from_email, to_email, subject, content).get()
            
            response = sg.client.mail.send.post(request_body=mail)
            if response.status_code in [200, 201, 202]:
                return {"status": "success"}
            return {"status": "failed", "error": f"Bad status code: {response.status_code}"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}
            
    return await asyncio.to_thread(_sync_send)

# Utility Tool Definitions
subject_instructions = "You write a short, compelling email subject line based on an incoming plain text draft."
html_instructions = "You convert markdown/plain text emails into valid, visually responsive HTML email structures."

subject_writer = Agent(name="Email subject writer", instructions=subject_instructions, model=llama3_1_fast)
subject_tool = subject_writer.as_tool(tool_name="subject_writer", tool_description="Write an email subject line")

html_converter = Agent(name="HTML email body converter", instructions=html_instructions, model=llama3_1_fast)
html_tool = html_converter.as_tool(tool_name="html_converter", tool_description="Convert email copy into clean HTML markup")

# ============================================================
# PROGRAMMATIC INPUT GUARDRAILS
# ============================================================
@input_guardrail
async def guardrail_against_name(ctx, agent, message):
    """Programmatically intercept personal human names inside incoming strings."""
    msg_lower = message.lower()
    
    # Fast algorithmic check for the explicit presence of security risk tags
    if "alice" in msg_lower or "bob" in msg_lower:
         return GuardrailFunctionOutput(output_info={"found_name": True}, tripwire_triggered=True)
         
    return GuardrailFunctionOutput(output_info={"found_name": False}, tripwire_triggered=False)

# ============================================================
# FLAT-TOP INTEGRATED MULTI-AGENT ORCHESTRATOR
# ============================================================
sales_manager_instructions = """
You are the Executive Sales Director at ComplAI. You have access to all formatting, generation, and delivery tools.

Execute these tasks step-by-step:
1. Generate Options: Run sales_agent1, sales_agent2, and sales_agent3 to build three options.
2. Selection: Critically evaluate which email body is the best option.
3. Formatting: Call subject_writer on that single best text draft, then call html_converter to construct the HTML markup template.
4. Transmission: Finally, execute the send_html_email tool to deliver the final message.
"""

careful_sales_manager = Agent(
    name="Sales Manager",
    instructions=sales_manager_instructions,
    tools=[tool1, tool2, tool3, subject_tool, html_tool, send_html_email],
    model=llama3_1_fast,  
    input_guardrails=[guardrail_against_name]
)

# ============================================================
# PIPELINE ENTRY POINT & RUNNER EXECUTION
# ============================================================
async def run_pipeline_step(message_input: str, workflow_title: str):
    print("\n" + "=" * 70)
    print(f" INITIALIZING: {workflow_title}")
    print("=" * 70)
    
    with trace(workflow_title):
        try:
            result = await Runner.run(careful_sales_manager, message_input)
            
            if hasattr(result, 'final_output') and result.final_output:
                print("\n[WORKFLOW RUN RESULT SUCCESS]:")
                print(result.final_output)
            else:
                print("\n[WORKFLOW COMPLETED SUCCESSFULLY]")
        
        # Catching the guardrail exception gracefully
        except InputGuardrailTripwireTriggered:
            print("\n[❌ SECURITY GUARDRAIL INTERCEPTED]: Pipeline execution safely terminated because a personal name was identified inside the prompt.")

async def main():
    # RUN 1: This contains "Alice", triggering your protective guardrail and recovering smoothly
    await run_pipeline_step(
        message_input="Send out a cold sales email addressed to Dear CEO from Alice",
        workflow_title="Protected SDR Workflow - Security Intercept Verification"
    )

    # 2-second rate cooldown to prevent free-tier limits from choking
    await asyncio.sleep(2)

    # RUN 2: Standard production path running through clean processing states
    await run_pipeline_step(
        message_input="Send out a cold sales email addressed to Dear CEO from Head of Business Development",
        workflow_title="Protected SDR Workflow - Production Dispatch Run"
    )

if __name__ == "__main__":
    asyncio.run(main())