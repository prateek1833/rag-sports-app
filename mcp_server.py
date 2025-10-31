# mcp_server.py
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import io
import sys
import traceback
load_dotenv()  # This pulls from .env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("mcp_server")

app = FastAPI(title="RAG MCP Server")

# === EMAIL CONFIGURATION ===
# Set these via environment variables for security
SENDER_EMAIL = os.getenv("SENDER_EMAIL")      # e.g., ragapp.feedback@gmail.com
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")  # App password (Gmail)
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL", "feedback@example.com")

# === TOOL: Refine Query ===
def refine_query_tool(original_query: str) -> str:
    """Refine the original query for better retrieval."""
    log.info(f"Refining query: {original_query}")
    return original_query.strip() + " (refined)"

# === TOOL: Extract Entities ===
def extract_entities_tool(query: str, text: str) -> str:
    """Extract key entities from the provided text based on the query."""
    log.info(f"Extracting entities from query: {query}, text: {text[:50]}...")
    tokens = text.split()
    return ", ".join(tokens[:5]) if tokens else ""

# === TOOL: Send Feedback Email ===
def send_feedback_email(query: str, answer: str, feedback: str, comments: Optional[str] = "") -> str:
    """Send user feedback via email to the application manager."""
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        raise ValueError("Email credentials (SENDER_EMAIL/SENDER_PASSWORD) not configured.")

    subject = "RAG App Feedback: " + ("Positive" if feedback == "positive" else "Negative")
    body = f"""
Query:
{query}

Answer:
{answer}

Feedback: {feedback.title()}
Comments: {comments or "None"}
    """.strip()

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL

    try:
        # Capture SMTP debug output (conversation) into a buffer that we'll log.
        debug_buffer = io.StringIO()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            # Enable smtplib internal debugging (writes to sys.stderr by default)
            server.set_debuglevel(1)

            # Temporarily redirect stderr to capture smtp debug prints
            old_stderr = sys.stderr
            try:
                sys.stderr = debug_buffer
                # IMPORTANT: do NOT log or print SENDER_PASSWORD anywhere.
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
            finally:
                sys.stderr = old_stderr

        smtp_debug = debug_buffer.getvalue()
        if smtp_debug:
            log.info("SMTP debug output:\n%s", smtp_debug)

        log.info("Feedback email sent: %s", feedback)
        return "Feedback sent successfully."
    except Exception as e:
        # Full traceback text
        tb_text = traceback.format_exc()
        # Log the error with stack trace and SMTP debug buffer if present
        try:
            # If debug_buffer may not be defined because error happened earlier, safe-get it
            smtp_debug = locals().get("smtp_debug", None)
        except Exception:
            smtp_debug = None

        log.error("Email send failed: %s", repr(e))
        log.error("Traceback:\n%s", tb_text)
        if smtp_debug:
            log.error("SMTP debug output (captured):\n%s", smtp_debug)

        # Raise a RuntimeError that includes the error and traceback for callers, and chain original.
        raise RuntimeError(f"Failed to send email: {repr(e)}\nSee logs for full traceback and SMTP debug.") from e

# === REGISTER TOOLS ===
TOOLS = {
    "refine_query_tool": {
        "function": refine_query_tool,
        "description": "Refine the original query for better retrieval.",
        "parameters": ["original_query"]
    },
    "extract_entities_tool": {
        "function": extract_entities_tool,
        "description": "Extract key entities from the provided text based on the query.",
        "parameters": ["query", "text"]
    },
    "send_feedback_email": {
        "function": send_feedback_email,
        "description": "Send user feedback via email to the application manager.",
        "parameters": ["query", "answer", "feedback", "comments"]
    }
}

# === API MODELS & ENDPOINTS ===
class ToolCallRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]

@app.get("/tools")
def list_tools():
    """List all available tools."""
    return {
        "tools": [
            {"name": name, "description": tool["description"], "parameters": tool["parameters"]}
            for name, tool in TOOLS.items()
        ]
    }

from fastapi import HTTPException

@app.post("/call_tool")
def call_tool(request: ToolCallRequest):
    tool_name = request.tool_name
    if tool_name not in TOOLS:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    try:
        result = TOOLS[tool_name]["function"](**request.parameters)
        return {"result": result}
    except ValueError as e:
        # config / parameter errors -> 400
        log.error(f"Tool {tool_name} failed with ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception(f"Tool {tool_name} execution failed")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

# === RUN SERVER ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8050)