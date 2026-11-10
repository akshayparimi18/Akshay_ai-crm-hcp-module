import os
import json
import traceback
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import Base, engine
from agent import get_agent, GROQ_MODEL
from langchain_core.messages import HumanMessage

# Ensure database tables are created on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI-First CRM HCP Module", 
    description="Production-ready FastAPI backend using LangGraph stateful agents and dual Groq LLMs"
)

# CORS configuration allowing Vercel deployment and local dev origins
allowed_origins = [
    "https://akshay-ai-crm-hcp-module.vercel.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
@app.get("/health")
async def health_check():
    """Health check endpoint to verify uptime and active model."""
    return {
        "status": "ok",
        "model": os.getenv("GROQ_MODEL", GROQ_MODEL)
    }

@app.get("/api/models")
async def list_available_models():
    """Diagnostic endpoint to inspect active models on the Groq account."""
    import urllib.request, json, urllib.error
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        return {"error": "GROQ_API_KEY environment variable is missing or empty"}
    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {key}",
                "User-Agent": "Groq-Python-SDK/0.37.1",
                "Content-Type": "application/json"
            }
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return {
                "api_key_configured": bool(key),
                "active_model": os.getenv("GROQ_MODEL", GROQ_MODEL),
                "models": [m["id"] for m in data.get("data", [])]
            }
    except urllib.error.HTTPError as e:
        return {
            "error": "Groq API HTTP Error",
            "status": e.code,
            "body": e.read().decode()
        }
    except Exception as e:
        return {"error": "Unexpected error", "detail": str(e)}

class ChatRequest(BaseModel):
    text: Optional[str] = None
    message: Optional[str] = None

    def get_prompt(self) -> str:
        prompt = (self.text or self.message or "").strip()
        return prompt

@app.post("/api/chat")
async def handle_chat(request: ChatRequest):
    user_prompt = request.get_prompt()
    if not user_prompt:
        raise HTTPException(status_code=400, detail="Text or message input is required")
        
    try:
        # Get the compiled LangGraph agent
        agent = get_agent()
        
        # 1. Run the LangGraph agent
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_prompt)]},
            config={"recursion_limit": 25}
        )
        
        # 2. Extract final reply from agent
        final_reply = result["messages"][-1].content
        print(f"📨 FINAL REPLY: {repr(final_reply)}")
        
        # Fallback if reasoning returned empty content
        if not final_reply:
            for msg in reversed(result["messages"]):
                if hasattr(msg, 'content') and msg.content and not hasattr(msg, 'tool_call_id'):
                    if not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                        final_reply = msg.content
                        break

        # 3. Multi-Tool Data Aggregation
        now = datetime.now()
        default_date = now.strftime("%Y-%m-%d")
        default_time = now.strftime("%H:%M")

        aggregated_data = {
            "hcp_name": "",
            "interaction_type": "Meeting",
            "date": default_date,
            "time": default_time,
            "attendees": "",
            "topics_discussed": "",
            "materials_shared": "",
            "samples_distributed": "",
            "sentiment": "Positive",
            "outcomes": "",
            "follow_up_actions": "",
            "ai_suggested_follow_ups": []
        }

        # Inspect all tool calls across all messages in this turn
        for msg in result["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    args = tc.get('args', {}) or {}
                    name = tc.get('name', '')

                    if name in ('log_complete_interaction', 'edit_interaction'):
                        for key, val in args.items():
                            if val is not None and str(val).strip() != "":
                                # Normalize sentiment
                                if key == "sentiment":
                                    s_val = str(val).capitalize()
                                    if s_val in ["Positive", "Neutral", "Negative"]:
                                        aggregated_data["sentiment"] = s_val
                                # Format samples distributed (e.g., 10 -> "10 samples")
                                elif key == "samples_distributed":
                                    s_val = str(val).strip()
                                    if s_val.isdigit():
                                        aggregated_data["samples_distributed"] = f"{s_val} samples"
                                    else:
                                        aggregated_data["samples_distributed"] = s_val
                                else:
                                    aggregated_data[key] = val

                    elif name == 'generate_ai_follow_ups':
                        suggestions = args.get('suggestions') or []
                        if isinstance(suggestions, list) and suggestions:
                            aggregated_data['ai_suggested_follow_ups'] = suggestions
                        elif isinstance(suggestions, str):
                            try:
                                parsed = json.loads(suggestions)
                                if isinstance(parsed, list):
                                    aggregated_data['ai_suggested_follow_ups'] = parsed
                            except Exception:
                                aggregated_data['ai_suggested_follow_ups'] = [suggestions]

        # Apply robust fallbacks for required form inputs if missing
        if not aggregated_data.get("interaction_type"):
            aggregated_data["interaction_type"] = "Meeting"
        if not aggregated_data.get("date"):
            aggregated_data["date"] = default_date
        if not aggregated_data.get("time"):
            aggregated_data["time"] = default_time
        if not aggregated_data.get("sentiment"):
            aggregated_data["sentiment"] = "Positive"

        # If topics_discussed is empty, infer from context
        if not aggregated_data.get("topics_discussed"):
            if aggregated_data.get("samples_distributed"):
                aggregated_data["topics_discussed"] = f"Product discussion & sample provision ({aggregated_data['samples_distributed']})"
            else:
                aggregated_data["topics_discussed"] = "Discussion on product clinical data and efficacy"

        return {
            "reply": final_reply,
            "form_data": aggregated_data,
            "extracted_data": aggregated_data
        }
        
    except Exception as e:
        print(f"CRITICAL ERROR IN /api/chat:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
