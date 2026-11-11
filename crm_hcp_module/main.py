import os
import re
import json
import traceback
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Union
from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import Base, engine
from agent import get_agent, GROQ_MODEL, IST
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

def user_mentioned_date(prompt: str) -> bool:
    patterns = [
        r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\b(?:jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b',
        r'\b(?:yesterday|tomorrow)\b',
        r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b',
        r'\b\d{1,2}(?:st|nd|rd|th)\b'
    ]
    return any(re.search(pat, prompt, re.IGNORECASE) for pat in patterns)

def user_mentioned_time(prompt: str) -> bool:
    patterns = [
        r'\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b',
        r'\b\d{1,2}:\d{2}\b',
        r"\b\d{1,2}\s*o'?clock\b",
        r'\b(?:morning|afternoon|evening|night|noon|midnight)\b',
        r'\bat\s+\d{1,2}\b'
    ]
    return any(re.search(pat, prompt, re.IGNORECASE) for pat in patterns)

def deduplicate_doctor_names(doc_list: List[str]) -> List[str]:
    """
    Cleans and deduplicates a list of doctor names.
    If a surname-only reference (e.g. 'Dr. Nair' or 'Dr. Verma') appears alongside
    a full name ending in that same surname (e.g. 'Dr. Priya Nair' or 'Dr. Ananya Verma'),
    the short duplicate is cleanly removed.
    """
    cleaned = [d.strip() for d in doc_list if d and d.strip()]
    result = []
    for d in cleaned:
        tokens = re.sub(r'^(?:Dr\.|Doctor)\s*', '', d).split()
        if len(tokens) == 1:
            surname = tokens[0].lower()
            has_full_name = any(
                len(re.sub(r'^(?:Dr\.|Doctor)\s*', '', other).split()) > 1 and 
                re.sub(r'^(?:Dr\.|Doctor)\s*', '', other).split()[-1].lower() == surname
                for other in cleaned if other != d
            )
            if has_full_name:
                continue
        if d not in result:
            result.append(d)
    return result

@app.get("/")
@app.get("/health")
async def health_check():
    """Health check endpoint to verify uptime and active model."""
    now_ist = datetime.now(IST)
    return {
        "status": "ok",
        "model": os.getenv("GROQ_MODEL", GROQ_MODEL),
        "ist_date": now_ist.strftime("%Y-%m-%d"),
        "ist_time": now_ist.strftime("%H:%M")
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

        # 3. Dynamic Indian Standard Time (IST, UTC+5:30) calculation
        now_ist = datetime.now(IST)
        default_date = now_ist.strftime("%Y-%m-%d")
        default_time = now_ist.strftime("%H:%M")

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
                                # Format samples distributed
                                elif key == "samples_distributed":
                                    s_val = str(val).strip()
                                    if s_val.isdigit():
                                        aggregated_data["samples_distributed"] = f"{s_val} samples"
                                    else:
                                        aggregated_data["samples_distributed"] = s_val
                                elif key == "attendees":
                                    if isinstance(val, list):
                                        aggregated_data["attendees"] = ", ".join(str(a) for a in val if a)
                                    else:
                                        aggregated_data["attendees"] = str(val)
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

        # Extract all doctor names from user prompt & deduplicate short surname references
        docs_raw = re.findall(r'\b(?:Dr\.|Doctor)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?', user_prompt)
        docs_deduped = deduplicate_doctor_names(docs_raw)
        
        # Ensure hcp_name is set cleanly to the primary doctor
        if docs_deduped:
            aggregated_data["hcp_name"] = docs_deduped[0]
            # Ensure attendees contains all deduplicated participants
            aggregated_data["attendees"] = ", ".join(docs_deduped)
        elif aggregated_data.get("hcp_name"):
            aggregated_data["attendees"] = aggregated_data["hcp_name"]

        # Extract exact sample string from user prompt to prevent ANY accidental overwrite
        user_sample_match = re.search(r'\b(\d+\s+samples?(?:\s+of\s+[^.,;\n]+)?)', user_prompt, re.IGNORECASE)
        if user_sample_match:
            aggregated_data["samples_distributed"] = user_sample_match.group(1).strip()

        # Date & Time IST Enforcement:
        # If user did NOT explicitly mention date/time, ALWAYS enforce live IST date & time
        if not user_mentioned_date(user_prompt) or not aggregated_data.get("date"):
            aggregated_data["date"] = default_date
        if not user_mentioned_time(user_prompt) or not aggregated_data.get("time"):
            aggregated_data["time"] = default_time

        # Fallbacks for interaction_type and sentiment
        if not aggregated_data.get("interaction_type"):
            aggregated_data["interaction_type"] = "Meeting"
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
