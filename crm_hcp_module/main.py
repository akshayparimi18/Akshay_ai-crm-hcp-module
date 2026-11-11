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
from tools import cap_samples_to_limit
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

def polish_reply(reply: str, user_prompt: str) -> str:
    """
    Guarantees AI reply aligns with the 10-sample compliance rule:
    - If user provided <= 10 samples (e.g. 5 samples of OncoBoost 50mg), strips any hallucinated 'adjusted from...' or 'capped at...'.
    - If user provided > 10 samples (e.g. 12 samples of CardioShield 10mg), ensures reply states 10 samples and politely mentions
      the compliance limit cap.
    """
    m = re.search(r'\b(\d+)\s+samples?\b', user_prompt, re.IGNORECASE)
    if not m:
        return reply
    qty = int(m.group(1))
    if qty <= 10:
        # User specified <= 10 samples: must NEVER claim adjustment
        reply = re.sub(r'\s*\((?:adjusted|capped)[^)]*\)', '', reply, flags=re.IGNORECASE)
        reply = re.sub(r'\s*,?\s*adjusted from the \d+ you mentioned', '', reply, flags=re.IGNORECASE)
        reply = re.sub(r'\s*,?\s*capped from the \d+ you mentioned', '', reply, flags=re.IGNORECASE)
    else:
        # User specified > 10 samples: must cap at 10 and mention compliance
        reply = re.sub(rf'\b{qty}\s+samples\b', '10 samples', reply, flags=re.IGNORECASE)
        if not any(w in reply.lower() for w in ['complian', 'cap', 'limit', 'adjust', 'policy']):
            reply = reply.rstrip() + " (Note: Recorded sample count was capped at the compliance limit of 10 samples per interaction)."
    return reply

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
        
        # Ensure hcp_name is set cleanly to ONLY the single primary doctor
        if docs_deduped:
            primary_hcp = docs_deduped[0]
            aggregated_data["hcp_name"] = primary_hcp
            # Any additional physicians go into attendees
            secondary_docs = docs_deduped[1:]
            if secondary_docs:
                aggregated_data["attendees"] = ", ".join(secondary_docs)
            else:
                aggregated_data["attendees"] = ""
        elif aggregated_data.get("hcp_name"):
            # Check if multiple doctors were placed in hcp_name by tool caller, split them
            raw_hcp = aggregated_data["hcp_name"]
            if "," in raw_hcp or " and " in raw_hcp.lower():
                parts = [p.strip() for p in re.split(r'[,;]\s*|\s+and\s+', raw_hcp, flags=re.IGNORECASE) if p.strip()]
                if parts:
                    aggregated_data["hcp_name"] = parts[0]
                    if len(parts) > 1 and not aggregated_data.get("attendees"):
                        aggregated_data["attendees"] = ", ".join(parts[1:])

        # Clean attendees: ensure primary HCP and its short surname are not in attendees
        if aggregated_data.get("attendees") and aggregated_data.get("hcp_name"):
            prim = aggregated_data["hcp_name"]
            prim_tokens = re.sub(r'^(?:Dr\.|Doctor)\s*', '', prim).strip().split()
            prim_surname = prim_tokens[-1].lower() if prim_tokens else ""
            
            att_items = [a.strip() for a in re.split(r'[,;]\s*', str(aggregated_data["attendees"])) if a.strip()]
            att_filtered = []
            for a in att_items:
                if a.lower() == prim.lower():
                    continue
                a_tokens = re.sub(r'^(?:Dr\.|Doctor)\s*', '', a).strip().split()
                if len(a_tokens) == 1 and prim_surname and a_tokens[0].lower() == prim_surname:
                    continue
                att_filtered.append(a)
            att_filtered = deduplicate_doctor_names(att_filtered)
            aggregated_data["attendees"] = ", ".join(att_filtered)

        # Extract sample string from user prompt & enforce 10-sample compliance cap
        user_sample_match = re.search(r'\b(\d+\s+samples?(?:\s+of\s+[^.,;\n]+)?)', user_prompt, re.IGNORECASE)
        if user_sample_match:
            aggregated_data["samples_distributed"] = cap_samples_to_limit(user_sample_match.group(1).strip())
        elif aggregated_data.get("samples_distributed"):
            aggregated_data["samples_distributed"] = cap_samples_to_limit(aggregated_data["samples_distributed"])

        # Date & Time IST Enforcement:
        # If user did NOT explicitly mention date/time, ALWAYS enforce live IST date & time
        if not user_mentioned_date(user_prompt) or not aggregated_data.get("date"):
            aggregated_data["date"] = default_date
        if not user_mentioned_time(user_prompt) or not aggregated_data.get("time"):
            aggregated_data["time"] = default_time

        # Polish final reply for compliance limit alignment
        final_reply = polish_reply(final_reply, user_prompt)

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
