import os
import traceback
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

# CORS configuration
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
                "key_preview": f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "too-short",
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
    text: str

@app.post("/api/chat")
async def handle_chat(request: ChatRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text input is required")
        
    try:
        # Get the compiled LangGraph agent
        agent = get_agent()
        
        # 1. Run the LangGraph agent
        result = agent.invoke(
            {"messages": [HumanMessage(content=request.text)]},
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
                        
        # 3. Extract parsed structured data from log_complete_interaction
        extracted_data = {}
        for msg in result["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc['name'] == 'log_complete_interaction':
                        extracted_data = tc['args']
                        break
                        
        return {
            "reply": final_reply,
            "extracted_data": extracted_data
        }
        
    except Exception as e:
        print(f"CRITICAL ERROR IN /api/chat:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
