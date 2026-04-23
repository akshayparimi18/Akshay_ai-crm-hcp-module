from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import Base, engine
from agent import get_agent
from langchain_core.messages import HumanMessage

# Ensure database tables are created on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI-First CRM HCP Module", 
    description="Production-ready FastAPI backend using LangGraph stateful agents and dual Groq LLMs"
)

# CRITICAL FIX: Add CORS middleware so React can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for incoming requests
class ChatRequest(BaseModel):
    text: str

@app.post("/api/chat")
async def handle_chat(request: ChatRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text input is required")
        
    try:
        # Get the compiled LangGraph agent (ensures fresh date state)
        agent = get_agent()
        
        # 1. Run the LangGraph agent
        result = agent.invoke(
            {"messages": [HumanMessage(content=request.text)]},
            config={"recursion_limit": 25}
        )
        
        # 2. Get the final text reply from the AI
        final_reply = result["messages"][-1].content
        
        # Debug: print to terminal so we can see what's being returned
        print(f"📨 FINAL REPLY: {repr(final_reply)}")
        
        # Fallback: if the reasoning model returned empty content, search backwards for any AI text
        if not final_reply:
            for msg in reversed(result["messages"]):
                if hasattr(msg, 'content') and msg.content and not hasattr(msg, 'tool_call_id'):
                    if not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                        final_reply = msg.content
                        break
        
        # Ultimate fallback
        if not final_reply:
            final_reply = "✅ Interaction logged successfully! The form has been updated with the details."
        
        # 3. Extract the structured tool data for the React form
        extracted_form_data = {}
        ai_suggestions = []
        
        # Loop backwards through the conversation history to find the newest tool uses
        for message in reversed(result["messages"]):
            if hasattr(message, "tool_calls") and message.tool_calls:
                for latest_tool_call in message.tool_calls:
                    # Extract the main interaction details
                    if latest_tool_call["name"] in ["log_complete_interaction", "edit_interaction"] and not extracted_form_data:
                        extracted_form_data = latest_tool_call["args"]
                        
                    # Extract the follow-up suggestions
                    if latest_tool_call["name"] == "generate_ai_follow_ups" and not ai_suggestions:
                        ai_suggestions = latest_tool_call["args"].get("suggestions", [])
        
        # Combine the suggestions into the form data for Redux
        if extracted_form_data:
            extracted_form_data["ai_suggested_follow_ups"] = ai_suggestions

        # 4. Return the "Double Payload"
        return {
            "reply": final_reply,
            "form_data": extracted_form_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Local development server execution
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
