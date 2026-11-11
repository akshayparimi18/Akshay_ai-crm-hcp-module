import os
import re
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from typing import Annotated, Optional, TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from tools import crm_tools

# Load environment variables
load_dotenv()

# Indian Standard Time (IST, UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

# Model & API configuration with whitespace stripping
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b").strip()
GROQ_TERTIARY_MODEL = os.getenv("GROQ_TERTIARY_MODEL", "qwen/qwen3.8-27b").strip()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def create_groq_llm(model_name: str, temperature: float = 0.0):
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        api_key=GROQ_API_KEY or None
    )

def parse_sample_count(text: str) -> Optional[int]:
    """
    Extracts the integer quantity of drug samples from text.
    Strips dosage strings (e.g. 50mg, 100ml) so '50mg' is never confused with 50 samples.
    """
    if not text:
        return None
    cleaned = re.sub(r'\b\d+\s*(?:mg|ml|mcg|g|kg|units?|tablets?|capsules?|pills?)\b', '', str(text), flags=re.IGNORECASE)
    match = re.search(r'\b(\d+)\s*(?:samples?|vials?|packs?|boxes?)?\b', cleaned, re.IGNORECASE)
    if match and match.group(1):
        return int(match.group(1))
    nums = [int(n) for n in re.findall(r'\b\d+\b', cleaned)]
    if nums:
        return nums[0]
    return None

def get_agent():
    # Primary model (120B / configured model)
    primary_tool_llm = create_groq_llm(GROQ_MODEL, temperature=0.0)
    primary_reasoning_llm = create_groq_llm(GROQ_MODEL, temperature=0.1)
    
    # Fallback models: 20B and Qwen 27B
    fallback_20b_tool_llm = create_groq_llm(GROQ_FALLBACK_MODEL, temperature=0.0)
    fallback_tertiary_tool_llm = create_groq_llm(GROQ_TERTIARY_MODEL, temperature=0.0)
    
    # Bind tools to models
    primary_tool_caller = primary_tool_llm.bind_tools(crm_tools)
    fallback_20b_caller = fallback_20b_tool_llm.bind_tools(crm_tools)
    fallback_tertiary_caller = fallback_tertiary_tool_llm.bind_tools(crm_tools)
    
    # LangChain declarative fallback chain
    tool_caller_with_fallbacks = primary_tool_caller.with_fallbacks([
        fallback_20b_caller,
        fallback_tertiary_caller
    ])
    
    fallback_20b_reasoning = create_groq_llm(GROQ_FALLBACK_MODEL, temperature=0.1)
    fallback_tertiary_reasoning = create_groq_llm(GROQ_TERTIARY_MODEL, temperature=0.1)
    reasoning_with_fallbacks = primary_reasoning_llm.with_fallbacks([
        fallback_20b_reasoning,
        fallback_tertiary_reasoning
    ])
    
    tool_node = ToolNode(crm_tools)

    # Calculate real-time IST date and 24-hour time dynamically on every request
    now_ist = datetime.now(IST)
    current_date_iso = now_ist.strftime("%Y-%m-%d")
    current_time_24h = now_ist.strftime("%H:%M")
    
    system_prompt = f"""You are an internal CRM assistant talking DIRECTLY to a pharmaceutical sales rep. Your job is to parse their messy natural language notes, extract structured information, and update the CRM database using your available tools.

CRITICAL EXTRACTION & FIELD RULES:
1. `hcp_name`: Primary Healthcare Professional (e.g. "Dr. Priya Nair").
2. `attendees`: ALL doctors, colleagues, or staff present at the meeting (e.g. "Dr. Priya Nair, Dr. Karthik Reddy"). NEVER leave `attendees` blank when meeting participants are named. If a doctor is referred to later by short surname (e.g. "Dr. Nair"), do not duplicate them in `attendees`.
3. `samples_distributed`: MUST preserve the EXACT quantity and product name stated by the user (e.g., "12 samples of CardioShield 10mg" or "5 samples of OncoBoost 50mg"). NEVER overwrite, reduce, or replace the user's sample count with "10 samples".
4. `materials_shared`: Clinical brochures, guides, trial summaries, or monographs shared (e.g., "clinical dosing guide").
5. `topics_discussed`: Clinical topics, trial data, efficacy, or dosing discussed (e.g., "CardioShield hypertension data").
6. `sentiment`: Strictly "Positive", "Neutral", or "Negative" (infer "Positive" when the interaction is favorable).
7. `outcomes`: Agreed outcomes, physician reactions, or promises made.
8. `follow_up_actions`: Agreed next steps, MSL coordination, or delivery commitments.
9. Today's date is {current_date_iso} (YYYY-MM-DD IST). The current time is {current_time_24h} (24-hour HH:MM IST). If the user does not specify date or time, default to "{current_date_iso}" and "{current_time_24h}".
10. ALWAYS use the `log_complete_interaction` tool first to record all captured fields.
11. After logging the interaction, ALWAYS use the `generate_ai_follow_ups` tool to generate 2-3 actionable next steps.
12. IMPORTANT: After ALL tool calls have executed, generate a brief, friendly text confirmation addressed directly to the sales rep confirming the primary doctor, attendees, materials shared, and the EXACT samples recorded. NEVER claim samples were adjusted unless explicitly instructed."""

    def call_tool_model(state: AgentState):
        messages = state['messages']
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt)] + messages
            
        try:
            response = tool_caller_with_fallbacks.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            print(f"⚠️ Tool caller primary with_fallbacks encountered error: {e}. Retrying rescue...")
            try:
                response = fallback_20b_caller.invoke(messages)
                return {"messages": [response]}
            except Exception as secondary_err:
                print(f"⚠️ Rescue fallback 20b failed: {secondary_err}. Falling back to Qwen...")
                response = fallback_tertiary_caller.invoke(messages)
                return {"messages": [response]}

    def should_continue(state: AgentState):
        last_message = state['messages'][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return "reasoning"

    def call_reasoning_model(state: AgentState):
        messages = state['messages']
        
        filtered = []
        for m in messages:
            if isinstance(m, HumanMessage):
                filtered.append(m)
            elif hasattr(m, "tool_calls") and m.tool_calls:
                filtered.append(m)
            elif isinstance(m, ToolMessage):
                filtered.append(m)
        
        reasoning_prompt = (
            "You are an internal CRM assistant speaking DIRECTLY to a pharmaceutical sales rep. "
            "Based on the conversation, generate a brief, friendly confirmation message. "
            "Mention the primary HCP name, all attendees, materials shared, what was logged, and any follow-up actions. "
            "Keep it to 2-3 sentences. Do NOT use any tools. Just reply with plain text. "
            "CRITICAL RULES: "
            "1) Address the sales rep directly (e.g., 'I\'ve logged your meeting...'). "
            "2) Confirm the EXACT samples and product name provided by the rep (e.g. '12 samples of CardioShield 10mg' or '5 samples of OncoBoost 50mg'). "
            "3) NEVER say samples were adjusted to 10 or changed. Always confirm the exact samples recorded."
        )
        summary_messages = [SystemMessage(content=reasoning_prompt)] + filtered
        try:
            response = reasoning_with_fallbacks.invoke(summary_messages)
            return {"messages": [response]}
        except Exception as e:
            print(f"⚠️ Reasoning model fallback: {e}")
            try:
                response = fallback_20b_reasoning.invoke(summary_messages)
                return {"messages": [response]}
            except Exception as e2:
                response = fallback_tertiary_reasoning.invoke(summary_messages)
                return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_tool_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("reasoning", call_reasoning_model)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "reasoning": "reasoning"})
    workflow.add_edge("tools", "agent")
    workflow.add_edge("reasoning", END)
    
    app = workflow.compile()
    return app
