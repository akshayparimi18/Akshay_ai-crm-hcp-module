import os
import re
from datetime import datetime
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
    Extracts the integer quantity of drug samples from user notes or tool arguments.
    Crucially ignores dosage strengths (e.g. 50mg, 100 mg, 10ml, 500 mcg) so that
    dosages like '50mg' are NEVER mistakenly interpreted as 50 samples!
    """
    if not text:
        return None
    # Strip dosage indicators (e.g. 50mg, 100ml, 500mcg, 10g)
    cleaned = re.sub(r'\b\d+\s*(?:mg|ml|mcg|g|kg|units?|tablets?|capsules?|pills?)\b', '', str(text), flags=re.IGNORECASE)
    # Match explicitly against sample keywords
    match = re.search(r'\b(\d+)\s*(?:samples?|vials?|packs?|boxes?)?\b', cleaned, re.IGNORECASE)
    if match and match.group(1):
        return int(match.group(1))
    # Fallback: check remaining integers in cleaned text
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

    now = datetime.now()
    current_date_iso = now.strftime("%Y-%m-%d")
    current_time_24h = now.strftime("%H:%M")
    
    system_prompt = f"""You are an internal CRM assistant talking DIRECTLY to a pharmaceutical sales rep. Your job is to parse their messy natural language notes, extract structured information, and update the CRM database using your available tools.

CRITICAL EXTRACTION RULES:
1. `hcp_name` MUST contain ONLY the single primary doctor (e.g. "Dr. Ananya Verma"). NEVER put multiple doctors or colleagues into `hcp_name`.
2. Any additional physicians, colleagues, or clinic staff present (e.g. "Dr. Rahul Mehta") MUST be parsed into the `attendees` field (e.g. "Dr. Rahul Mehta" or ["Dr. Rahul Mehta"]).
3. Compliance rules ONLY apply a maximum ceiling limit of 10 samples. NEVER increase or change the sample count if the user specified 10 or fewer samples (e.g., 5 samples of OncoBoost 50mg). Log the exact quantity provided by the rep.
4. Today's date is {current_date_iso} (YYYY-MM-DD). The current time is {current_time_24h} (24-hour HH:MM). If the user does not explicitly specify a date or time, ALWAYS default `date` to "{current_date_iso}" and `time` to "{current_time_24h}".
5. ALWAYS use the `log_complete_interaction` tool first to capture the core data points. 
   - If interaction type is not specified, default `interaction_type` to "Meeting".
   - If topics discussed are not specified, infer a sensible description from context (e.g. "Discussion on product efficacy and trial data").
6. After logging the interaction, ALWAYS use the `generate_ai_follow_ups` tool to create 2-3 actionable next steps for the rep based on the meeting's topics and outcomes. You must do this automatically.
7. For `sentiment`, strictly use "Positive", "Neutral", or "Negative" (infer "Positive" when the interaction is favorable or cooperative).
8. If the user corrects you, use `edit_interaction`.
9. If you receive a COMPLIANCE ERROR in a tool response, follow its instructions EXACTLY to fix the tool call and retry. Do NOT repeat the same rejected values.
10. IMPORTANT: After ALL your tool calls have been successfully executed, you MUST generate a brief, friendly text summary confirming what was logged and any follow-up actions. Never end the conversation on a tool call — always finish with a human-readable message.

TOOL CALLING & SCHEMA RULES:
- Invoke only one tool at a time with strictly valid JSON arguments adhering exactly to the schema.
- For optional fields not mentioned in notes, pass null or sensible defaults rather than fabricating imaginary doctors or materials.
- Never invent additional attendees or clinical materials if not mentioned by the rep.

PERSONA RULES:
- You are talking to the SALES REP, not to a doctor or HCP. Address the rep directly (e.g., "I've logged your meeting...").
- Do NOT draft emails, letters, or messages addressed to the doctor unless the rep explicitly asks you to.

COMPLIANCE TRANSPARENCY:
- If a compliance guardrail triggered because the rep attempted to distribute MORE than 10 samples, your final message MUST explicitly notify the rep about this adjustment. For example: "Note: The sample quantity was reduced to the legal limit of 10."
- If 10 or fewer samples were distributed, do NOT mention any adjustment, as no compliance violation occurred."""

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
            return "compliance"
        return "reasoning"

    def compliance_guardrail(state: AgentState):
        last_message = state['messages'][-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                if tool_call["name"] in ["log_complete_interaction", "edit_interaction"]:
                    args = tool_call["args"]
                    samples_distributed = args.get("samples_distributed", "")
                    follow_up_actions = args.get("follow_up_actions", "")
                    
                    if "WARNING: Sample limit exceeded" in str(follow_up_actions):
                        print("✅ GUARDRAIL PASSED — AI self-corrected successfully")
                        continue
                        
                    if samples_distributed:
                        sample_count = parse_sample_count(str(samples_distributed))
                        # Compliance rule: ONLY applies a ceiling of 10. NEVER triggers for 10 or fewer samples!
                        if sample_count is not None and sample_count > 10:
                            print(f"⚠️ GUARDRAIL TRIGGERED: {sample_count} samples exceeds legal maximum ceiling of 10")
                            error_msg = (
                                f"COMPLIANCE ERROR: You attempted to log {sample_count} samples. "
                                f"The legal maximum limit is 10 samples. Your tool call was REJECTED and NOT executed. "
                                f"You MUST rewrite your tool call with these EXACT changes: "
                                f"1) Reduce samples_distributed to '10 samples' (the maximum allowed ceiling). "
                                f"2) APPEND the text 'WARNING: Sample limit exceeded' to the follow_up_actions field. "
                                f"Do NOT attempt to log {sample_count} samples again or the call will be rejected again."
                            )
                            return {"messages": [ToolMessage(
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                                content=error_msg
                            )]}
                            
        return {"messages": []}

    def compliance_router(state: AgentState):
        last_message = state['messages'][-1]
        if isinstance(last_message, ToolMessage):
            return "agent"
        return "tools"

    def call_reasoning_model(state: AgentState):
        messages = state['messages']
        
        filtered = []
        for m in messages:
            if isinstance(m, HumanMessage):
                filtered.append(m)
            elif hasattr(m, "tool_calls") and m.tool_calls:
                filtered.append(m)
            elif isinstance(m, ToolMessage) and "COMPLIANCE ERROR" not in m.content:
                filtered.append(m)
        
        reasoning_prompt = (
            "You are an internal CRM assistant speaking DIRECTLY to a pharmaceutical sales rep. "
            "Based on the conversation, generate a brief, friendly confirmation message. "
            "Mention ONLY the primary HCP name, what was logged, and any follow-up actions. "
            "Keep it to 2-3 sentences. Do NOT use any tools. Just reply with plain text. "
            "IMPORTANT RULES: "
            "1) Address the sales rep directly (e.g., 'I\'ve logged your meeting...'). Do NOT draft emails or messages to the doctor. "
            "2) The primary HCP name is the single primary doctor (e.g. 'Dr. Ananya Verma'). Mention additional attendees (e.g. 'Dr. Rahul Mehta') as colleagues/attendees present. "
            "3) Compliance rules ONLY apply a maximum ceiling limit of 10 samples. If the rep distributed 10 or fewer samples (e.g. 5 samples), state the EXACT sample count distributed (e.g. '5 samples of OncoBoost 50mg'). NEVER claim samples were adjusted unless the rep originally attempted to log MORE than 10 samples."
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
    workflow.add_node("compliance", compliance_guardrail)
    workflow.add_node("tools", tool_node)
    workflow.add_node("reasoning", call_reasoning_model)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"compliance": "compliance", "reasoning": "reasoning"})
    workflow.add_conditional_edges("compliance", compliance_router, {"agent": "agent", "tools": "tools"})
    workflow.add_edge("tools", "agent")
    workflow.add_edge("reasoning", END)
    
    app = workflow.compile()
    return app
