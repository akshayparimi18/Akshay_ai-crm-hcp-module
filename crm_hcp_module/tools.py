import json
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.tools import tool
from database import SessionLocal, Interaction

# =====================================================================
# Tool 1: log_complete_interaction
# =====================================================================
class LogInteractionInput(BaseModel):
    hcp_name: str = Field(description="Full name of the Healthcare Professional (e.g. Dr. Rajesh Sharma)")
    interaction_type: Optional[str] = Field(default=None, description="Type of interaction (e.g. In-Person Meeting, Phone Call, Email, Presentation)")
    date: Optional[str] = Field(default=None, description="Date of the interaction in YYYY-MM-DD format (infer from context or today's date)")
    time: Optional[str] = Field(default=None, description="Time of the interaction (e.g. 10:30 AM)")
    attendees: Optional[str] = Field(default=None, description="Names of other attendees, clinic staff, or colleagues present")
    topics_discussed: Optional[str] = Field(default=None, description="Clinical topics, drug details, trial data, or subjects discussed")
    materials_shared: Optional[str] = Field(default=None, description="Materials, brochures, clinical trial papers, or decks shared")
    samples_distributed: Optional[str] = Field(default=None, description="Number or description of drug samples distributed (e.g. '5 samples')")
    sentiment: Optional[str] = Field(default=None, description="Observed sentiment: strictly 'Positive', 'Neutral', or 'Negative'")
    outcomes: Optional[str] = Field(default=None, description="Key takeaways, conclusions, or physician response")
    follow_up_actions: Optional[str] = Field(default=None, description="Follow-up tasks, scheduled meetings, or commitments agreed upon")

@tool("log_complete_interaction", args_schema=LogInteractionInput)
def log_complete_interaction(
    hcp_name: str, 
    interaction_type: Optional[str] = None, 
    date: Optional[str] = None, 
    time: Optional[str] = None, 
    attendees: Optional[str] = None, 
    topics_discussed: Optional[str] = None, 
    materials_shared: Optional[str] = None, 
    samples_distributed: Optional[str] = None, 
    sentiment: Optional[str] = None, 
    outcomes: Optional[str] = None, 
    follow_up_actions: Optional[str] = None
) -> str:
    """Logs a new interaction by extracting primary fields from the user's natural language input."""
    db = SessionLocal()
    try:
        new_interaction = Interaction(
            hcp_name=hcp_name,
            interaction_type=interaction_type,
            date=date,
            time=time,
            attendees=attendees,
            topics_discussed=topics_discussed,
            materials_shared=materials_shared,
            samples_distributed=samples_distributed,
            sentiment=sentiment,
            outcomes=outcomes,
            follow_up_actions=follow_up_actions
        )
        db.add(new_interaction)
        db.commit()
        db.refresh(new_interaction)
        return f"Successfully logged interaction {new_interaction.id} for {hcp_name}."
    except Exception as e:
        db.rollback()
        return f"Failed to log interaction: {str(e)}"
    finally:
        db.close()

# =====================================================================
# Tool 2: generate_ai_follow_ups
# =====================================================================
class GenerateAIFollowUpsInput(BaseModel):
    hcp_name: str = Field(description="Name of the HCP")
    suggestions: List[str] = Field(default_factory=list, description="List of 2-3 actionable next steps to populate the UI checkboxes.")

@tool("generate_ai_follow_ups", args_schema=GenerateAIFollowUpsInput)
def generate_ai_follow_ups(hcp_name: str, suggestions: List[str]) -> str:
    """Generates actionable follow-up suggestions based on the meeting context. CALL THIS AFTER log_complete_interaction."""
    db = SessionLocal()
    try:
        interaction = db.query(Interaction).filter(Interaction.hcp_name == hcp_name).order_by(Interaction.id.desc()).first()
        if interaction:
            interaction.ai_suggested_follow_ups = json.dumps(suggestions)
            db.commit()
            return f"Successfully attached {len(suggestions)} AI follow-up suggestions."
        return f"Could not find an interaction for {hcp_name} to attach suggestions to."
    except Exception as e:
        db.rollback()
        return f"Failed to attach AI suggestions: {str(e)}"
    finally:
        db.close()

# =====================================================================
# Tool 3: edit_interaction
# =====================================================================
class EditInteractionInput(BaseModel):
    hcp_name: str = Field(description="Name of the Healthcare Professional whose record is being updated")
    interaction_type: Optional[str] = Field(default=None, description="Updated interaction type")
    date: Optional[str] = Field(default=None, description="Updated date in YYYY-MM-DD format")
    time: Optional[str] = Field(default=None, description="Updated time")
    attendees: Optional[str] = Field(default=None, description="Updated attendees")
    topics_discussed: Optional[str] = Field(default=None, description="Updated topics discussed")
    materials_shared: Optional[str] = Field(default=None, description="Updated materials shared")
    samples_distributed: Optional[str] = Field(default=None, description="Updated samples distributed")
    sentiment: Optional[str] = Field(default=None, description="Updated sentiment: strictly 'Positive', 'Neutral', or 'Negative'")
    outcomes: Optional[str] = Field(default=None, description="Updated outcomes")
    follow_up_actions: Optional[str] = Field(default=None, description="Updated follow-up actions")

@tool("edit_interaction", args_schema=EditInteractionInput)
def edit_interaction(
    hcp_name: str,
    interaction_type: Optional[str] = None, 
    date: Optional[str] = None, 
    time: Optional[str] = None, 
    attendees: Optional[str] = None, 
    topics_discussed: Optional[str] = None, 
    materials_shared: Optional[str] = None, 
    samples_distributed: Optional[str] = None, 
    sentiment: Optional[str] = None, 
    outcomes: Optional[str] = None, 
    follow_up_actions: Optional[str] = None
) -> str:
    """Updates an existing interaction record. Use this if the user issues a correction to any field."""
    db = SessionLocal()
    try:
        interaction = db.query(Interaction).filter(Interaction.hcp_name == hcp_name).order_by(Interaction.id.desc()).first()
        if not interaction:
            return f"No interaction found for {hcp_name}."
        
        # Update fields dynamically if they are provided
        updates = {
            "interaction_type": interaction_type,
            "date": date,
            "time": time,
            "attendees": attendees,
            "topics_discussed": topics_discussed,
            "materials_shared": materials_shared,
            "samples_distributed": samples_distributed,
            "sentiment": sentiment,
            "outcomes": outcomes,
            "follow_up_actions": follow_up_actions
        }
        for field, value in updates.items():
            if value is not None:
                setattr(interaction, field, value)
            
        db.commit()
        return f"Successfully updated interaction for {hcp_name}."
    except Exception as e:
        db.rollback()
        return f"Failed to edit interaction: {str(e)}"
    finally:
        db.close()

# =====================================================================
# Tool 4: get_hcp_history
# =====================================================================
class GetHCPHistoryInput(BaseModel):
    hcp_name: str = Field(description="Name of the Healthcare Professional to search for past meetings")

@tool("get_hcp_history", args_schema=GetHCPHistoryInput)
def get_hcp_history(hcp_name: str) -> str:
    """Retrieves past interaction records for a specific Healthcare Professional. Use this if the rep asks 'Have I met with Dr. X before?'"""
    db = SessionLocal()
    try:
        interactions = db.query(Interaction).filter(Interaction.hcp_name == hcp_name).order_by(Interaction.id.desc()).limit(3).all()
        if not interactions:
            return f"No past interactions found in the database for {hcp_name}."
        
        history = [f"Date: {i.date}, Topic: {i.topics_discussed}, Sentiment: {i.sentiment}" for i in interactions]
        return f"Recent history for {hcp_name}: " + " | ".join(history)
    except Exception as e:
        return f"Failed to retrieve history: {str(e)}"
    finally:
        db.close()

# =====================================================================
# Tool 5: escalate_compliance_issue
# =====================================================================
class EscalateIssueInput(BaseModel):
    hcp_name: str = Field(description="Name of the Healthcare Professional")
    reason: str = Field(description="Specific reason for escalating compliance violation to regional management")

@tool("escalate_compliance_issue", args_schema=EscalateIssueInput)
def escalate_compliance_issue(hcp_name: str, reason: str) -> str:
    """Escalates a compliance violation or critical issue to the regional manager."""
    return f"ALERT SENT: Successfully escalated issue regarding {hcp_name} to the Regional Manager. Reason: {reason}"

# =====================================================================
# Export ALL 5 tools for LangGraph
# =====================================================================
crm_tools = [
    log_complete_interaction, 
    generate_ai_follow_ups, 
    edit_interaction, 
    get_hcp_history, 
    escalate_compliance_issue
]
