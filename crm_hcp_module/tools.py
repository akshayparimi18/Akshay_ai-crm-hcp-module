import json
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.tools import tool
from database import SessionLocal, Interaction

# =====================================================================
# Tool 1: log_complete_interaction
# =====================================================================
class LogInteractionInput(BaseModel):
    hcp_name: str = Field(description="Name of the Healthcare Professional")
    interaction_type: Optional[str] = Field(None, description="Type of interaction (e.g. Meeting, Call)")
    date: Optional[str] = Field(None, description="Date of the interaction")
    time: Optional[str] = Field(None, description="Time of the interaction")
    attendees: Optional[str] = Field(None, description="Other attendees")
    topics_discussed: Optional[str] = Field(None, description="Topics discussed")
    materials_shared: Optional[str] = Field(None, description="Materials or brochures shared")
    samples_distributed: Optional[str] = Field(None, description="Samples distributed")
    sentiment: Optional[str] = Field(None, description="Observed sentiment: Positive, Neutral, or Negative")
    outcomes: Optional[str] = Field(None, description="Outcomes of the meeting")
    follow_up_actions: Optional[str] = Field(None, description="Follow up actions agreed upon")

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
    """Logs a new interaction by extracting ALL 11 primary fields from the user's natural language input."""
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
    suggestions: List[str] = Field(description="List of 2-3 actionable next steps to populate the UI checkboxes.")

@tool("generate_ai_follow_ups", args_schema=GenerateAIFollowUpsInput)
def generate_ai_follow_ups(hcp_name: str, suggestions: List[str]) -> str:
    """Generates actionable follow-up suggestions based on the meeting context. CALL THIS AFTER log_complete_interaction."""
    db = SessionLocal()
    try:
        # We attach these suggestions to the most recent interaction for this HCP
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
    hcp_name: str = Field(description="Name of the HCP")
    interaction_type: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    attendees: Optional[str] = None
    topics_discussed: Optional[str] = None
    materials_shared: Optional[str] = None
    samples_distributed: Optional[str] = None
    sentiment: Optional[str] = Field(None, description="Positive, Neutral, or Negative")
    outcomes: Optional[str] = None
    follow_up_actions: Optional[str] = None

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
        for field, value in locals().items():
            if field not in ['hcp_name', 'db'] and value is not None:
                setattr(interaction, field, value)
            
        db.commit()
        return f"Successfully updated interaction for {hcp_name}."
    except Exception as e:
        db.rollback()
        return f"Failed to edit interaction: {str(e)}"
    finally:
        db.close()

# Export tools for LangGraph
crm_tools = [log_complete_interaction, generate_ai_follow_ups, edit_interaction]
