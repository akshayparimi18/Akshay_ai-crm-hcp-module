from database import SessionLocal, Interaction
import json

def test_db():
    print("Testing database connection and insertion...")
    db = SessionLocal()
    try:
        test_interaction = Interaction(
            hcp_name="Test Doctor",
            interaction_type="Meeting",
            date="2026-04-23",
            time="10:00 AM",
            sentiment="Positive",
            topics_discussed="Asthma inhalers",
            samples_distributed="5",
            ai_suggested_follow_ups=json.dumps(["Follow up in a week"])
        )
        db.add(test_interaction)
        db.commit()
        print("Success! Test interaction committed to PostgreSQL.")
        
        # Verify
        latest = db.query(Interaction).order_by(Interaction.id.desc()).first()
        print(f"Retrieved Interaction ID: {latest.id} for {latest.hcp_name}")
        
        # Cleanup
        db.delete(latest)
        db.commit()
        print("Cleanup successful.")
    except Exception as e:
        print(f"Error during database test: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    test_db()
