# AI-First HCP CRM Module 🩺🤖

![App Screenshot](screenshot.png)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18.0-blue.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon-336791.svg)](https://www.postgresql.org/)
[![LangGraph](https://img.shields.io/badge/AI-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)

## 🌟 Overview
This project is a production-ready **AI-First CRM Module** designed for pharmaceutical sales representatives. It transforms messy, natural language notes from HCP (Healthcare Professional) interactions into structured, validated database entries. 

By leveraging **LangGraph** and state-of-the-art LLMs, the system doesn't just log data—it enforces business logic and compliance rules autonomously before any data hits the cloud database.

## 🛡️ Key Features
- **LangGraph Compliance Guardrail**: A self-correcting agent loop that intercepts tool calls. If a sales rep attempt to log more than 10 samples (the legal limit), the guardrail rejects the call, instructs the LLM on the violation, and forces a compliant rewrite.
- **Natural Language Parsing**: Reps can type or dictate notes naturally (e.g., *"Met Dr. Smith today at 2pm, left 5 samples"*), and the agent extracts all 11 core data points automatically.
- **AI Follow-up Generation**: Automatically suggests actionable next steps based on the meeting context.
- **PostgreSQL Cloud Integration**: Fully migrated from SQLite to **PostgreSQL (Neon)** for enterprise-grade data persistence and scalability.
- **Real-time "Double Payload" Sync**: The backend returns both a conversational text reply and a structured JSON object to instantly populate the React frontend forms.

## 💻 Technical Stack
### **Backend**
- **Core**: Python & FastAPI
- **AI Orchestration**: LangGraph & LangChain
- **LLMs**: Llama 3.1 (via Groq Cloud)
- **ORM**: SQLAlchemy

### **Frontend**
- **Framework**: React (Vite)
- **State Management**: Redux Toolkit
- **Styling**: Vanilla CSS with modern UI/UX patterns

### **Database**
- **Cloud Provider**: Neon PostgreSQL
- **Driver**: Psycopg2-binary

---

## 🚀 Local Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/akshayparimi18/Akshay_ai-crm-hcp-module.git
cd Akshay_ai-crm-hcp-module
```

### 2. Backend Setup
```bash
cd crm_hcp_module
pip install -r requirements.txt
```

Create a `.env` file in the `crm_hcp_module` directory:
```env
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=postgresql://user:pass@hostname/db_name?sslmode=require
```

Run the FastAPI server:
```bash
uvicorn main:app --reload
```

### 3. Frontend Setup
```bash
cd ../crm_frontend
npm install
```

Run the React app:
```bash
npm run dev
```

The app will be available at `http://localhost:5173`. Ensure the backend is running on `http://127.0.0.1:8000`.

---

## 📜 License
This project is for educational/assignment purposes as part of the AI-First CRM project.
