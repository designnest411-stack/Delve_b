# Delve Backend — Multi-Agent Deep Research Engine

FastAPI backend service powering autonomous multi-agent academic research, literature synthesis, and publication-grade manuscript generation.

---

## Tech Stack
- **Runtime:** Python 3.12, FastAPI, Uvicorn
- **Orchestration:** LangGraph / StateGraph
- **LLM Engine:** Google Gemini REST API (Free-Tier Cascade & Cooldown Handler)
- **Database:** Supabase (PostgreSQL, Row Level Security, Auth JWT, pgvector, Storage)
- **Background Jobs:** Upstash QStash (Durable Execution) & Upstash Redis REST (Rate Limiting)

---

## Setup & Local Run

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env

# 4. Start backend server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## Database Migrations
Execute SQL migration scripts in order within the Supabase SQL Editor:
1. `supabase/migrations/001_delve_public_beta.sql`
2. `supabase/migrations/002_lifetime_quota.sql`
3. `supabase/migrations/003_increase_paper_quota.sql`

---

## Deployment (Render)
This repository is configured for direct deployment on Render via Docker using `render.yaml`.
