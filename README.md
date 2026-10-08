# ResearchAgent Backend — Multi-Agent Deep Research Engine

FastAPI backend service powering autonomous multi-agent academic research, literature synthesis, and publication-grade manuscript typesetting.

---

## Architecture & Subsystems

1. **Multi-Agent Orchestration (LangGraph):**
   - **Research Planner:** Generates multi-angle queries, constraints, and target paper structure.
   - **Parallel Retrieval:** Queries ArXiv, Semantic Scholar, CrossRef, OpenAlex, PubMed, and Tavily.
   - **Paper Summarizer:** Extracts methodology, benchmark results, limitations, and key findings.
   - **Debate & Synthesis Loop:** Proposer ↔ Critic adversarial debate to stress-test claims.
   - **Cross-Paper Analysis & Gap Discovery:** Identifies consensus, contradictions, and 5-year roadmaps.
   - **Manuscript Architect:** Drafts 9-section academic publication with formal mathematics and empirical data matrices.
   - **Citation QA:** Validates inline citation markers and generates clean BibTeX bibliographies.

2. **Typesetting & Export Engine:**
   - **Typst 0.15 Compiler:** Produces authentic academic PDFs across 4 standards:
     - `IEEE`: Two-column conference grid with IEEEtran geometry and IEEE CSL references.
     - `ACM`: Two-column Sigconf proceedings with CCS Concepts and ACM CSL references.
     - `APA 7th`: Single-column professional manuscript with running head and Author-Date citations.
     - `MLA 9th`: Single-column double-spaced academic paper with Works Cited.
   - **LaTeX Exporter:** Packages complete `.tex` manuscript source, `.bib` BibTeX databases, and figures into downloadable `.zip` archives.
   - **ReportLab Platypus Fallback:** Graceful fallback renderer ensuring 100% PDF generation uptime.

3. **LLM Engine (Google Gemini Free-Tier):**
   - **Dual-Workhorse Load Balancing:** Dynamically alternates requests between `gemini-3.1-flash-lite` and `gemini-3.5-flash-lite` based on 60-second sliding-window request volume.
   - **Rate-Pacing Protection:** Guarantees traffic never exceeds Google's 15 RPM cap while delivering a combined **30–45 RPM** and **1,000+ requests per day**.
   - **1,000,000+ Token Context:** Zero risk of prompt truncation or context window overflow.

4. **Database & Storage (Supabase):**
   - **PostgreSQL with RLS:** User data isolation on `research_sessions`, `research_jobs`, `document_chunks`, and `llm_usage`.
   - **pgvector:** 384-dimensional vector similarity search via `match_document_chunks` for uploaded PDF papers.
   - **Object Storage:** Private `delve-documents` bucket scoped to user directories.

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
```

Edit `backend/.env`:
```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_PRIMARY_MODEL=gemini-3.1-flash-lite
TAVILY_API_KEY=your_tavily_api_key

SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_STORAGE_BUCKET=delve-documents

WS_TICKET_SECRET=your_32_char_secret
```

### Start Server:
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## API Endpoints

- `POST /research/start` — Start an autonomous research run (topic, depth, paper_format, paper_type, uploaded_paper_ids).
- `GET /research/{session_id}/paper` — Fetch the final manuscript, metadata, and bibliography.
- `GET /research/{session_id}/paper.pdf?format=ieee` — Download compiled publication PDF (supports `ieee`, `acm`, `apa`, `mla`).
- `GET /research/{session_id}/latex?format=ieee` — Download complete LaTeX source, `.bib`, and figures `.zip` archive.
- `GET /research/{session_id}/status` — Poll current session status and active agent node.
- `GET /research/{session_id}/detail` — Comprehensive session breakdown with timeline, token metrics, and gaps.
- `POST /research/{session_id}/cancel` — Cancel an in-progress research session.
- `POST /research/{session_id}/retry` — Retry an interrupted or failed session.
- `GET /research/sessions/list` — List user's historical research sessions.
- `POST /upload/pdf` — Upload custom reference PDF documents to Supabase storage and compute vector embeddings.
- `POST /research/{session_id}/ws-ticket` — Generate short-lived (60s) HMAC ticket for WebSocket streaming.
- `WS /ws/{session_id}?ticket=...` — Real-time deliberation and node progression stream.

---

## Verification & Testing

```bash
# Test all external services (Tavily, Gemini, Supabase)
python scratch/test_all_services.py

# Test compilation of all 4 publication PDF standards
python test_generate_four_papers.py
```

---

## Production Deployment (Render / Docker)
Configured for automated zero-downtime deployment on Render using `render.yaml` and the root `Dockerfile`.
