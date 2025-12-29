# JobPilot - AI-Powered Job Application Platform (US + UK)

JobPilot is an intelligent job search and CV optimization platform that uses Claude AI to generate tailored CVs for job applications in both the United States and United Kingdom.

## Version

**v3-combined** - Combined US + UK job search with ATS score iteration

## Features

### Region Toggle (US or UK)
Switch between US and UK job markets with one click.

### US Job Search (5 Sites)
- Indeed (indeed.com)
- LinkedIn
- Glassdoor (glassdoor.com)
- Dice (dice.com)
- ZipRecruiter (ziprecruiter.com)

### UK Job Search (7 Sites)
- Indeed UK (uk.indeed.com)
- LinkedIn
- Glassdoor UK (glassdoor.co.uk)
- Reed (reed.co.uk)
- CV-Library (cv-library.co.uk)
- TotalJobs (totaljobs.com)
- Jobserve (jobserve.com)

### ATS Score Iteration (NEW in V3)
- CV generation iterates until achieving **90% ATS score** or max 3 iterations
- Each iteration incorporates feedback from previous score
- Shows final score and iteration count in UI
- Will not allow download until best possible score is achieved

### Other Features
- **Natural Language Queries**: Search using plain English (e.g., "remote data engineer jobs in Texas" or "remote IR35 data engineer jobs in London")
- **AI-Powered CV Generation**: Uses Claude Opus with the Master CV Optimisation Mega-Prompt v3.0
- **Company Constraint Compliance**: Automatically applies Meta/Amazon/Google vocabulary constraints
- **30-Day Filter**: Only shows jobs posted in the last 30 days
- **Formatted Output**: Downloads CV as formatted TXT matching your input CV style

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         JOBPILOT V3 SYSTEM                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────┐
                    │         USER INTERFACE           │
                    │    (HTML/CSS/JavaScript)         │
                    │  - US/UK Region Toggle           │
                    │  - Search box + NLU parsing      │
                    │  - CV upload (PDF/DOCX/TXT)      │
                    │  - Job cards with actions        │
                    │  - ATS Score display             │
                    │  - Download generated CV         │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI BACKEND                                     │
│                         (app.py)                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  ENDPOINTS:                                                                 │
│  ├─ POST /api/search         → Search jobs (US or UK based on region)       │
│  ├─ POST /api/upload-cv      → Upload and analyze base CV                   │
│  ├─ POST /api/generate-cv    → Generate CV with ATS iteration (target 90%)  │
│  └─ GET  /api/download-cv    → Download formatted CV (TXT)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
┌───────────────┐      ┌───────────────────┐      ┌───────────────────┐
│  NLU PARSER   │      │  JOB SCRAPER      │      │  CV GENERATOR     │
│  (IntentParser)│      │  (BrightData)     │      │  (Claude Opus)    │
├───────────────┤      ├───────────────────┤      ├───────────────────┤
│ Extracts:     │      │ Sources:          │      │ Uses:             │
│ - Job titles  │      │ - Indeed          │      │ - Master Prompt   │
│ - Location    │      │ - LinkedIn        │      │   v3.0            │
│ - Remote/     │      │ - Glassdoor       │      │ - Base CV         │
│   Hybrid      │      │ - Dice (Google)   │      │ - Job Description │
│ - Days filter │      │ - ZipRecruiter    │      │                   │
│               │      │   (Google)        │      │ Applies:          │
│               │      │                   │      │ - Meta constraints│
│               │      │ Features:         │      │ - Amazon rules    │
│               │      │ - 30-day filter   │      │ - Google rules    │
│               │      │ - Remote filter   │      │ - ATS optimization│
│               │      │ - US location     │      │                   │
└───────────────┘      └───────────────────┘      └───────────────────┘
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CV GENERATION FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────┘

User Search Query                    User Uploads CV
      │                                    │
      ▼                                    ▼
┌─────────────┐                    ┌─────────────────┐
│ NLU Parser  │                    │ CV Analyzer     │
│ Extract:    │                    │ Extract:        │
│ - Job title │                    │ - Sections      │
│ - Location  │                    │ - Bullet style  │
│ - Type      │                    │ - Date format   │
└──────┬──────┘                    └────────┬────────┘
       │                                    │
       ▼                                    │
┌─────────────────┐                         │
│ BrightData      │                         │
│ Job Scraper     │                         │
│ - Indeed        │                         │
│ - LinkedIn      │                         │
│ - Glassdoor     │                         │
│ - Dice          │                         │
│ - ZipRecruiter  │                         │
└────────┬────────┘                         │
         │                                  │
         ▼                                  │
┌─────────────────┐                         │
│ Job Results     │                         │
│ (Cached 1hr)    │                         │
└────────┬────────┘                         │
         │                                  │
         │    User clicks "Generate CV"    │
         │              │                   │
         ▼              ▼                   ▼
    ┌─────────────────────────────────────────┐
    │         FETCH JOB DESCRIPTION           │
    │    (Scrape full JD from job URL)        │
    └───────────────────┬─────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │           CLAUDE OPUS API               │
    │                                         │
    │  Input:                                 │
    │  ├─ Master Prompt v3.0                  │
    │  ├─ Base CV (user's CV)                 │
    │  └─ Job Description (scraped)           │
    │                                         │
    │  Process:                               │
    │  ├─ Section 1: JD Analysis              │
    │  ├─ Section 2: CV Analysis              │
    │  ├─ Section 3: Constraint Declaration   │
    │  ├─ Section 4: Reverse Engineering      │
    │  ├─ Section 5-7: Content Generation     │
    │  ├─ Section 8: Compliance Audit         │
    │  ├─ Section 9: ATS Scoring              │
    │  └─ Section 10: Final Output            │
    │                                         │
    │  Output: Tailored CV (Markdown)         │
    └───────────────────┬─────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │         FORMAT AS TXT                   │
    │  - Centered name                        │
    │  - Centered contact                     │
    │  - Section headers with underline       │
    │  - Job entries with dates               │
    │  - Bullet points with •                 │
    └───────────────────┬─────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │         DOWNLOAD CV                     │
    │    (tailored_cv_{job_id}.txt)           │
    └─────────────────────────────────────────┘
```

### Master CV Optimisation Framework v3.0

The CV generation uses a comprehensive 10-section framework:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MASTER CV OPTIMISATION FRAMEWORK v3.0                    │
└─────────────────────────────────────────────────────────────────────────────┘

SECTION 1: Comprehensive JD Analysis
├── 1.1 Target Role Metadata
├── 1.2 Exhaustive Requirement Extraction (all 60+ requirements)
├── 1.3 Dynamic Category Generation (emergent from JD)
├── 1.4 Priority Classification (MUST-HAVE / SHOULD-HAVE / NICE-TO-HAVE)
├── 1.5 Tools, Technologies & Methodologies
└── 1.6 Language & Terminology Patterns

SECTION 2: Base CV Analysis
├── 2.1 Candidate Profile
├── 2.2 Employer Inventory
├── 2.3 Sector Alignment Ranking
└── 2.4 Seniority Mapping

SECTION 3: Constraint Declaration
├── Meta/Facebook constraints (infrastructure abstraction)
├── Amazon/AWS constraints (AWS-only)
├── Google/Alphabet constraints (GCP-only)
└── Standard constraints (no restrictions)

SECTION 4: Reverse Engineering Strategy
├── 4.1 Category-to-Employer Mapping
├── 4.2 Tools/Methodology Distribution
├── 4.3 Sector-Specific Project Contexts
└── 4.4 Coverage Validation

SECTION 5: Experience Section Generation
├── 4-5 bullets per role with constraint compliance
└── POST-GENERATION CHECK for each employer

SECTION 6: Skills Section Generation
SECTION 7: Remaining Sections (Education, Certifications, Publications)
SECTION 8: Compliance Audit
SECTION 9: ATS Relevance Simulation (target: 90%+)
SECTION 10: Final CV Output
```

### Company Constraint System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ META / FACEBOOK                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ PROHIBITED: AWS, GCP, Azure, Snowflake, Databricks, Kubernetes, Docker,    │
│             Terraform, DBT, Airflow, Prometheus, Grafana, Jenkins, Kafka,   │
│             PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch (all by name)  │
│                                                                             │
│ USE INSTEAD:                                                                │
│   "internal data infrastructure"      instead of Snowflake/BigQuery        │
│   "proprietary orchestration systems" instead of Kubernetes/Airflow        │
│   "internal monitoring infrastructure" instead of Prometheus/Grafana       │
│   "internal deployment systems"       instead of Jenkins/GitLab CI         │
│   "internal database systems"         instead of PostgreSQL/MySQL          │
│   "proprietary event streaming"       instead of Kafka                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ AMAZON / AWS ECOSYSTEM                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ PROHIBITED: GCP, Azure, Snowflake (competing platforms)                     │
│ USE: AWS services (S3, Redshift, Glue, Lambda, EMR, Aurora, etc.)           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ GOOGLE / ALPHABET                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ PROHIBITED: AWS, Azure, Snowflake (competing platforms)                     │
│ USE: GCP services (BigQuery, Dataflow, etc.) or "internal" terminology      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STANDARD (all other companies)                                              │
├─────────────────────────────────────────────────────────────────────────────┘
│ No restrictions - any appropriate tools/technologies may be used            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
jobpilot/
├── app_v3.py              # Main FastAPI application
├── static/
│   └── index_v3.html      # Web UI
├── data/                  # Runtime data (gitignored)
│   ├── cache/             # Search result cache
│   └── cv_*.txt           # Generated CVs
├── agents/                # AI agent modules
│   ├── cv_architect/      # CV generation logic
│   ├── discovery/         # Job discovery
│   ├── form_filler/       # Application form filling
│   └── vault/             # Credential storage
├── core/
│   ├── config.py          # Configuration
│   ├── llm_client.py      # LLM client wrapper
│   └── schemas.py         # Pydantic schemas
├── services/
│   ├── orchestrator.py    # Workflow orchestration
│   └── notification.py    # Notifications
└── requirements.txt       # Python dependencies
```

---

## Installation

### Prerequisites

- Python 3.9+
- Anthropic API key (Claude)
- Bright Data API key (for job scraping)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/jobpilot.git
   cd jobpilot
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys:
   # ANTHROPIC_API_KEY=sk-ant-...
   # BRIGHT_DATA_API_KEY=...
   # BRIGHT_DATA_ZONE=mcp_unlocker
   ```

5. **Run the server**
   ```bash
   cd jobpilot
   uvicorn app_v3:app --reload --port 8000
   ```

6. **Open in browser**
   ```
   http://localhost:8000
   ```

---

## Usage

### 1. Search for Jobs

Enter a natural language query:
- "remote data engineer jobs in Texas"
- "hybrid product manager jobs in California"
- "senior software engineer jobs in New York"

### 2. Upload Your CV

Click "Upload CV" and select your base CV file (PDF, DOCX, or TXT).

### 3. Generate Tailored CV

Click "Generate CV" on any job card. The system will:
1. Fetch the full job description
2. Send your CV + JD to Claude Opus
3. Execute the Master Prompt v3.0 framework
4. Generate a tailored CV with proper constraints

### 4. Download CV

Click "Download CV" to save the formatted TXT file.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve web UI |
| POST | `/api/search` | Search jobs |
| POST | `/api/upload-cv` | Upload base CV |
| POST | `/api/generate-cv` | Generate tailored CV |
| GET | `/api/download-cv/{user_id}/{job_id}` | Download CV |

### Example: Search Jobs

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "remote data engineer jobs in Texas", "user_id": "user123", "page": 1}'
```

### Example: Generate CV

```bash
curl -X POST http://localhost:8000/api/generate-cv \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "job_id": "linkedin_1"}'
```

---

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Claude API key | Yes |
| `BRIGHT_DATA_API_KEY` | Bright Data API key | Yes |
| `BRIGHT_DATA_ZONE` | Bright Data zone (default: mcp_unlocker) | Yes |

### Search Filters

| Filter | Default | Description |
|--------|---------|-------------|
| `days_ago` | 30 | Only jobs posted within X days |
| `location_type` | any | remote, hybrid, onsite, any |

---

## Tech Stack

- **Backend**: FastAPI, Python 3.9+
- **AI**: Claude Opus (Anthropic)
- **Scraping**: Bright Data MCP, BeautifulSoup
- **Frontend**: HTML, CSS, JavaScript

---

## License

MIT License - see LICENSE file for details.

