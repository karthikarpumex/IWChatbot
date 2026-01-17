# IronWorker Union Chatbot

A chatbot I built to help query our union database using natural language. Ask questions in plain English and get SQL queries + charts automatically.

## What it does

- Talk to it like a person, it generates SQL queries for you
- Connects to our ClickHouse database (iw_dev)
- Remembers your conversation so you can ask follow-up questions
- Makes charts automatically when it makes sense
- Everything runs in Docker so setup is pretty straightforward

## Tech Stack

**Backend:** FastAPI + LangChain + OpenAI GPT-4  
**Databases:** ClickHouse (union data) + PostgreSQL (chat history)  
**Frontend:** Streamlit + Plotly  
**Infrastructure:** Docker, Grafana, Loki

---

## Quick Setup

### What You Need

- Docker Desktop running
- An OpenAI API key (get at platform.openai.com)
- ClickHouse server running with iw_dev database

### Installation

```bash
# 1. Clone the repo
git clone <repo-url>
cd IWChatbot

# 2. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and CLICKHOUSE_PASSWORD

# 3. Start everything
docker-compose up -d

# 4. Test it
curl http://localhost:8000/health
```

### First Query

Open http://localhost:8000/docs and try the `/query` endpoint:

```json
{
  "question": "How many active members are there?"
}
```

Or use PowerShell:
```powershell
$body = @{ question = "How many active members?" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/query -Method Post -ContentType "application/json" -Body $body
```

---

## Using the Web Interface

```bash
pip install -r requirements.txt
streamlit run frontend.py
```

Opens at http://localhost:8501

---

## Common Commands

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f iw-backend

# Restart after code changes
docker-compose restart iw-backend

# Rebuild from scratch
docker-compose build --no-cache iw-backend
docker-compose up -d
```

---

## Access Points

- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Grafana**: http://localhost:3002 (admin/admin)
- **Frontend**: http://localhost:8501

---

## Troubleshooting

### Can't connect to ClickHouse

Check if it's running:
```bash
curl http://localhost:8123/ping
```

Make sure `.env` has the right host:
- Windows/Mac: `CLICKHOUSE_HOST=host.docker.internal`
- Linux: `CLICKHOUSE_HOST=172.17.0.1`

### Port conflicts

Edit `docker-compose.yml` and change the port:
```yaml
ports:
  - "8001:8000"  # Use 8001 instead
```

### Containers keep restarting

Check what's wrong:
```bash
docker-compose logs iw-backend
```

Usually means ClickHouse isn't reachable or environment variables are missing.

---

## Environment Variables

Your `.env` file needs:

```bash
# Required
OPENAI_API_KEY=sk-proj-...
CLICKHOUSE_HOST=host.docker.internal
CLICKHOUSE_PASSWORD=your-password
CLICKHOUSE_DATABASE=iw_dev

# Optional (for AI tracing)
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
```

---

## Project Structure

```
backend.py          - Main FastAPI application
frontend.py         - Streamlit web interface
docker-compose.yml  - Docker configuration
requirements.txt    - Python dependencies
.env                - Your configuration (not committed)
.env.example        - Configuration template
```

---

## Notes

- Uses GPT-4 which costs money (few cents per query)
- Chat history is saved so you can resume conversations
- ClickHouse must run separately - this just connects to it
- Langfuse is optional, only for detailed AI call tracing

---

Built this to make querying our union database easier. Questions? Check the logs with `docker-compose logs -f iw-backend`.
