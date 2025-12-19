# IronWorker Analytics Chat System

A conversational analytics platform for IronWorker union data with intelligent SQL generation, role-based access control, and persistent chat history.

## Features

- 🤖 **AI-Powered SQL Generation**: Natural language to ClickHouse SQL using GPT-4
- 🔒 **Role-Based Access Control (RBAC)**: Admin, Local Union Officer, and Member roles
- 💬 **Persistent Chat History**: PostgreSQL-backed conversation storage
- 📊 **Interactive Visualizations**: Automatic chart generation with Plotly
- 🔍 **Context-Aware Queries**: Follow-up questions understand previous context
- 📈 **Real-time Analytics**: Query 6.6M+ records across 8 tables
- 🎯 **Langfuse Integration**: Complete observability and tracing

## Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│                 │         │                  │         │                 │
│   Streamlit     │────────▶│   FastAPI        │────────▶│   ClickHouse    │
│   Frontend      │         │   Backend        │         │   (Data)        │
│                 │         │                  │         │                 │
└─────────────────┘         └──────────────────┘         └─────────────────┘
                                     │
                                     │
                                     ▼
                            ┌──────────────────┐
                            │                  │
                            │   PostgreSQL     │
                            │   (Chat History) │
                            │                  │
                            └──────────────────┘
```

## Tech Stack

- **Backend**: FastAPI, LangChain, LangGraph
- **Frontend**: Streamlit
- **AI**: OpenAI GPT-4
- **Databases**: ClickHouse (analytics), PostgreSQL (chat storage)
- **Observability**: Langfuse
- **SQL Processing**: SQLGlot

## Prerequisites

- Python 3.10+
- ClickHouse server
- PostgreSQL server
- OpenAI API key

## Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd ironworker-analytics
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.template .env
# Edit .env with your credentials
```

5. **Initialize databases**
```bash
# PostgreSQL chat storage tables are auto-created on first run
# Ensure ClickHouse has the iw_dev database with required tables
```

## Configuration

### Environment Variables

Create a `.env` file based on `.env.template`:

```env
# ClickHouse Configuration
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=
CLICKHOUSE_DATABASE=iw_dev

# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=chat_history
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Langfuse Configuration (optional)
LANGFUSE_HOST=http://localhost:3001
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
```

## Running the Application

### Start Backend Server

```bash
python chat_storage_backend.py
```

Backend will be available at: `http://localhost:8000`

### Start Frontend Interface

```bash
streamlit run newfrontend.py
```

Frontend will be available at: `http://localhost:8501`

## API Endpoints

### Query Endpoint
```http
POST /query
Content-Type: application/json

{
  "question": "How many active members in local 782?",
  "session_id": "optional-uuid"
}
```

### Get User Sessions
```http
GET /sessions?user_id=<user_id>
```

### Get Session History
```http
GET /session/{session_id}
```

### Delete Session
```http
DELETE /session/{session_id}
```

### Health Check
```http
GET /health
```

## Database Schema

### ClickHouse Tables (iw_dev)

| Table | Records | Description |
|-------|---------|-------------|
| `iw_contact08` | 712,972 | Master member registry |
| `member_certifications` | 1,289,032 | Certification tracking |
| `drugtestrecords` | 571,086 | Drug test compliance |
| `member_course_history` | 3,481,401 | Training history |
| `memberunionhistory1` | 524,306 | Status change history |
| `member_national_fund_trainings` | 32,120 | National training programs |
| `member_online_registration_courses` | 3,612 | Online course participation |
| `trainingprogramcertificates` | 433 | Instructor certifications |

### PostgreSQL Tables (chat_history)

- `chat_sessions`: User conversation sessions
- `chat_messages`: Individual messages with metadata

## Role-Based Access Control

### Admin
- Full access to all data
- No automatic filtering applied

### Local Union Officer
- Access to members in their local union
- Queries automatically filtered by `localunionid`

### IronWorker Member
- Access only to their own data
- Queries automatically filtered by `userid`

## Usage Examples

### Basic Queries
```
"How many active members are there?"
"Show me certifications expiring in 2025"
"What's the drug test compliance rate?"
```

### Follow-up Queries (Context-Aware)
```
User: "How many members in local 782?"
Assistant: [Shows results]
User: "What about their certifications?"
Assistant: [Automatically filters for local 782]
```

### Complex Analytics
```
"Compare training completion rates by state"
"Show me members with expired OSHA certifications"
"What's the average course completion time?"
```

## Project Structure

```
ironworker-analytics/
├── chat_storage_backend.py    # FastAPI backend with RBAC
├── newfrontend.py              # Streamlit frontend
├── .env.template               # Environment template
├── .env                        # Your configuration (gitignored)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── clickhouse_iw_config.md     # Database schema documentation
```

## Security Features

- **SQL Injection Prevention**: SQLGlot parsing and validation
- **Sensitive Data Redaction**: SSN, SIN, phone numbers automatically filtered
- **Role-Based Filtering**: Automatic query modification based on user role
- **Predicate Enforcement**: Required WHERE clauses based on role

## Monitoring & Observability

The system includes Langfuse integration for:
- Query tracing
- Performance monitoring
- Error tracking
- Usage analytics

Access Langfuse dashboard at: `http://localhost:3001`

## Development

### Adding New Tables

1. Update `TABLE_DATES` in `chat_storage_backend.py`
2. Add to `LOCAL_TABLE_FIELDS` or `MEMBER_TABLE_FIELDS`
3. Update schema documentation in `clickhouse_iw_config.md`
4. Update SQL generation prompt with new table info

### Testing RBAC

Change the `user_role` in the `/query` endpoint:

```python
user_role = UserRole(
    role="local_union_officer",  # or "admin", "ironworker"
    user_id="test-user-id",
    local_union_id="782"
)
```

## Troubleshooting

### Backend won't start
- Check database connections in `.env`
- Ensure PostgreSQL is running
- Verify ClickHouse is accessible

### Frontend connection error
- Ensure backend is running on port 8000
- Check `BACKEND_URL` in `newfrontend.py`

### SQL errors
- Verify table names match schema
- Check date ranges for temporal queries
- Review ClickHouse logs

### Chat history not loading
- Check PostgreSQL connection
- Verify tables were created (check logs on first run)
- Ensure session_id is valid UUID

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Your License Here]

## Support

For issues and questions, please open a GitHub issue or contact the development team.

## Acknowledgments

- Built for IronWorker Union data analytics
- Powered by OpenAI GPT-4
- LangChain/LangGraph for agent workflows
- Streamlit for rapid UI development
