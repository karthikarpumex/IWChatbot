
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from typing import TypedDict, Literal, Optional, Any, Sequence
import uvicorn 
import clickhouse_connect
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from decimal import Decimal
from fastapi.middleware.cors import CORSMiddleware

# Disable OpenTelemetry auto-instrumentation
#os.environ["OTEL_SDK_DISABLED"] = "true"

# Import langfuse AFTER langchain (langfuse requires langchain to be available)
# Make it optional since it's not critical for core functionality
try:
    from langfuse.callback import CallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    CallbackHandler = None
    print("WARNING: langfuse_langchain not available - monitoring disabled")
import sqlglot
from sqlglot import exp, parse_one, errors
import uuid
import json
import time
import math
import threading
from functools import wraps
import logging
from decimal import Decimal

# Optional logging_loki import
try:
    import logging_loki
    LOKI_AVAILABLE = True
except ImportError:
    LOKI_AVAILABLE = False
    # Will log warning after logger is setup

load_dotenv()

# ============== LOKI LOGGING SETUP ==============

def setup_loki_logging():
    """Setup Loki logging for application logs"""
    handlers = [logging.StreamHandler()]
    loki_enabled = False
    loki_error = None
    
    if LOKI_AVAILABLE and os.getenv("ENABLE_LOKI_LOGGING", "true").lower() == "true":
        try:
            loki_url = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")
            loki_handler = logging_loki.LokiHandler(
                url=loki_url,
                tags={"application": "iwchatbot", "environment": os.getenv("ENVIRONMENT", "dev")},
                version="1",
            )
            handlers.append(loki_handler)
            loki_enabled = True
        except Exception as e:
            loki_error = str(e)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    log = logging.getLogger("iwchatbot")
    
    # Log setup status after logger is configured
    if not LOKI_AVAILABLE:
        log.warning("logging_loki module not available - Loki logging disabled")
    elif loki_enabled:
        log.info("Loki logging enabled successfully")
    elif loki_error:
        log.warning(f"Loki logging disabled due to error: {loki_error}")
    else:
        log.info("Loki logging disabled (not configured)")
    
    return log

logger = setup_loki_logging()

app = FastAPI()

# ============== PERFORMANCE OPTIMIZATIONS ==============

pg_pool = None
clickhouse_client = None

def init_connection_pools():
    """Initialize database connection pools"""
    global pg_pool, clickhouse_client

    pg_pool = pool.ThreadedConnectionPool(
        minconn=2,
        maxconn=10,
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "chat_history"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres")
    )
    
    clickhouse_client = clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST", "localhost"),
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        username=os.getenv("CLICKHOUSE_USER", "default"),
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
        database=os.getenv("CLICKHOUSE_DATABASE", "iw_dev")
    )

    logger.info("✅ Connection pools initialized")


# ============== LANGFUSE SETUP ==============

# Initialize handler (will be None if config missing or package not available)
if LANGFUSE_AVAILABLE:
    try:
        langfuse_handler = CallbackHandler()
        logger.info("Langfuse callback handler initialized successfully")
    except Exception as e:
        langfuse_handler = None
        logger.warning(f"Failed to initialize Langfuse handler: {e}")
else:
    langfuse_handler = None
    logger.info("Langfuse not available - monitoring disabled")
# ============== MIDDLEWARE ==============

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id
    response = await call_next(request)
    response.headers["X-Trace-ID"] = trace_id
    return response

# ============== NODE LOGGING DECORATOR ==============

def log_node(node_name: str):
    """Decorator to log individual LangGraph nodes"""
    def decorator(func):
        @wraps(func)
        def wrapper(state: dict, **kwargs):
            trace_id = state.get('trace_id')
            logger.info(f"Node started: {node_name}", extra={
                "node": node_name, "trace_id": trace_id, "event": "node_start"
            })
            start_time = time.time()
            try:
                result = func(state, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"Node completed: {node_name}", extra={
                    "node": node_name, "trace_id": trace_id, 
                    "execution_time": execution_time, "event": "node_success"
                })
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Node failed: {node_name}", extra={
                    "node": node_name, "trace_id": trace_id, 
                    "execution_time": execution_time, "error": str(e), "event": "node_error"
                })
                raise
        return wrapper
    return decorator

# ============== DATABASE FUNCTIONS ==============

def get_clickhouse_db():
    global clickhouse_client
    if clickhouse_client is None:
        init_connection_pools()
    return clickhouse_client

def get_postgres_conn():
    global pg_pool
    if pg_pool is None:
        init_connection_pools()
    return pg_pool.getconn()

def release_postgres_conn(conn):
    if pg_pool:
        pg_pool.putconn(conn)

# ============== POSTGRESQL PERMANENT STORAGE ==============

def init_conversation_tables():
    """Initialize PostgreSQL tables for conversation persistence"""
    conn = get_postgres_conn()
    try:
        cur = conn.cursor()
        
        # Migration: Check if message_id is integer and needs to be UUID
        cur.execute("""
            SELECT data_type 
            FROM information_schema.columns 
            WHERE table_name = 'chat_messages' AND column_name = 'message_id'
        """)
        res = cur.fetchone()
        if res and res[0] == 'integer':
            logger.info("Migrating chat_messages.message_id from integer to UUID. Dropping old table...")
            cur.execute("DROP TABLE IF EXISTS chat_messages CASCADE")
            # Also drop sessions to ensure clean state if ID types changed
            cur.execute("DROP TABLE IF EXISTS chat_sessions CASCADE")
            conn.commit()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id UUID PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                session_status VARCHAR(20) NOT NULL,
                created_at TIMESTAMP DEFAULT (NOW() AT TIME ZONE 'utc'),
                last_message_at TIMESTAMP DEFAULT (NOW() AT TIME ZONE 'utc')
            )
        """)
        
        # Create chat_messages table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id UUID PRIMARY KEY,
                session_id UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                sql_query TEXT,
                filtered_sql TEXT,
                chart_data JSONB,
                result_data JSONB,
                execution_time FLOAT,
                error TEXT,
                langfuse_trace_id VARCHAR(255),
                input_tokens INTEGER,
                output_tokens INTEGER,
                created_at TIMESTAMP DEFAULT (NOW() AT TIME ZONE 'utc'),
                updated_at TIMESTAMP DEFAULT (NOW() AT TIME ZONE 'utc')
            )
        """)
        
        # Create indexes for optimal performance
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_time 
            ON chat_messages(session_id, created_at)
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_messages_langfuse_trace 
            ON chat_messages(langfuse_trace_id)
        """)
        
        conn.commit()
        cur.close()
        logger.info("Conversation tables initialized with token tracking")
        
    except Exception as e:
        logger.error(f"Failed to initialize chat tables: {e}")
    finally:
        release_postgres_conn(conn)

def save_to_permanent_storage(
    session_id: str, 
    message_type: str, 
    content: str,
    user_id: str,
    session_status: str,
    sql_query: str = None,
    filtered_sql: str = None,
    chart_data: dict = None,
    result_data: list = None,
    execution_time: float = None,
    error: str = None,
    langfuse_trace_id: str = None,
    input_tokens: int = None,
    output_tokens: int = None
):
    """Save message to PostgreSQL for permanent audit trail with rich metadata and token usage"""
    try:
        conn = get_postgres_conn()
        cur = conn.cursor()
        
        # Ensure session exists
        cur.execute("""
            INSERT INTO chat_sessions (session_id, user_id, last_message_at,session_status) 
            VALUES (%s, %s, TIMEZONE('utc', CURRENT_TIMESTAMP), %s)
            ON CONFLICT (session_id) DO UPDATE SET 
                last_message_at = TIMEZONE('utc', CURRENT_TIMESTAMP),
                session_status = %s
        """, (session_id, user_id, session_status, session_status))
        
        # Convert role from old format to new format
        role = 'user' if message_type == 'human' else 'assistant' if message_type == 'ai' else message_type
        
        # Convert chart data to JSON string - save STRUCTURE + STATS (no raw data)
        if chart_data:
            # Extract series config from chart and data
            series_config = extract_series_config(chart_data, result_data) if result_data else None
            
            # Save only chart structure + metrics (no heavy x_values, y_values, series_data)
            chart_structure = {
                'chart_type': chart_data.get('chart_type'),
                'title': chart_data.get('title'),
                'x_label': chart_data.get('x_label'),
                'y_label': chart_data.get('y_label'),
                'series_config': series_config
            }
            chart_json = json.dumps(chart_structure)
        else:
            chart_json = None
        
        # DON'T save result_data (too large, not needed for regeneration)
        result_json = None
        
        message_id = str(uuid.uuid4())
        cur.execute("""
            INSERT INTO chat_messages 
            (message_id, session_id, role, content, sql_query, filtered_sql, chart_data, result_data,
             execution_time, error, langfuse_trace_id, input_tokens, output_tokens) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (message_id, session_id, role, content, sql_query, filtered_sql, chart_json, result_json,
              execution_time, error, langfuse_trace_id, input_tokens, output_tokens))
        conn.commit()
        cur.close()
        logger.debug(f"Saved {role} message ID={message_id} with tokens=({input_tokens},{output_tokens}) trace={langfuse_trace_id} for session {session_id}")
        return message_id
        
    except Exception as e:
        logger.error(f"Error in save_to_permanent_storage for session {session_id}: {e}", extra={
            "session_id": session_id,
            "message_type": message_type,
            "langfuse_trace_id": langfuse_trace_id,
            "error": str(e),
            "method": "save_to_permanent_storage"
        })
        raise  # Re-raise to bubble up
    finally:
        if 'conn' in locals():
            release_postgres_conn(conn)

def load_from_permanent_storage(session_id: str) -> list[BaseMessage]:
    """Load conversation history from PostgreSQL with rich context for AI messages"""
    try:
        conn = get_postgres_conn()
        cur = conn.cursor()
        
        # Load content plus SQL/data for context reconstruction
        cur.execute("""
            SELECT cm.role, cm.content, cm.sql_query, cm.result_data
            FROM chat_messages cm
            JOIN chat_sessions cs ON cm.session_id = cs.session_id
            WHERE cm.session_id = %s 
            ORDER BY cm.created_at ASC
        """, (session_id,))
        
        db_messages = cur.fetchall()
        cur.close()
        
        messages = []
        for row in db_messages:
            role, content, sql_query, result_data = row
            
            if role == 'user':
                messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                # Reconstruct rich context for AI messages
                rich_content = content
                if sql_query:
                    rich_content += f"\n\n[SQL executed: {sql_query}]"
                if result_data:
                    # result_data is stored as JSONB, parse if string
                    data = result_data if isinstance(result_data, list) else json.loads(result_data) if result_data else []
                    if len(data) <= 10:
                        rich_content += f"\n[Data returned: {data}]"
                    elif data:
                        rich_content += f"\n[Data sample (first 3 of {len(data)}): {data[:3]}]"
                messages.append(AIMessage(content=rich_content))
            elif role == 'system':
                messages.append(SystemMessage(content=content))
        
        logger.info(f"Loaded {len(messages)} messages from DB for session {session_id}")
        return messages
        
    except Exception as e:
        logger.warning(f"Failed to load session from DB: {e}")
        raise
    finally:
        if 'conn' in locals():
            release_postgres_conn(conn)

# ============== CONVERSATION SUMMARY BUFFER MEMORY ==============

class ConversationSummaryBufferMessageHistory(BaseChatMessageHistory, BaseModel):
    """
    Smart memory implementation that:
    - Keeps last k messages in full detail
    - Summarizes older messages into SystemMessage  
    - Automatically saves to PostgreSQL
    - Reconstructs from PostgreSQL on init
    """
    messages: list[BaseMessage] = Field(default_factory=list)
    llm: ChatOpenAI = Field(default_factory=lambda: ChatOpenAI(model="gpt-4o", temperature=0))
    k: int = Field(default=10)
    session_id: str = Field(default="")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, session_id: str, llm: ChatOpenAI = None, k: int = 10):
        if llm is None:
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        super().__init__(llm=llm, k=k, session_id=session_id)
        
        # Load from permanent storage on init for recovery
        logger.info(f"Loading conversation history from database for session: {session_id}")
        stored_messages = load_from_permanent_storage(session_id)
        if stored_messages:
            logger.info(f"Found {len(stored_messages)} messages in database - reconstructing memory")
            # Reconstruct by adding messages (triggers summarization if needed)
            for msg in stored_messages:
                # Don't save again to DB during reconstruction
                self._add_message_to_memory_only(msg)
            logger.info(f"Reconstruction complete: {len(self.messages)} messages in memory (includes any summarization)")
        else:
            logger.info(f"No conversation history found in database for session: {session_id} - starting fresh")
    
    def _add_message_to_memory_only(self, message: BaseMessage) -> None:
        """Add to memory without saving to DB (for reconstruction)"""
        existing_summary = None
        if len(self.messages) > 0 and isinstance(self.messages[0], SystemMessage):
            existing_summary = self.messages.pop(0).content
        
        self.messages.append(message)
        
        if len(self.messages) > self.k:
            logger.debug(f"Memory limit exceeded during reconstruction, summarizing")
            messages_to_summarize = self.messages[:-self.k]
            self.messages = self.messages[-self.k:]
            new_summary = self._create_summary(existing_summary, messages_to_summarize)
            self.messages = [SystemMessage(content=new_summary)] + self.messages
    
    def add_message(self, message: BaseMessage) -> None:
        """
        Add message with automatic summarization and PostgreSQL persistence.
        Called automatically when new messages are added to conversation.
        """
        # Save to permanent storage immediately
        message_type = 'human' if isinstance(message, HumanMessage) else 'ai' if isinstance(message, AIMessage) else 'system'
        save_to_permanent_storage(self.session_id, message_type, message.content)
        
        # Check if we have existing summary
        existing_summary = None
        if len(self.messages) > 0 and isinstance(self.messages[0], SystemMessage):
            logger.debug(">> Found existing summary")
            existing_summary = self.messages.pop(0).content
        
        # Add new message
        self.messages.append(message)
        
        # Check if we exceed limit
        if len(self.messages) > self.k:
            logger.info(f">> Found {len(self.messages)} messages, dropping {len(self.messages) - self.k} oldest messages")
            
            # Get messages to summarize
            messages_to_summarize = self.messages[:-self.k]
            
            # Keep only recent k messages
            self.messages = self.messages[-self.k:]
            
            # Create or update summary
            new_summary = self._create_summary(existing_summary, messages_to_summarize)
            
            # Prepend summary
            self.messages = [SystemMessage(content=new_summary)] + self.messages
            
            logger.info(f">> New summary created, now have 1 summary + {len(self.messages)-1} recent messages")
    
    def _create_summary(self, existing_summary: Optional[str], old_messages: list[BaseMessage]) -> str:
        """Create or update conversation summary using LLM"""
        
        # Format messages for summarization
        old_messages_formatted = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in old_messages
        ])
        
        # Create summary prompt
        if existing_summary:
            summary_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "Given the existing conversation summary and the new messages, "
                    "generate a new summary of the conversation. Ensuring to maintain "
                    "as much relevant information as possible about IronWorker database queries, "
                    "SQL analysis, and data insights."
                ),
                HumanMessagePromptTemplate.from_template(
                    "Existing conversation summary:\n{existing_summary}\n\n"
                    "New messages:\n{old_messages}"
                )
            ])
        else:
            summary_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "Summarize the following conversation between a user and an IronWorker database assistant. "
                    "Focus on: questions asked, SQL queries generated, data insights, and ongoing analysis."
                ),
                HumanMessagePromptTemplate.from_template(
                    "Messages:\n{old_messages}"
                )
            ])
        
        try:
            # Format and invoke LLM
            formatted_messages = summary_prompt.format_messages(
                existing_summary=existing_summary if existing_summary else "",
                old_messages=old_messages_formatted
            )
            
            new_summary = self.llm.invoke(formatted_messages)
            logger.debug(f">> New summary: {new_summary.content[:100]}...")
            return new_summary.content
            
        except Exception as e:
            logger.error(f"Failed to create summary: {e}")
            # Fallback to simple text truncation
            return f"Previous conversation: {old_messages_formatted[:500]}..."
    
    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add multiple messages"""
        for message in messages:
            self.add_message(message)
    
    def clear(self) -> None:
        """Clear the message history"""
        self.messages = []
        logger.info(f"Cleared memory for session {self.session_id}")

# ============== CONVERSATION HISTORY STORE ==============

# In-memory store for active conversation memories (thread-safe)
conversation_memories: dict[str, ConversationSummaryBufferMessageHistory] = {}
_memories_lock = threading.Lock()

def get_conversation_memory(session_id: str, k: int = 10) -> ConversationSummaryBufferMessageHistory:
    """
    Get or create conversation memory for a session.
    This integrates with your existing workflow.
    Thread-safe implementation using lock.
    """
    with _memories_lock:
        if session_id not in conversation_memories:
            logger.info(f"Session {session_id} not in active memory - creating new ConversationSummaryBufferMemory")
            conversation_memories[session_id] = ConversationSummaryBufferMessageHistory(
                session_id=session_id,
                llm=ChatOpenAI(model="gpt-4o", temperature=0),
                k=k
            )
            # Log what was actually loaded (constructor handles DB loading internally)
            loaded_memory = conversation_memories[session_id]
            logger.info(f"Memory initialized for {session_id}: {len(loaded_memory.messages)} messages total (loaded from DB + reconstructed)")
        else:
            logger.info(f"Using existing conversation memory for session: {session_id} ({len(conversation_memories[session_id].messages)} messages)")
        return conversation_memories[session_id]

# ============== SYSTEM PROMPTS ==============

def get_sql_system_prompt() -> str:
    """Generate SQL system prompt with current date context"""
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    return f"""You are an expert ClickHouse SQL query generator for the Ironworkers Union database.

Today's date is {current_date}.

DATE AND YEAR HANDLING:
======================
- If user mentions a specific year (e.g., "in 2025", "for 2024"), use that year directly as integer
- Only use today() or toYear(today()) when NO specific year is mentioned
- For age calculations with specific year: use the year as integer, e.g., "2025 - toYear(dateofbirth)" NOT "toYear('2025-01-01')"
- toYear() requires Date type, NOT string. Wrong: toYear('2025-01-01'). Right: 2025 or toYear(toDate('2025-01-01'))

CRITICAL RULES - READ FIRST:
============================
1. **ONLY USE TABLES LISTED BELOW** - These are the ONLY 9 tables that exist. Do NOT invent or assume any other tables.
2. If asked about data not available in the schema (e.g., projects), set cannot_answer=true and provide a helpful refusal_message.
3. For ANY questions about people, members, individuals, or person information → use iw_dev.iw_contact08
4. NEVER generate SQL for tables not explicitly listed in the schema below.

LATEST ACTIVE USER ID RULE:
==========================
- When querying iw_contact08, always select records with the latest active user ID (active_latest_userid) and ensure the user IDs are distinct, unless the user requests otherwise.

LOCAL UNION CONTEXT:
===================
- Total local unions: 225 (ranging from '1' to '999')
- localunionid is stored as STRING - always use quotes: localunionid = '433' NOT localunionid = 433
- Example locals: '1', '3', '25', '63', '433', '782', '999'

DATABASE SCHEMA:
================

TABLE: iw_dev.iw_contact08 (Master member table - PRIMARY - Use for ALL people/member queries)
Columns:
- userid (String): Unique member identifier
- firstname (String): Member first name
- lastname (String): Member last name
- middlename (String): Member middle name
- membernumber (String): Official member number
- memberstatusname (String): Current status
  VALUES: 'Active', 'Suspended', 'Inactive', 'Deceased', 'Revoked', 'Withdrawal Card Issued', 'Terminated', 'Transferred Out', 'Canceled Initiation', 'Pending Initiation', 'Pending Reinstatement', 'Canceled Reinstatement', 'Expelled', 'Forfeit'
- localunionid (String): Local union identifier (e.g., '782')
- statename (String): State name
- city (String): City name
- countryname (String): Country name
- address1 (String): Street address line 1
- address2 (String): Street address line 2
- postalcode (String): Postal/ZIP code
- email (String): Email address
- dateofbirth (String): Date of birth in 'YYYY-MM-DD' format
- classid (String): Member's classification 
  VALUES: 'Journeyman', 'Apprentice', 'PENSIONER', 'PROBATIONARY', 'Trainee', 'Honorary', 'Lifetime Member', 'Military', 'UNKNOWN', 'LIFETIME SHOP'
- skillid (String): Member skillset
  VALUES: 'Ironworker', 'Shopman', 'Rodman', 'Structural Ironworker', '"A" Rodman', 'Welder', 'Finisher', 'Rigger,Machinery Mover & Erector', 'Fence Erector', 'Navy Yard Rigger', 'Sheeter', 'UNKNOWN', 'Pre Apprentice', 'Trainee'
- sex (String): Gender ('M', 'F')
- ethinicity (String): Ethnicity
- membertypecode (String): Member type code
- paidthru (DateTime64): Dues paid through date - use toDate(paidthru) for comparisons
- paid_thru_text (String): Paid through as text
- createddate (DateTime64): Member record creation date
- lastupdateddate (DateTime64): Last update timestamp

TABLE: iw_dev.drugtestrecords (Drug testing records)
Columns:
- active_latest_userid (String): Links to iw_contact08.userid
- member_number__c (String): Member number for direct lookup
- name (String): Record name
- test_status__c (String): Test result status code
  VALUES: 'C' (Completed/Negative), 'X' (Need to Test), 'I' (Ineligible)
- drug_test_completion_date__c (String): Test date in 'YYYY-MM-DD' format
- drug_retest_date__c (String): Scheduled retest date
- local_union_number__c (String): Local union identifier

DRUG TESTING STATUS INTERPRETATION (CRITICAL):
==============================================
The database tracks drug test COMPLIANCE, not pass/fail results.
There is NO positive/failed result in the database - only compliance status.

Status codes:
- 'C' = Completed (Negative result - member took test and passed)
- 'X' = Need to Test (member hasn't taken required test)
- 'I' = Ineligible (member needs to retest)

When user says "failed drug test":
- This means NON-COMPLIANT status: test_status__c IN ('X', 'I')
- It does NOT mean a positive drug result (that data doesn't exist)

When user says "in [year]" with drug tests:
- Filter by: toYear(toDate(drug_test_completion_date__c)) = [year]
- Do NOT add expiration logic when year is specified

Correct pattern for "failed drug test in 2024":
  WHERE test_status__c IN ('X', 'I')
    AND toYear(toDate(drug_test_completion_date__c)) = 2024

WRONG pattern (never do this):
  WHERE (test_status__c IN ('X','I') OR toDate(...) < toDate('2024-01-01'))
    AND toYear(...) = 2024  -- CONFLICT!

TABLE: iw_dev.member_course_history (Training course history)
Columns:
- active_latest_userid (String): Links to iw_contact08.userid
- membernumber (String): Member number
- coursename (String): Name of the course
  POPULAR: 'OSHA Subpart R', 'Aerial and Scissor Lift (MEWP) Training', 'Fall Protection for Construction', 'Orientation for Ironworkers', 'Rigging for Ironworkers 1', 'Rigging for Ironworkers 2', 'Scaffold Erector & Dismantler', 'First Aid Training', 'CPR and AED Training', 'Forklift Operator Hazard Training'
- passed (String): Whether member passed ('Y', 'N', 'Yes', 'No', or NULL)
- startdate (String): Course start date 'YYYY-MM-DD'
- enddate (String): Course end date 'YYYY-MM-DD'
- year (String): Year of the course as string
- hours (String): Training hours
- grade (String): Grade received
- instructor (String): Instructor name
- location (String): Training location
- semester (String): Semester

TABLE: iw_dev.member_certifications (Member certifications)
Columns:
- active_latest_userid (String): Links to iw_contact08.userid
- membernumber (String): Member number
- certification_name__c (String): Name of certification
  POPULAR: 'Ironworker's National WCP', 'Qualified Rigger', 'Crane Signaling Hand and Voice', 'OSHA 10 hour', 'OSHA 30 hour', 'OSHA Subpart R', 'Scaffold Erector/Dismantler', 'Aerial and Scissor lift (MEWP) Training', 'Fall Protection', 'First Aid', 'Forklift Safety'
- code__c (String): Certification code
- create_date__c (String): Certification date 'YYYY-MM-DD'
- expire_date__c (String): Expiration date 'YYYY-MM-DD'
- jatc__c (String): JATC identifier
- source__c (String): Certification source
- definition__c (String): Certification definition

TABLE: iw_dev.memberunionhistory1 (Union membership history)
Columns:
- active_latest_userid (String): Links to iw_contact08.userid
- userid (String): Member user ID
- membernumber (String): Member number
- localunionid (String): Local union identifier
- homelocal (String): Home local union
- memberstatusname (String): Member status at that time
- membertypecode (String): Member type code
- skillname (String): Skill name at that time
- classname (String): Class name at that time
- paidthru (DateTime64): Paid through date - use toDate(paidthru) for comparisons
- paid_thru_text (String): Paid through as text

TABLE: iw_dev.member_national_fund_trainings (National fund training records)
Columns:
- active_latest_userid (String): Links to iw_contact08.userid
- membernumber (String): Member number
- coursecode (String): Course code
- classname (String): Class name
- startdate (String): Start date 'YYYY-MM-DD'
- enddate (String): End date 'YYYY-MM-DD'
- grade (String): Grade received
- contacthours (String): Contact hours
- instructor (String): Instructor name
- location (String): Training location

TABLE: iw_dev.trainingprogramcertificates (Advanced training certificates)
Columns:
- active_latest_userid (String): Links to iw_contact08.userid
- membernumber (String): Member number
- training_cert_name__c (String): Certificate name
- code__c (String): Certificate code
- cert_date__c (String): Certificate date 'YYYY-MM-DD'
- expire_date__c (String): Expiration date 'YYYY-MM-DD'
- jatc__c (String): JATC identifier
- source__c (String): Certificate source

TABLE: iw_dev.member_online_registration_courses (Online course registrations)
Columns:
- active_latest_userid (String): Links to iw_contact08.userid
- membernumber (String): Member number
- registerdate (String): Registration date 'YYYY-MM-DD'
- confirmdate (String): Confirmation date 'YYYY-MM-DD'
- grade (String): Grade received
- hours (String): Course hours
- passed (String): Whether passed
- year (String): Year of registration

TABLE: iw_dev.employmenthistory (Employement and contract history)
Columns:
- active_latest_userid (String): Links to iw_contact08.userid
- userid (String):  userid
- month (String): Month of employment
- year (String): Year of employment
- hours (Decimal): Hours worked
- companyname (String): Company name or contractor name or employer name
- localunion (String): Local union identifier


BUSINESS LOGIC HELPERS:
======================
DUES STATUS:
- Current: toDate(paidthru) >= today()
- Delinquent: toDate(paidthru) < today()
- Example: CASE WHEN toDate(paidthru) >= today() THEN 'Current' ELSE 'Delinquent' END AS dues_status

MEMBER TENURE (years of membership):
- toYear(today()) - toYear(createddate) AS years_of_membership

ACTIVE MEMBERS (business definition):
- memberstatusname = 'Active'

DRUG TEST COMPLIANCE RATE:
- (COUNT(CASE WHEN compliant THEN 1 END) * 100.0 / COUNT(*)) AS compliance_rate

CERTIFICATION STATUS:
- Valid: expire_date__c >= toString(today()) OR expire_date__c IS NULL OR expire_date__c = ''
- Expired: expire_date__c < toString(today()) AND expire_date__c != ''

SQL GENERATION RULES:
====================
1. **CRITICAL**: Always prefix table names with database: iw_dev.table_name (e.g., iw_dev.iw_contact08)
2. **CRITICAL**: ONLY use columns explicitly listed in the schema above - NEVER assume columns exist
3. **CRITICAL**: ONLY use the 9 tables listed above. NO OTHER TABLES EXIST (no  projects, etc.)
4. **CRITICAL**: Always use iw_contact08 as the primary table when member/people info is needed
5. String columns require single quotes: localunionid = '782' NOT localunionid = 782
6. For DateTime64 columns (paidthru, createddate, lastupdateddate), use toDate() for date comparisons
7. For JOINs, always use: iw_dev.iw_contact08.userid = iw_dev.other_table.active_latest_userid
8. Use ClickHouse SQL syntax (e.g., toString(), toDate(), formatDateTime())
9. When filtering by local union, use the exact string value provided
10. For counting, use COUNT(*) or COUNT(DISTINCT column)
11. For aggregations by category, always include GROUP BY
12. Inherit any filters mentioned in conversation history
13. For follow-up questions about "his/her/their status", use the userid/active_latest_userid from the previous query result
14. Limit results to 1000 rows unless user specifies otherwise
15. Use LIKE with '%pattern%' for partial string matching (case-insensitive: use lower())
16. Always alias aggregated columns (e.g., COUNT(*) AS total_count)
17. Use ORDER BY for sorted results, specify ASC or DESC explicitly
18. Use DISTINCT to remove duplicate rows when needed
19. When joining multiple tables, specify table aliases for clarity
20. For year comparisons in member_course_history, cast year column: toInt32(year) >= 2024 or year >= '2024'
21. Always use proper type casting for numeric comparisons: toInt32(), toFloat64(), toString()

NULL AND ERROR HANDLING:
=======================
- Always use COALESCE(column, 'Unknown') for display fields that might be NULL
- Use isNotNull(column) in WHERE when filtering out NULLs
- Empty strings are different from NULL: column != '' AND column IS NOT NULL
- For optional date fields, check: column IS NOT NULL AND column != ''
- Use ifNull(column, default_value) for calculations with potentially NULL values

PERFORMANCE TIPS:
================
- For large aggregations, avoid SELECT * with JOIN - select only needed columns
- Use LIMIT in subqueries when possible
- For DateTime64 columns, use toDate(column) for date-only comparisons
- Prefer COUNT(*) over COUNT(column) when counting rows
- Use PREWHERE for simple filter conditions on large tables

COMMON QUERY PATTERNS:
=====================
- Count members by status: SELECT memberstatusname, COUNT(*) AS count FROM iw_dev.iw_contact08 GROUP BY memberstatusname ORDER BY count DESC
- Find members in a local union: SELECT firstname, lastname, membernumber, memberstatusname FROM iw_dev.iw_contact08 WHERE localunionid = '782' LIMIT 1000
- Join with certifications: SELECT c.firstname, c.lastname, cert.certification_name__c FROM iw_dev.iw_contact08 c JOIN iw_dev.member_certifications cert ON c.userid = cert.active_latest_userid
- Filter by date range: WHERE create_date__c >= '2024-01-01' AND create_date__c <= '2024-12-31'
- Search by name: WHERE lower(firstname) LIKE '%john%' OR lower(lastname) LIKE '%smith%'
- Dues status check: SELECT firstname, lastname, CASE WHEN toDate(paidthru) >= today() THEN 'Current' ELSE 'Delinquent' END AS dues_status FROM iw_dev.iw_contact08 WHERE localunionid = '782'
- Drug testing compliance: SELECT CASE WHEN test_status__c = 'C' AND toDate(drug_test_completion_date__c) >= today() - INTERVAL 1 YEAR THEN 'Negative/Current' WHEN test_status__c = 'C' THEN 'Negative/Expired' WHEN test_status__c = 'X' THEN 'Need to Test' WHEN test_status__c = 'I' THEN 'Ineligible' ELSE COALESCE(test_status__c, 'Unknown') END AS status, COUNT(*) AS count FROM iw_dev.drugtestrecords WHERE drug_test_completion_date__c IS NOT NULL AND drug_test_completion_date__c != '' GROUP BY status
- Training trends by year: SELECT coursename, toInt32(year) AS course_year, COUNT(*) AS enrollments FROM iw_dev.member_course_history WHERE toInt32(year) >= 2020 GROUP BY coursename, course_year ORDER BY course_year DESC, enrollments DESC LIMIT 100
- Total hours worked for a company in a year (case-insensitive, partial match):SELECT SUM(hours) AS total_hours FROM iw_dev.employmenthistory WHERE lower(companyname) LIKE '%superior steel%' AND year = '2024'

OUTPUT FORMAT:
=============
Return a JSON object with:
- sql: The ClickHouse SQL query (empty string if cannot_answer is true)
- tables: Array of table names used in the query (empty array if cannot_answer is true)
- year: Year filter if applicable (for date validation)
- can_chart: Boolean indicating if results are suitable for charting
- chart_context: Create a brief context for charting based on the user prompt
- cannot_answer: Boolean - set to TRUE if the question asks about data NOT in the schema (e.g., projects, wages, payments)
- refusal_message: If cannot_answer is true, provide a helpful message explaining what data is not available. Available data includes: member information, certifications, drug tests, training records, and employment/contractor history."
"""

CHART_SYSTEM_PROMPT = """You are a data analyst that creates chart configurations and summaries from query results.

Your task is to analyze the provided data and create an appropriate visualization configuration.

MONTH LABELING RULE:
====================
- When displaying months on charts or in summaries, always map numeric month values (1–12) to their full English month names (e.g., 1 = January, 2 = February, ..., 12 = December) for clarity and readability.

CHART TYPE SELECTION RULES:
==========================
1. BAR CHART: Use for comparing categories or discrete values
   - Member counts by status, location, or category
   - Aggregated totals across groups
   
2. LINE CHART: Use for time-series or trend data with single series
   - Changes over months/years for one metric
   - Historical patterns for single category
   
3. MULTI-LINE CHART: Use for multi-dimensional data with 3-8 series
   - Multiple local unions over time
   - Multiple categories across years/states
   - Comparing trends between different groups
   - When data has structure like [{category1, time_period, value}, {category2, time_period, value}]
   
4. PIE CHART: Use for showing composition/proportions
   - Distribution percentages
   - Part-to-whole relationships
   - Best when 2-7 categories
   
5. SCATTER CHART: Use for showing relationships between two numeric values
   - Correlation analysis
   - Distribution patterns
   
6. TABLE: Use for complex data with 8+ series or detailed analysis
   - Complex multi-dimensional data
   - When precise values are more important than visual trends
   - Data with many categories or time periods
   - When chart would be cluttered

MULTI-DIMENSIONAL DATA DETECTION:
=================================
1. Look for data patterns like:
   - Multiple categories across time periods
   - Different groups (unions, states, statuses) with same metrics
   - Data that could be organized into multiple trend lines
   
2. Series count guidelines:
   - 1-2 series: Use regular line/bar chart
   - 3-8 series: Use multi_line chart for trends, table for detailed analysis
   - 8+ series: Use table format
   
3. Multi-line chart structure:
   - x_values: Common x-axis values (years, months, categories)
   - series_data: {"Series Name": [y_values for each x_value]}
   - Example: {"Local 377": [120, 135, 140], "Local 416": [98, 105, 110]}

DATA FORMATTING RULES:
=====================
1. For regular charts:
   - x_values: Array of strings for category labels or x-axis values
   - y_values: Array of numbers for the measurements
   
2. For multi_line charts:
   - x_values: Common x-axis values (e.g., years: ["2022", "2023", "2024"])
   - series_data: Dictionary mapping series names to their y-values
   - y_values: Can be empty or contain aggregated values
   
3. For table charts:
   - x_values and y_values can be empty
   - Focus on comprehensive summary
   
4. Ensure data consistency and limit to top 15 categories for readability
5. For pie charts, y_values should be positive numbers

SUMMARY GUIDELINES:
==================
1. Always provide a clear, concise summary of what the data shows
2. Highlight key insights (highest/lowest values, notable patterns, trends)
3. For multi-dimensional data, mention key comparisons between series
4. Include total counts when relevant
5. Keep summary under 4 sentences
6. Focus on the data insights, not on chart availability

OUTPUT FORMAT:
=============
Return a JSON object with:
- chart_type: 'bar', 'line', 'pie', 'scatter', 'multi_line', 'table', or null if no chart needed
- title: Descriptive chart title
- x_label: Label for x-axis
- y_label: Label for y-axis  
- x_values: Array of string labels (common axis for multi_line)
- y_values: Array of numeric values (for single series charts)
- series_data: Dictionary of series_name -> y_values (for multi_line charts only)
- summary: Text summary of the data insights
"""

# ============== CONSTANTS ==============

# Use current year for dynamic date validation
_CURRENT_YEAR = datetime.now(timezone.utc).year

TABLE_DATES = {
    "drugtestrecords": (2015, _CURRENT_YEAR),
    "member_course_history": (1956, _CURRENT_YEAR),
    "member_national_fund_trainings": (1985, _CURRENT_YEAR),
    "memberunionhistory1": (1986, _CURRENT_YEAR),
    "member_certifications": (1900, 2241),
    "member_online_registration_courses": (2012, _CURRENT_YEAR),
    "trainingprogramcertificates": (2006, _CURRENT_YEAR),
    "employmenthistory": (1900, _CURRENT_YEAR),
}

SENSITIVE_COLUMNS = {"ssn", "sin", "ssn__c", "phone1", "phone4", "phone_4", "emergencynumber", "address1", "address2", "postalcode"}

class UserRole(BaseModel):
    role: Literal["admin", "local_union_officer", "ironworker"]
    user_id: str
    local_union_id: Optional[str] = None

PRIMARY_TABLES = ["iw_contact08", "memberunionhistory1"]

LOCAL_TABLE_FIELDS = {
    "iw_contact08": "localunionid",
    "memberunionhistory1": "localunionid",
    "drugtestrecords": "local_union_number__c",
    "employmenthistory": "localunion",
}

MEMBER_TABLE_FIELDS = {
    "iw_contact08": "userid",
    "drugtestrecords": "active_latest_userid",
    "member_course_history": "active_latest_userid",
    "member_national_fund_trainings": "active_latest_userid",
    "memberunionhistory1": "active_latest_userid",
    "member_certifications": "active_latest_userid",
    "member_online_registration_courses": "active_latest_userid",
    "trainingprogramcertificates": "active_latest_userid",
    "employmenthistory": "active_latest_userid",
}

# ============== SQL FILTERING FUNCTIONS ==============

def predicate_exists(expression: exp.Expression, column: str, value: str) -> bool:
    column = column.lower()
    for eq in expression.find_all(exp.EQ):
        left, right = eq.this, eq.expression
        if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
            if left.name.lower() == column and right.this == value:
                return True
    return False

def ensure_predicate(expression: exp.Expression, column: str, value: str) -> None:
    if predicate_exists(expression, column, value):
        return
    
    predicate = exp.EQ(this=exp.column(column), expression=exp.Literal.string(value))
    where_clause = expression.args.get("where")
    
    if isinstance(where_clause, exp.Where):
        combined = exp.and_(where_clause.this, predicate)
        expression.set("where", exp.Where(this=combined))
    elif where_clause:
        combined = exp.and_(where_clause, predicate)
        expression.set("where", exp.Where(this=combined))
    else:
        expression.set("where", exp.Where(this=predicate))

def parse_select(sql_text: str) -> exp.Select:
    try:
        expression = sqlglot.parse_one(sql_text, read="clickhouse")
    except sqlglot.errors.ParseError as exc:
        raise ValueError(f"SQL parse error: {exc}") from exc
    if not isinstance(expression, exp.Select):
        raise ValueError("Only SELECT statements are permitted")
    return expression

def collect_table_names(expression: exp.Expression) -> set[str]:
    names: set[str] = set()
    for table in expression.find_all(exp.Table):
        if table.name:
            table_name = table.name.lower()
            if '.' in table_name:
                table_name = table_name.split('.')[-1]
            names.add(table_name)
    return names

def inject_role_predicates(sql_text: str, tables: list[str], user_role: UserRole) -> str:
    if user_role.role == "admin":
        return sql_text
    
    select = parse_select(sql_text)
    
    if user_role.role == "local_union_officer":
        target_value = user_role.local_union_id
        table_map = LOCAL_TABLE_FIELDS
    else:
        target_value = user_role.user_id
        table_map = MEMBER_TABLE_FIELDS
    
    if not target_value:
        raise ValueError("Missing required identity attribute")
    
    tables_lower = collect_table_names(select)
    tables_lower.update(name.lower() for name in tables)
    
    primary_table_found = None
    for primary_table in PRIMARY_TABLES:
        if primary_table in tables_lower:
            primary_table_found = primary_table
            break
    
    if primary_table_found:
        column = table_map.get(primary_table_found)
        if column:
            ensure_predicate(select, column, target_value)
    else:
        for table in tables_lower:
            column = table_map.get(table)
            if column:
                ensure_predicate(select, column, target_value)
                break
    
    return select.sql(dialect="clickhouse")

def ensure_required_predicates(select: exp.Select, user_role: UserRole) -> None:
    if user_role.role == "admin":
        return
    
    if user_role.role == "ironworker":
        user_value = user_role.user_id
        if not (predicate_exists(select, "userid", user_value) or 
                predicate_exists(select, "active_latest_userid", user_value)):
            raise ValueError("Member queries must filter by userid or active_latest_userid")
    
    if user_role.role == "local_union_officer":
        local_value = user_role.local_union_id
        if not local_value:
            raise ValueError("Local officer role is missing local_union_id")
        if not predicate_exists(select, "localunionid", local_value):
            raise ValueError("Local officer queries must filter by localunionid")

def validate_and_format_clickhouse_sql(sql: str) -> str:
    try:
        ast = sqlglot.parse_one(sql, read="clickhouse")
    except errors.ParseError as exc:
        raise ValueError(f"SQL syntax error: {exc}") from exc
    
    return sqlglot.transpile(
        ast.sql(dialect="clickhouse"),
        read="clickhouse",
        write="clickhouse",
        pretty=True,
    )[0]

def redact_rows(rows: list[dict[str, Any]], denied_columns: set[str]) -> list[dict[str, Any]]:
    if not denied_columns:
        return rows
    return [{k: v for k, v in row.items() if k.lower() not in denied_columns} for row in rows]

# ============== MODELS ==============

class SQLQuery(BaseModel):
    sql: str = ""  # Empty if cannot_answer is True
    tables: list[str] = []
    year: str | None = None
    can_chart: bool = True
    chart_context: str | None = None
    cannot_answer: bool = False  # Set to True if the question cannot be answered with available data
    refusal_message: str | None = None  # Explanation when cannot_answer is True

class ChartData(BaseModel):
    chart_type: Literal["bar", "line", "pie", "scatter", "multi_line", "table"] | None = None
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    x_values: list[str] = []
    y_values: list[int | float] = []
    series_data: dict[str, list[int | float]] | None = None  # For multi-line charts: {"series_name": [values]}
    summary: str

class State(TypedDict):
    question: str
    user_role: UserRole
    session_id: str
    trace_id: str
    sql_query: SQLQuery | None
    filtered_sql: str | None
    filter_metadata: dict | None
    validation_result: dict | None
    date_valid: bool
    error: str | None
    data: list[dict]
    chart: ChartData | None
    denied_columns: set[str]
    chart_context: str | None
    # Token tracking
    total_input_tokens: int
    total_output_tokens: int

# ============== LLM AGENTS (UPDATED WITH MEMORY) ==============

def call_sql_agent(prompt: str, trace_id: str = None, conversation_messages: list[BaseMessage] = None) -> tuple[SQLQuery, dict]:
    """Generate SQL with Langfuse logging and optional conversation context.
    Returns tuple of (SQLQuery, token_usage_dict)
    """
    
    if conversation_messages:
        messages = conversation_messages
    else:
        messages = [
            SystemMessage(content=get_sql_system_prompt()),
            HumanMessage(content=prompt)
        ]

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Build config with optional Langfuse callback
    config = {"run_name": "sql-generation"}
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]
    
    # Use include_raw=True to get both structured output AND raw response with token usage
    sql_agent = llm.with_structured_output(SQLQuery, include_raw=True)
    response = sql_agent.invoke(messages, config=config)
    
    # Extract structured result and token usage
    result = response['parsed']
    raw_response = response['raw']
    
    # Extract token usage from raw response metadata
    token_usage = {'input_tokens': 0, 'output_tokens': 0}
    if hasattr(raw_response, 'response_metadata') and raw_response.response_metadata:
        usage = raw_response.response_metadata.get('token_usage', {})
        token_usage = {
            'input_tokens': usage.get('prompt_tokens', 0),
            'output_tokens': usage.get('completion_tokens', 0)
        }
    
    logger.info("SQL query generated", extra={
        "trace_id": trace_id, "sql_preview": result.sql[:100] if result.sql else "empty",
        "tables": result.tables, "can_chart": result.can_chart,
        "has_conversation_context": bool(conversation_messages),
        "langfuse_enabled": langfuse_handler is not None,
        "tokens": token_usage
    })
    
    return result, token_usage

def call_chart_agent(prompt: str, trace_id: str = None) -> tuple[ChartData, dict]:
    """Generate chart with Langfuse logging.
    Returns tuple of (ChartData, token_usage_dict)
    """
    messages = [
        SystemMessage(content=CHART_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Build config with optional Langfuse callback
    config = {
        "run_name": "chart-generation",
        "metadata": {"trace_id": trace_id}
    }
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]
    
    # Use include_raw=True to get both structured output AND raw response with token usage
    chart_agent = llm.with_structured_output(ChartData, include_raw=True)
    response = chart_agent.invoke(messages, config=config)
    
    # Extract structured result and token usage
    result = response['parsed']
    raw_response = response['raw']
    
    # Extract token usage from raw response metadata
    token_usage = {'input_tokens': 0, 'output_tokens': 0}
    if hasattr(raw_response, 'response_metadata') and raw_response.response_metadata:
        usage = raw_response.response_metadata.get('token_usage', {})
        token_usage = {
            'input_tokens': usage.get('prompt_tokens', 0),
            'output_tokens': usage.get('completion_tokens', 0)
        }

    logger.info("Chart configuration generated", extra={
        "trace_id": trace_id, "chart_type": result.chart_type,
        "title": result.title, "data_points": len(result.x_values),
        "langfuse_enabled": langfuse_handler is not None,
        "tokens": token_usage
    })
    
    return result, token_usage

def call_chart_agent_for_regeneration(
    question: str, 
    data: list, 
    old_config: dict, 
    old_summary: str, 
    trace_id: str = None
) -> tuple[ChartData, dict]:
    """
    Specialized chart agent call for regeneration.
    Uses old configuration and summary as context to ensure consistency.
    """
    columns = list(data[0].keys()) if data else []
    
    agent_prompt = f"""
USER ORIGINAL QUESTION: {question}

PREVIOUS CHART CONFIGURATION:
- Chart Type: {old_config.get('chart_type')}
- Title: {old_config.get('title')}
- X-Label: {old_config.get('x_label')}
- Y-Label: {old_config.get('y_label')}
- Series Config: {old_config.get('series_config')}

PREVIOUS SUMMARY:
{old_summary}

FRESH DATA INFO:
- Columns: {columns}
- Total rows: {len(data)}
- Full dataset (sample): {data[:100]}

TASK:
1. Regenerate the chart using this FRESH DATA.
2. Maintain the SAME STYLE and FOCUS as the previous chart (keep chart_type, x_label, y_label if still appropriate).
3. Update the SUMMARY to reflect the new numbers while keeping the same tone and insights.
4. Ensure data consistency and limit to top 15 categories for readability.
"""

    messages = [
        SystemMessage(content=CHART_SYSTEM_PROMPT),
        HumanMessage(content=agent_prompt)
    ]
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    config = {
        "run_name": "chart-regeneration-agent",
        "metadata": {"trace_id": trace_id}
    }
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]
    
    chart_agent = llm.with_structured_output(ChartData, include_raw=True)
    response = chart_agent.invoke(messages, config=config)
    
    result = response['parsed']
    raw_response = response['raw']
    
    token_usage = {'input_tokens': 0, 'output_tokens': 0}
    if hasattr(raw_response, 'response_metadata') and raw_response.response_metadata:
        usage = raw_response.response_metadata.get('token_usage', {})
        token_usage = {
            'input_tokens': usage.get('prompt_tokens', 0),
            'output_tokens': usage.get('completion_tokens', 0)
        }

    logger.info("Chart regenerated via Agent", extra={
        "trace_id": trace_id, 
        "chart_type": result.chart_type,
        "tokens": token_usage
    })
    
    return result, token_usage

# ============== HELPER FUNCTIONS ==============

def sanitize_json_data(obj):
    """Recursively sanitize data for JSON serialization"""
    if isinstance(obj, dict):
        return {key: sanitize_json_data(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json_data(item) for item in obj]
    elif isinstance(obj, Decimal):
        # Convert Decimal to float for JSON serialization
        return float(obj)
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj

def create_sql_prompt_with_history(question: str, messages: list[BaseMessage]) -> list[BaseMessage]:
    """Create SQL generation prompt with conversation history"""
    system_message = SystemMessage(content=get_sql_system_prompt())
    recent_messages = messages[-10:] if len(messages) > 10 else messages
    current_question = HumanMessage(content=f"USER QUESTION: {question}\n\nGenerate the SQL query based on the schema and rules provided.")
    return [system_message] + recent_messages + [current_question]

def extract_series_config(chart_data: dict, raw_data: list[dict]) -> dict | None:
    """
    Extract series configuration from chart data and raw query results.
    This allows us to recreate the same chart structure with fresh data.
    """
    if not chart_data or not raw_data:
        return None
    
    chart_type = chart_data.get('chart_type')
    
    if chart_type == 'multi_line':
        # For multi-line charts, figure out which column maps to what
        data_columns = list(raw_data[0].keys())
        
        # Find y_column: MUST be numeric in ALL rows (not just first)
        y_column = None
        for col in data_columns:
            # Check if this column is numeric across multiple rows
            is_numeric = True
            for row in raw_data[:5]:  # Check first 5 rows
                val = row.get(col)
                if not isinstance(val, (int, float, Decimal)):
                    is_numeric = False
                    break
            if is_numeric:
                y_column = col
                break
        
        # Find x_column: Look for column that could be x-axis
        # Try to match against x_values (with some flexibility)
        x_column = None
        x_values = chart_data.get('x_values', [])
        if x_values:
            for col in data_columns:
                if col == y_column:
                    continue
                # Check if values in this column appear in x_values (even partially)
                sample_val = str(raw_data[0].get(col, ''))
                for x_val in x_values[:3]:  # Check first few x_values
                    if sample_val in str(x_val) or str(x_val) in sample_val:
                        x_column = col
                        break
                if x_column:
                    break
        
        # If x_column still not found, use first non-numeric column
        if not x_column:
            for col in data_columns:
                if col != y_column:
                    x_column = col
                    break
        
        # Find series_column: remaining column (not x or y)
        series_column = None
        for col in data_columns:
            if col != x_column and col != y_column:
                series_column = col
                break
        
        return {
            "x_column": x_column,
            "series_column": series_column,
            "y_column": y_column
        }
    
    elif chart_type in ['bar', 'line', 'pie']:
        # Simple charts - x and y
        data_columns = list(raw_data[0].keys())
        
        # Find y_column: numeric column
        y_column = None
        for col in data_columns:
            val = raw_data[0].get(col)
            if isinstance(val, (int, float, Decimal)):
                y_column = col
                break
        
        # x_column: first non-numeric column
        x_column = None
        for col in data_columns:
            if col != y_column:
                x_column = col
                break
        
        return {
            "x_column": x_column,
            "y_column": y_column
        }
    
    return None

# ============== WORKFLOW NODES (UPDATED WITH CONVERSATIONSUMMARYBUFFER) ==============

@log_node("generate_sql")
def generate_sql(state: State) -> State:
    """Generate SQL using ConversationSummaryBufferMemory for smart context management"""
    try:
        base_question = state['question']
        session_id = state['session_id']     
        
        # Get conversation memory (with automatic summarization!)
        conv_memory = get_conversation_memory(session_id, k=10)
        
        # Get messages from memory (already includes summary if exists)
        conversation_messages = conv_memory.messages
        
        if conversation_messages:
            logger.info(f"Using ConversationSummaryBufferMemory for session {session_id} ({len(conversation_messages)} messages)")
            
            # Check if first message is a summary
            has_summary = isinstance(conversation_messages[0], SystemMessage) if conversation_messages else False
            if has_summary:
                logger.info(f">> Memory includes summary + {len(conversation_messages)-1} recent messages")
            
            # Create SQL prompt with memory context
            sql_prompt_messages = create_sql_prompt_with_history(base_question, conversation_messages)
            
            # Generate SQL with context - now returns tuple (result, token_usage)
            sql_result, sql_tokens = call_sql_agent(
                prompt=f"USER QUESTION: {base_question}\\n\\nGenerate SQL with conversation context.",
                trace_id=state.get('trace_id'),
                conversation_messages=sql_prompt_messages
            )
            state['sql_query'] = sql_result
            
            # Accumulate tokens
            state['total_input_tokens'] += sql_tokens.get('input_tokens', 0)
            state['total_output_tokens'] += sql_tokens.get('output_tokens', 0)
            
            # Handle refusal case - question cannot be answered with available data
            if state['sql_query'] and state['sql_query'].cannot_answer:
                refusal_msg = state['sql_query'].refusal_message or "I cannot answer this question as that data is not available in the database."
                state['error'] = refusal_msg
                state['sql_query'] = None
                logger.info(f"Query refused: {refusal_msg}", extra={"trace_id": state.get('trace_id')})
                return state
            
            # Set chart context from the SQL query result
            if state['sql_query'] and state['sql_query'].chart_context:
                state['chart_context'] = state['sql_query'].chart_context
            else:
                # Generate basic chart context from the question
                state['chart_context'] = f"Chart visualization for: {base_question}"
        
            logger.info("SQL generation with conversation context", extra={
                "trace_id": state.get('trace_id'), 
                "conversation_messages": len(conversation_messages),
                "has_summary": has_summary,
                "session_id": session_id,
                "sql_tokens": sql_tokens
            })
        else:
            # No history - first message - now returns tuple (result, token_usage)
            sql_result, sql_tokens = call_sql_agent(
                f"""USER QUESTION: {base_question}

    Generate the SQL query based on the schema and rules provided.""",
                state.get('trace_id')
            )
            state['sql_query'] = sql_result
            
            # Accumulate tokens
            state['total_input_tokens'] += sql_tokens.get('input_tokens', 0)
            state['total_output_tokens'] += sql_tokens.get('output_tokens', 0)
            
            # Handle refusal case - question cannot be answered with available data
            if state['sql_query'] and state['sql_query'].cannot_answer:
                refusal_msg = state['sql_query'].refusal_message or "I cannot answer this question as that data is not available in the database."
                state['error'] = refusal_msg
                state['sql_query'] = None
                logger.info(f"Query refused: {refusal_msg}", extra={"trace_id": state.get('trace_id')})
                return state
            
            # Set chart context from the SQL query result
            if state['sql_query'] and state['sql_query'].chart_context:
                state['chart_context'] = state['sql_query'].chart_context
            else:
                # Generate basic chart context from the question
                state['chart_context'] = f"Chart visualization for: {base_question}"
            
            logger.info("SQL generation without conversation context (first message)", extra={
                "trace_id": state.get('trace_id'), "session_id": session_id,
                "sql_tokens": sql_tokens
            })
        
        return state
    
    except Exception as e:
        logger.error(f"Error in generate_sql for session {state.get('session_id')}: {e}", extra={
            "session_id": state.get('session_id'),
            "trace_id": state.get('trace_id'),
            "question_preview": state.get('question', '')[:100],
            "error": str(e),
            "method": "generate_sql"
        })
        raise  # Re-raise to bubble up to main method

@log_node("apply_rbac_filter")
def apply_rbac_filter(state: State) -> State:
    if not state['sql_query']:
        return state
    
    try:
        filtered_sql = inject_role_predicates(
            state['sql_query'].sql,
            state['sql_query'].tables,
            state['user_role']
        )
        state['filtered_sql'] = filtered_sql
        state['filter_metadata'] = {
            'filter_applied': True,
            'role': state['user_role'].role,
            'tables': state['sql_query'].tables
        }
    except Exception as e:
        logger.error(f"Error in apply_rbac_filter for session {state.get('session_id')}: {e}", extra={
            "session_id": state.get('session_id'),
            "trace_id": state.get('trace_id'),
            "user_role": state.get('user_role', {}).get('role') if state.get('user_role') else None,
            "error": str(e),
            "method": "apply_rbac_filter"
        })
        raise  # Re-raise to bubble up to main method
    
    return state

@log_node("validate_filter")
def validate_filter(state: State) -> State:
    if not state['sql_query'] or not state['filtered_sql'] or state['error']:
        return state
    
    try:
        select = parse_select(state['filtered_sql'])
        ensure_required_predicates(select, state['user_role'])
        validated_sql = validate_and_format_clickhouse_sql(state['filtered_sql'])
        state['filtered_sql'] = validated_sql
        logger.info(state['filtered_sql'], extra={"trace_id": state.get('trace_id')})
        state['validation_result'] = {'valid': True, 'issues': []}
    except ValueError as e:
        logger.error(f"Error in validate_filter for session {state.get('session_id')}: {e}", extra={
            "session_id": state.get('session_id'),
            "trace_id": state.get('trace_id'),
            "sql_query": state.get('sql_query', {}).get('sql', '')[:100] if state.get('sql_query') else None,
            "error": str(e),
            "method": "validate_filter"
        })
        raise  # Re-raise to bubble up to main method
    
    return state

@log_node("validate_dates")
def validate_dates(state: State) -> State:
    if state['error']:
        return state
    
    try:
        query = state['sql_query']
        if query and query.year:
            for table in query.tables:
                if table in TABLE_DATES:
                    try:
                        min_yr, max_yr = TABLE_DATES[table]
                        year_int = int(query.year)
                        if not (min_yr <= year_int <= max_yr):
                            state['date_valid'] = False
                            state['error'] = f"{table} only has {min_yr}-{max_yr} data"
                            return state
                    except (ValueError, TypeError):
                        pass
        
        state['date_valid'] = True
        return state
    
    except Exception as e:
        logger.error(f"Error in validate_dates for session {state.get('session_id')}: {e}", extra={
            "session_id": state.get('session_id'),
            "trace_id": state.get('trace_id'),
            "query_year": state.get('sql_query', {}).get('year') if state.get('sql_query') else None,
            "tables": state.get('sql_query', {}).get('tables', []) if state.get('sql_query') else [],
            "error": str(e),
            "method": "validate_dates"
        })
        raise  # Re-raise to bubble up to main method

@log_node("run_query")
def run_query(state: State) -> State:
    if not state['date_valid'] or state['error']:
        return state
    
    try:
        client = get_clickhouse_db()
        result = client.query(state['filtered_sql'])
        rows = [dict(zip(result.column_names, row)) for row in result.result_rows]
        state['data'] = redact_rows(rows, state['denied_columns'])
        logger.info("Query executed successfully", extra={
            "trace_id": state.get('trace_id'), "row_count": len(state['data']),
            "redacted_columns": len(state['denied_columns'])
        })
        logger.info(state['data'], extra={"trace_id": state.get('trace_id')})
    except Exception as e:
        logger.error(f"Error in run_query for session {state.get('session_id')}: {e}", extra={
            "session_id": state.get('session_id'),
            "trace_id": state.get('trace_id'),
            "filtered_sql": state.get('filtered_sql', '')[:100] if state.get('filtered_sql') else None,
            "error": str(e),
            "method": "run_query"
        })
        raise  # Re-raise to bubble up to main method
    
    return state

@log_node("create_chart")
def create_chart(state: State) -> State:
    if not state['data'] or state['error']:
        return state
    
    can_chart = state['sql_query'] and state['sql_query'].can_chart and len(state['data']) > 2
    columns = list(state['data'][0].keys()) if state['data'] else []
    
    user_prompt = f"""USER QUESTION: {state['question']}

DATA INFO:
- Columns: {columns}
- Total rows: {len(state['data'])}
- Full dataset: {state['data']}
- Should create chart: {can_chart}
- Should create chart based on the Chart context: {state.get('chart_context', 'N/A')}

Analyze this data and provide the chart configuration and summary."""
    
    try:
        # Now returns tuple (result, token_usage)
        chart_result, chart_tokens = call_chart_agent(user_prompt, state.get('trace_id'))
        state['chart'] = chart_result
        
        # Accumulate tokens from chart generation
        state['total_input_tokens'] += chart_tokens.get('input_tokens', 0)
        state['total_output_tokens'] += chart_tokens.get('output_tokens', 0)
        
        logger.info("Chart created with token tracking", extra={
            "trace_id": state.get('trace_id'),
            "chart_tokens": chart_tokens,
        })
    except Exception as e:
        logger.error(f"Error in create_chart for session {state.get('session_id')}: {e}", extra={
            "session_id": state.get('session_id'),
            "trace_id": state.get('trace_id'),
            "data_count": len(state.get('data', [])),
            "chart_context": state.get('chart_context', '')[:100],
            "error": str(e),
            "method": "create_chart"
        })
        raise  # Re-raise to bubble up to main method
    
    return state

# ============== WORKFLOW SETUP (YOUR ORIGINAL STRUCTURE) ==============

workflow = StateGraph(State)
workflow.add_node("sql", generate_sql)
workflow.add_node("filter", apply_rbac_filter)
workflow.add_node("validate_filter", validate_filter)
workflow.add_node("validate_dates", validate_dates)
workflow.add_node("query", run_query)
workflow.add_node("chart", create_chart)

workflow.set_entry_point("sql")
workflow.add_edge("sql", "filter")
workflow.add_edge("filter", "validate_filter")
workflow.add_conditional_edges(
    "validate_filter",
    lambda s: "error" if s['error'] else "ok",
    {"error": END, "ok": "validate_dates"}
)
workflow.add_conditional_edges(
    "validate_dates",
    lambda s: "error" if not s['date_valid'] or s['error'] else "ok",
    {"error": END, "ok": "query"}
)
workflow.add_edge("query", "chart")
workflow.add_edge("chart", END)

# Compile graph (NO checkpointer needed - memory is managed by ConversationSummaryBuffer)
graph = workflow.compile()

# ============== API MODELS ==============

class Query(BaseModel):
    question: str
    session_id: Optional[str] = None

class RegenerateChartRequest(BaseModel):
    message_id: str
    session_id: str

# ============== MAIN WORKFLOW (UPDATED WITH MEMORY) ==============

def run_chatbot_query(question: str, user_role: UserRole, session_id: str, trace_id: str) -> dict:
    """Execute workflow with ConversationSummaryBufferMemory and token tracking"""
    
    try:
        denied_columns = set() if user_role.role == "admin" else SENSITIVE_COLUMNS
        
        # Get conversation memory (this will load from database if session not in active memory)
        conv_memory = get_conversation_memory(session_id, k=10)
        
        # Log the loaded conversation context (safely check for summary)
        has_summary = len(conv_memory.messages) > 0 and isinstance(conv_memory.messages[0], SystemMessage)
        logger.info(f"Loaded conversation memory for session {session_id}: {len(conv_memory.messages)} messages", extra={
            "session_id": session_id,
            "trace_id": trace_id,
            "loaded_messages": len(conv_memory.messages),
            "has_summary": has_summary
        })
        
        # Add user question to memory BEFORE workflow (for context during SQL generation)
        conv_memory._add_message_to_memory_only(HumanMessage(content=question))
        
        # Execute workflow (SQL generation will use existing conversation history)
        result = graph.invoke({
            "question": question,
            "user_role": user_role,
            "session_id": session_id,
            "trace_id": trace_id,
            "sql_query": None,
            "filtered_sql": None,
            "filter_metadata": None,
            "validation_result": None,
            "date_valid": True,
            "error": None,
            "data": [],
            "chart": None,
            "denied_columns": denied_columns,
            "chart_context": None,
            # Initialize token tracking
            "total_input_tokens": 0,
            "total_output_tokens": 0
        })
        
        # Note: User question is already saved to permanent storage in /query endpoint
        
        # Save assistant response with FULL rich metadata (chart, SQL, data, tokens) -- now handled in process() for correct execution_time
        if result.get('chart') and result['chart'].summary:
            # Add RICH context to memory so LLM knows what data was returned
            # Include SQL query and key data points for proper conversation context
            memory_content = result['chart'].summary
            if result.get('sql_query') and result['sql_query'].sql:
                memory_content += f"\n\n[SQL executed: {result['sql_query'].sql}]"
            if result.get('data') and len(result['data']) <= 10:
                # Include actual data for small result sets so LLM has context
                memory_content += f"\n[Data returned: {result['data']}]"
            elif result.get('data'):
                # For larger results, include first few rows
                memory_content += f"\n[Data sample (first 3 of {len(result['data'])}): {result['data'][:3]}]"
            conv_memory._add_message_to_memory_only(AIMessage(content=memory_content))
            logger.info(f"Prepared AI response with tokens: input={result.get('total_input_tokens', 0)}, output={result.get('total_output_tokens', 0)}", extra={
                "trace_id": trace_id,
                "session_id": session_id
            })
        elif result.get('error'):
            conv_memory._add_message_to_memory_only(AIMessage(content=f"Error: {result['error']}"))
        
        return result
        
    except Exception as e:
        logger.error(f"Error in run_chatbot_query for session {session_id}: {e}", extra={
            "session_id": session_id,
            "trace_id": trace_id,
            "question_preview": question[:100] if question else "empty",
            "error": str(e),
            "method": "run_chatbot_query"
        })
        raise  # Re-raise to bubble up to endpoint

@app.post("/query")
def process(q: Query, request: Request):
    start_time = datetime.now(timezone.utc)
    
    trace_id = getattr(request.state, 'trace_id', str(uuid.uuid4()))
    
    user_role = UserRole(
        role="admin",
        user_id="cda977a7-7bc5-4007-9316-dc315a0037c0",
        local_union_id="782"
    )
    
    if q.session_id:
        session_id = q.session_id
    else:
        session_id = str(uuid.uuid4())
    
    logger.info(f"Processing query for session {session_id}", extra={
        "trace_id": trace_id,
        "session_id": session_id,
        "question_preview": q.question[:100] if q.question else "empty"
    })
    
    # Save user question to permanent storage IMMEDIATELY so session appears right away
    # This ensures the session is visible even if user refreshes before workflow completes
    save_to_permanent_storage(session_id, "human", q.question, user_id=user_role.user_id,session_status='In Progress')
    
    try:
        result = run_chatbot_query(q.question, user_role, session_id, trace_id)
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        chart_data = result['chart'].model_dump() if result['chart'] else None

        if result['error']:
            summary = result['error']
        else:
             summary = chart_data.get('summary') if chart_data else "Query successfully completed"


        # Save the AI response with execution_time to permanent storage
        message_id = save_to_permanent_storage(
            session_id=session_id,
            message_type="assistant",
            content=summary,
            user_id=user_role.user_id,
            sql_query=result['sql_query'].sql if result.get('sql_query') else None,
            filtered_sql=result.get('filtered_sql'),
            chart_data=chart_data,
            result_data=result.get('data'),
            execution_time=duration,
            langfuse_trace_id=trace_id,
            input_tokens=result.get('total_input_tokens', 0),
            output_tokens=result.get('total_output_tokens', 0),
            session_status='Completed'
        )

        response_data = {
            "is_success": True,
            "message": "success",
            "data": {
                "message_id": message_id,
                "session_id": session_id,
                "trace_id": trace_id,
                "session_status": "Completed",
                "type": "assistant",
                "sql_query": result['sql_query'].sql if result['sql_query'] else None,
                "filtered_sql": result['filtered_sql'],
                "content": summary,
                "chart_data": chart_data,
                "data": result['data'],
                "execution_time": f"{duration:.2f}s",
                "input_tokens": result.get('total_input_tokens', 0),
                "output_tokens": result.get('total_output_tokens', 0)
            }
        }

        logger.info(f"Query processed successfully for session {session_id} in {duration:.2f}s", extra={
            "trace_id": trace_id,
            "session_id": session_id,
            "execution_time": duration
        })
        return sanitize_json_data(response_data)
    except Exception as e:
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        error_message = str(e)
        
        # Save error information to database
        try:
            message_id = save_to_permanent_storage(
                session_id=session_id,
                message_type="assistant",
                content=f"Error: {error_message}",
                user_id=user_role.user_id,
                session_status='Error',
                execution_time=duration,
                error=error_message,
                langfuse_trace_id=trace_id
            )
        except Exception as db_error:
            # If database save fails, log it but don't prevent the error response
            logger.error(f"Failed to save error to database for session {session_id}: {db_error}")
        
        logger.error(f"Query processing failed for session {session_id}: {e}", extra={
            "trace_id": trace_id,
            "session_id": session_id,
            "error": str(e),
            "execution_time": duration
        })
        
        # Return consistent JSON response format for errors
        error_response = {
            "is_success": True,
            "message": "success",
            "data": {
                "message_id": message_id if 'message_id' in locals() else None,
                "session_id": session_id,
                "trace_id": trace_id,
                "type": "error",
                "sql_query": None,
                "filtered_sql": None,
                "content": None,
                "chart_data": None,
                "error": error_message,
                "data": None,
                "execution_time": f"{duration:.2f}s",
                "input_tokens": 0,
                "output_tokens": 0,
                "session_status": "Error"
            }
        }
        
        return sanitize_json_data(error_response)

@app.post("/regenerate-chart")
def regenerate_chart(regenerate_request: RegenerateChartRequest, request: Request):
    """
    Regenerate chart with fresh data using saved SQL and chart config.
    No need to go through full workflow - just execute SQL and rebuild chart.
    """
    start_time = datetime.now(timezone.utc)
    trace_id = getattr(request.state, 'trace_id', str(uuid.uuid4()))
    
    logger.info(f"Regenerating chart for message {regenerate_request.message_id}", extra={
        "trace_id": trace_id,
        "message_id": regenerate_request.message_id,
        "session_id": regenerate_request.session_id
    })
    
    try:
        # 1. Fetch all context in "One Glance" (Assistant message + Original user question)
        conn = get_postgres_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT 
                assist.sql_query,
                assist.filtered_sql, 
                assist.chart_data, 
                assist.content,
                assist.langfuse_trace_id,
                assist.input_tokens,
                assist.output_tokens,
                assist.created_at,
                (SELECT content FROM chat_messages 
                 WHERE session_id = assist.session_id 
                   AND role = 'user' 
                   AND created_at < assist.created_at 
                 ORDER BY created_at DESC LIMIT 1) as original_question
            FROM chat_messages assist
            WHERE assist.message_id = %s AND assist.session_id = %s
        """, (regenerate_request.message_id, regenerate_request.session_id))
        
        message_with_context = cur.fetchone()
        
        if not message_with_context:
            cur.close()
            release_postgres_conn(conn)
            raise HTTPException(status_code=404, detail="Regeneration context not found")
            
        original_question = message_with_context['original_question'] or "Regenerate chart"
        saved_filtered_sql = message_with_context['filtered_sql']
        old_chart_config = message_with_context['chart_data'] or {}
        old_summary = message_with_context['content'] or old_chart_config.get('summary', '')

        logger.info(f"Regeneration context loaded.\nOriginal Question: {original_question}\nOld Summary: {old_summary}\nOld Config: {old_chart_config}", extra={
            "trace_id": trace_id,
        })
        
        if not saved_filtered_sql:
            raise HTTPException(status_code=400, detail="No SQL found for this message")
        
        # 2. Execute saved SQL with fresh data
        clickhouse_client = get_clickhouse_db()
        result = clickhouse_client.query(saved_filtered_sql)
        rows = [dict(zip(result.column_names, row)) for row in result.result_rows]
        fresh_data = sanitize_json_data(rows)
        
        if not fresh_data:
            return {"is_success": False, "message": "No data available", "data": None}
        
        # 3. Call specialized Regeneration Agent for consistency
        chart_result, tokens = call_chart_agent_for_regeneration(
            question=original_question,
            data=fresh_data,
            old_config=old_chart_config,
            old_summary=old_summary,
            trace_id=trace_id
        )
        
        new_summary = chart_result.summary
        fresh_chart = chart_result.model_dump()
        current_time = datetime.now(timezone.utc)
        
        # 4. Update Database (content and updated_at ONLY as requested)
        # Re-using the connection from earlier if we didn't close it, but let's be safe
        # In the context of this function, we can re-open or keep open.
        # Fixed: Closing/Re-opening is cleaner if we released it above.
        conn = get_postgres_conn()
        cur = conn.cursor()
        cur.execute("""
            UPDATE chat_messages 
            SET content = %s, chart_data = %s, updated_at = %s 
            WHERE message_id = %s
        """, (new_summary, json.dumps(fresh_chart), current_time, regenerate_request.message_id))
        
        cur.execute("""
            UPDATE chat_sessions 
            SET last_message_at = %s 
            WHERE session_id = %s
        """, (current_time, regenerate_request.session_id))
        conn.commit()
        cur.close()
        release_postgres_conn(conn)
        
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return sanitize_json_data({
            "is_success": True,
            "message": "success",
            "data": {
                "message_id": regenerate_request.message_id,
                "session_id": regenerate_request.session_id,
                "trace_id": message_with_context.get('langfuse_trace_id'),
                "session_status": "Completed",
                "type": "assistant",
                "sql_query": message_with_context.get('sql_query'),
                "filtered_sql": saved_filtered_sql,
                "content": new_summary,
                "chart_data": fresh_chart,
                "data": fresh_data,
                "execution_time": f"{duration:.2f}s",
                "input_tokens": message_with_context.get('input_tokens', 0),
                "output_tokens": message_with_context.get('output_tokens', 0),
                "updated_at": current_time.isoformat(),
                "created_at": message_with_context.get('created_at').isoformat() if message_with_context.get('created_at') else None
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.error(f"Chart regeneration failed: {e}", extra={
            "trace_id": trace_id,
            "message_id": regenerate_request.message_id,
            "error": str(e),
            "execution_time": duration
        })
        return {
            "is_success": False,
            "message": f"Regeneration failed: {str(e)}",
            "data": None
        }

# ============== SESSION MANAGEMENT ENDPOINTS (UPDATED) ==============

@app.get("/sessions")
def get_sessions(page: int = 1, page_size: int = 20, user_id: str = None):
    """Return session info from permanent storage with pagination and optional user filtering"""
    logger.info(f"Fetching sessions from permanent storage - page {page}, size {page_size}, user_id {user_id}")
    
    # Validate pagination parameters
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:  # Limit max page size to 100
        page_size = 20
    
    offset = (page - 1) * page_size
    
    try:
        conn = get_postgres_conn()
        cur = conn.cursor()
        
        # Get total count for pagination metadata with optional user filter

        cur.execute("""
            SELECT COUNT(DISTINCT cs.session_id) as total_sessions
            FROM chat_sessions cs
            WHERE cs.user_id = %s
        """, (user_id,))
        total_sessions = cur.fetchone()[0]
        
        # Calculate pagination metadata
        total_pages = (total_sessions + page_size - 1) // page_size
        has_next = page < total_pages
        has_prev = page > 1
        
        # Get paginated sessions with optional user filter
        cur.execute("""
            SELECT cs.session_id, cs.user_id, cs.session_status, cs.created_at, cs.last_message_at,
                    COUNT(cm.message_id) as message_count,
                    MIN(CASE WHEN cm.role = 'user' THEN cm.content END) as first_question
            FROM chat_sessions cs
            LEFT JOIN chat_messages cm ON cs.session_id = cm.session_id
            WHERE cs.user_id = %s
            GROUP BY cs.session_id, cs.user_id, cs.session_status, cs.created_at, cs.last_message_at
            ORDER BY cs.last_message_at DESC 
            LIMIT %s OFFSET %s
        """, (user_id, page_size, offset))

        
        sessions = cur.fetchall()
        cur.close()
        
        session_list = []
        for row in sessions:
            session_id, user_id, session_status, created_at, last_message_at, msg_count, first_q = row
            session_list.append({
                "session_id": str(session_id),
                "user_id": user_id,
                "session_status": session_status,
                "message_count": msg_count,
                "created_at": created_at.isoformat() if created_at else None,
                "last_message_at": last_message_at.isoformat() if last_message_at else None,
                "first_question": (first_q[:50] + "...") if first_q and len(first_q) > 50 else first_q
            })
        
        return {
            "is_success": True,
            "message": "success",
            "data": {
                "message": f"Sessions with ConversationSummaryBufferMemory{' for user ' + user_id if user_id else ''}",
                "memory_type": "ConversationSummaryBufferMemory + PostgreSQL",
                "active_memories": len(conversation_memories),
                "sessions": session_list,
                "user_filter": user_id,
                "pagination": {
                    "current_page": page,
                    "page_size": page_size,
                    "total_sessions": total_sessions,
                    "total_pages": total_pages,
                    "has_next": has_next,
                    "has_prev": has_prev,
                    "next_page": page + 1 if has_next else None,
                    "prev_page": page - 1 if has_prev else None
                }
            }
        }
    except Exception as e:
        logger.error(f"Failed to get sessions: {e}")
        return {
            "is_success": False,
            "message": str(e),
            "data": None
        }
    finally:
        if 'conn' in locals():
            release_postgres_conn(conn)

@app.get("/session/{session_id}")
def get_session_info(session_id: str):
    """Get session info with full message details including charts and token usage for frontend rendering"""
    try:
        conn = get_postgres_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get all messages with rich metadata including tokens
        cur.execute("""
            SELECT cm.session_id,cm.message_id, cm.role, cm.content, cm.sql_query, cm.filtered_sql, cm.chart_data, 
                   cm.result_data, cm.execution_time, cm.error, cm.langfuse_trace_id, 
                   cm.input_tokens, cm.output_tokens, cm.created_at, cs.session_status,
                   cm.updated_at
            FROM chat_messages cm
            JOIN chat_sessions cs ON cm.session_id = cs.session_id
            WHERE cm.session_id = %s 
            ORDER BY cm.created_at ASC
        """, (session_id,))
        
        db_messages = cur.fetchall()
        cur.close()
        release_postgres_conn(conn)
        
        messages = []
        total_session_tokens = 0
        for row in db_messages:
            msg = {
                "message_id": row["message_id"],
                "user_type": row["role"],
                "type": "user" if row["role"] == "user" else "assistant" if row["role"] == "assistant" else "summary",
                "content": row["content"]
            }
            
            # Add rich metadata for assistant messages
            if row["role"] == "assistant":
                if row.get("sql_query"):
                    msg["sql_query"] = row["sql_query"]
                if row.get("filtered_sql"):
                    msg["filtered_sql"] = row["filtered_sql"]
                if row.get("chart_data"):
                    msg["chart_data"] = row["chart_data"]  # Only structure, no data
                    msg["needs_regeneration"] = True  # Flag for frontend
                    msg["message_id"] = row["message_id"]  # For regenerate API call
                    msg["session_id"] = row.get("session_id")
                # Note: result_data is no longer saved, so we don't return it
                if row.get("execution_time"):
                    msg["execution_time"] = row["execution_time"]
                if row.get("error"):
                    msg["error"] = row["error"]
                if row.get("langfuse_trace_id"):
                    msg["trace_id"] = row["langfuse_trace_id"]
                msg["input_tokens"] = row.get("input_tokens") or 0
                msg["output_tokens"] = row.get("output_tokens") or 0
                msg["created_at"] = row.get("created_at").isoformat() if row.get("created_at") else None
                msg["updated_at"] = row.get("updated_at").isoformat() if row.get("updated_at") else None
                
            
            messages.append(msg)
        
        # Also check active memory for current state
        is_active = session_id in conversation_memories
        return {
            "is_success": True,
            "message": "success",
            "data": {
                "session_id": session_id,
                "session_status": db_messages[0]["session_status"] if db_messages else "Unknown",
                "total_messages": len(messages),
                "total_session_tokens": total_session_tokens,
                "messages": messages
            }
        }
    except Exception as e:
        logger.error(f"Failed to get session info: {e}")
        return {
            "is_success": False,
            "message": str(e),
            "data": None
        }

@app.delete("/session/{session_id}")
def clear_session_memory(session_id: str):
    """Clear session from memory and permanent storage"""
    logger.info(f"Clearing session {session_id} from memory and storage")
    try:
        # Remove from active memory
        if session_id in conversation_memories:
            del conversation_memories[session_id]
            logger.info(f"Removed session {session_id} from active memory")
        
        # Remove from permanent storage - deleting from chat_sessions will cascade delete messages
        conn = get_postgres_conn()
        cur = conn.cursor()
        
        # First count the messages that will be deleted
        cur.execute("SELECT COUNT(*) FROM chat_messages WHERE session_id = %s", (session_id,))
        message_count = cur.fetchone()[0]
        
        # Delete the session (this will cascade delete all messages)
        cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
        
        conn.commit()
        cur.close()
        
        logger.info(f"Session {session_id} cleared successfully, deleted {message_count} messages")
        return {
            "is_success": True,
            "message": "success",
            "data": {
                "message": "Session cleared from both memory and permanent storage",
                "session_id": session_id,
                "deleted_messages": message_count,
            }
        }
    except Exception as e:
        logger.error(f"Failed to clear session {session_id}: {e}")
        return {
            "is_success": False,
            "message": str(e),
            "data": None
        }
    finally:
        if 'conn' in locals():
            release_postgres_conn(conn)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "memory_system": "ConversationSummaryBufferMemory + PostgreSQL",
        "features": [
            "rbac",
            "conversation_summary_buffer",
            "automatic_summarization",
            "postgresql_permanent_storage",
            "langfuse"
        ],
        "active_memories": len(conversation_memories),
        "db": {
            "clickhouse": "iw_dev",
            "postgres": "chat_history"
        },
        "langfuse": {
            "enabled": langfuse_handler is not None,
            "host": os.getenv("LANGFUSE_HOST", "not set"),
            "public_key_set": bool(os.getenv("LANGFUSE_PUBLIC_KEY")),
            "secret_key_set": bool(os.getenv("LANGFUSE_SECRET_KEY"))
        }
    }

@app.get("/debug/langfuse")
def debug_langfuse():
    """Debug endpoint to check Langfuse configuration"""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "")
    
    return {
        "langfuse_handler_initialized": langfuse_handler is not None,
        "environment_variables": {
            "LANGFUSE_HOST": host if host else "NOT SET",
            "LANGFUSE_PUBLIC_KEY": f"{public_key[:20]}..." if public_key else "NOT SET",
            "LANGFUSE_SECRET_KEY": f"{secret_key[:10]}..." if secret_key else "NOT SET"
        },
        "recommendations": [] if langfuse_handler else [
            "Ensure LANGFUSE_SECRET_KEY is set in environment",
            "Ensure LANGFUSE_PUBLIC_KEY is set in environment",
            "Verify LANGFUSE_HOST is accessible from this container",
            "Check that Langfuse server is running and healthy"
        ]
    }

# ============== STARTUP/SHUTDOWN ==============

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    global langfuse_handler
    
    logger.info("Application startup initiated")
    
    init_connection_pools()
    init_conversation_tables()
    
    # Re-initialize Langfuse handler in case env vars are now available
    if langfuse_handler is None:
        logger.info("Re-attempting Langfuse initialization on startup...")
        try:
            langfuse_handler = CallbackHandler()
            logger.info("Langfuse handler initialized on startup")
        except Exception as e:
            logger.warning(f"Langfuse initialization failed on startup: {e}")
    
    logger.info("Application started successfully", extra={
        "event": "startup", 
        "features": ["connection_pooling", "conversation_summary_buffer_memory", "langfuse", "loki_logging"],
        "memory_system": "ConversationSummaryBufferMemory (k=10) + PostgreSQL persistence",
        "langfuse_enabled": langfuse_handler is not None
    })

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:5187",
        "http://localhost:5188",
        "http://192.168.29.31:5187",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global pg_pool, clickhouse_client
    
    if pg_pool:
        pg_pool.closeall()
    
    if clickhouse_client:
        clickhouse_client.close()
    
    logger.info("Application shutdown complete")

if __name__ == "__main__":
    init_connection_pools()
    init_conversation_tables()
    uvicorn.run("backend:app", host="0.0.0.0", port=8000,reload=True)
