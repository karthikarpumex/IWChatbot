from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, Optional, Any
import uvicorn 
import clickhouse_connect
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime
from dotenv import load_dotenv
from langfuse import observe
from langfuse.langchain import CallbackHandler
import sqlglot
from sqlglot import exp, parse_one, errors
import uuid

load_dotenv()
app = FastAPI()

langfuse_handler = CallbackHandler()

def get_clickhouse_db():
    return clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST", "localhost"),
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        username=os.getenv("CLICKHOUSE_USER", "default"),
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
        database=os.getenv("CLICKHOUSE_DATABASE", "iw_dev")
    )

def get_postgres_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        database=os.getenv("POSTGRES_DB", "chat_history"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres")
    )

def init_chat_tables():
    conn = get_postgres_conn()
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id UUID PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            user_role VARCHAR(50) NOT NULL,
            local_union_id VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_message_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            message_id SERIAL PRIMARY KEY,
            session_id UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
            role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            sql_query TEXT,
            filtered_sql TEXT,
            chart_data JSONB,
            result_count INTEGER,
            execution_time FLOAT,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cur.execute("CREATE INDEX IF NOT EXISTS idx_session_messages ON chat_messages(session_id, created_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions ON chat_sessions(user_id, created_at DESC)")
    
    conn.commit()
    cur.close()
    conn.close()

def create_chat_session(user_id: str, user_role: str, local_union_id: Optional[str] = None) -> str:
    conn = get_postgres_conn()
    cur = conn.cursor()
    
    session_id = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO chat_sessions (session_id, user_id, user_role, local_union_id) VALUES (%s, %s, %s, %s)",
        (session_id, user_id, user_role, local_union_id)
    )
    
    conn.commit()
    cur.close()
    conn.close()
    
    return session_id

def save_message(
    session_id: str,
    role: str,
    content: str,
    sql_query: Optional[str] = None,
    filtered_sql: Optional[str] = None,
    chart_data: Optional[dict] = None,
    result_count: Optional[int] = None,
    execution_time: Optional[float] = None,
    error: Optional[str] = None
):
    conn = get_postgres_conn()
    cur = conn.cursor()
    
    import json
    chart_json = json.dumps(chart_data) if chart_data else None
    
    cur.execute(
        """INSERT INTO chat_messages 
        (session_id, role, content, sql_query, filtered_sql, chart_data, result_count, execution_time, error)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (session_id, role, content, sql_query, filtered_sql, chart_json, result_count, execution_time, error)
    )
    
    cur.execute(
        "UPDATE chat_sessions SET last_message_at = CURRENT_TIMESTAMP WHERE session_id = %s",
        (session_id,)
    )
    
    conn.commit()
    cur.close()
    conn.close()

def get_session_history(session_id: str, limit: int = 10) -> list[dict]:
    conn = get_postgres_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute(
        """SELECT role, content, sql_query, filtered_sql, created_at
        FROM chat_messages WHERE session_id = %s
        ORDER BY created_at DESC LIMIT %s""",
        (session_id, limit)
    )
    
    messages = cur.fetchall()
    cur.close()
    conn.close()
    
    return [dict(msg) for msg in reversed(messages)]

def get_user_sessions(user_id: str, limit: int = 20) -> list[dict]:
    conn = get_postgres_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute(
        """SELECT 
            s.session_id, s.user_role, s.local_union_id, s.created_at, s.last_message_at,
            COUNT(m.message_id) as message_count,
            (SELECT content FROM chat_messages 
             WHERE session_id = s.session_id AND role = 'user' 
             ORDER BY created_at ASC LIMIT 1) as first_question
        FROM chat_sessions s
        LEFT JOIN chat_messages m ON s.session_id = m.session_id
        WHERE s.user_id = %s
        GROUP BY s.session_id
        ORDER BY s.last_message_at DESC
        LIMIT %s""",
        (user_id, limit)
    )
    
    sessions = cur.fetchall()
    cur.close()
    conn.close()
    
    return [dict(session) for session in sessions]

def delete_session(session_id: str):
    conn = get_postgres_conn()
    cur = conn.cursor()
    
    cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
    
    conn.commit()
    cur.close()
    conn.close()

TABLE_DATES = {
    "drugtestrecords": (2015, 2025),
    "member_course_history": (1956, 2025),
    "member_national_fund_trainings": (1985, 2025),
    "memberunionhistory1": (1986, 2025),
    "member_certifications": (1900, 2241),
    "member_online_registration_courses": (2012, 2025),
    "trainingprogramcertificates": (2006, 2025),
}

SENSITIVE_COLUMNS = {"ssn", "sin", "phone1", "phone4"}

class UserRole(BaseModel):
    role: Literal["admin", "local_union_officer", "ironworker"]
    user_id: str
    local_union_id: Optional[str] = None

PRIMARY_TABLES = ["iw_contact08", "memberunionhistory1"]

LOCAL_TABLE_FIELDS = {
    "iw_contact08": "localunionid",
    "memberunionhistory1": "localunionid",
    "drugtestrecords": "local_union_number__c",
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
}

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

class SQLQuery(BaseModel):
    sql: str
    tables: list[str]
    year: str | None = None
    can_chart: bool = True

class ChartData(BaseModel):
    chart_type: Literal["bar", "line", "pie", "scatter"] | None = None
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    x_values: list[str] = []
    y_values: list[int | float] = []
    summary: str

class State(TypedDict):
    question: str
    user_role: UserRole
    session_id: str
    sql_query: SQLQuery | None
    filtered_sql: str | None
    filter_metadata: dict | None
    validation_result: dict | None
    date_valid: bool
    error: str | None
    data: list[dict]
    chart: ChartData | None
    denied_columns: set[str]
    conversation_history: list[dict]

llm = ChatOpenAI(model="gpt-4o", temperature=0)
sql_agent = llm.with_structured_output(SQLQuery)
chart_agent = llm.with_structured_output(ChartData)

@observe(name="sql-agent")
def call_sql_agent(prompt: str) -> SQLQuery:
    return sql_agent.invoke(prompt, config={"callbacks": [langfuse_handler]})

@observe(name="chart-agent")
def call_chart_agent(prompt: str) -> ChartData:
    return chart_agent.invoke(prompt, config={"callbacks": [langfuse_handler]})

def build_context_string(history: list[dict]) -> str:
    if not history:
        return ""
    
    context_parts = []
    for msg in history[-6:]:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        
        if role == 'user':
            context_parts.append(f"User asked: {content}")
        elif role == 'assistant':
            sql = msg.get('filtered_sql') or msg.get('sql_query')
            if sql:
                context_parts.append(f"Assistant queried: {sql[:200]}...")
            context_parts.append(f"Assistant replied: {content[:200]}...")
    
    return "\n".join(context_parts)

def generate_sql(state: State) -> State:
    base_question = state['question']
    
    context_section = ""
    if state.get('conversation_history'):
        context_str = build_context_string(state['conversation_history'])
        if context_str:
            context_section = f"""
CONVERSATION HISTORY:
{context_str}

Extract relevant filters from history and apply to new query.
"""
    
    prompt = f"""{context_section}Generate ClickHouse SQL for: {base_question}

TABLES:
- iw_contact08: Master (userid, firstname, lastname, membernumber, memberstatusname, localunionid, statename, city, email, dateofbirth)
- drugtestrecords: Tests (active_latest_userid, test_status__c, drug_test_completion_date__c, local_union_number__c)
- member_course_history: Training (active_latest_userid, coursename, passed, startdate, enddate, year)
- member_certifications: Certs (active_latest_userid, certification_name__c, create_date__c, expire_date__c)
- memberunionhistory1: History (active_latest_userid, memberstatusname, paidthru)
- member_national_fund_trainings: National (active_latest_userid, coursecode, classname, startdate, enddate, grade)
- trainingprogramcertificates: Advanced (active_latest_userid, training_cert_name__c, cert_date__c, expire_date__c)

RULES:
- Use iw_contact08 as primary
- String columns need quotes: localunionid = '782'
- Dates are STRING 'YYYY-MM-DD'
- JOIN: iw_contact08.userid = other_table.active_latest_userid
- Inherit filters from context

OUTPUT: sql, tables, year, can_chart"""
    
    state['sql_query'] = call_sql_agent(prompt)
    return state

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
        state['error'] = f"Filter application failed: {str(e)}"
    
    return state

def validate_filter(state: State) -> State:
    if not state['sql_query'] or not state['filtered_sql'] or state['error']:
        return state
    
    try:
        select = parse_select(state['filtered_sql'])
        ensure_required_predicates(select, state['user_role'])
        validated_sql = validate_and_format_clickhouse_sql(state['filtered_sql'])
        state['filtered_sql'] = validated_sql
        state['validation_result'] = {'valid': True, 'issues': []}
    except ValueError as e:
        state['error'] = f"Validation failed: {str(e)}"
        state['validation_result'] = {'valid': False, 'issues': [str(e)]}
    
    return state

def validate_dates(state: State) -> State:
    if state['error']:
        return state
    
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

def run_query(state: State) -> State:
    if not state['date_valid'] or state['error']:
        return state
    
    try:
        client = get_clickhouse_db()
        result = client.query(state['filtered_sql'])
        rows = [dict(zip(result.column_names, row)) for row in result.result_rows]
        state['data'] = redact_rows(rows, state['denied_columns'])
    except Exception as e:
        state['error'] = f"Query failed: {str(e)}"
    
    return state

def create_chart(state: State) -> State:
    if not state['data'] or state['error']:
        return state
    
    can_chart = state['sql_query'] and state['sql_query'].can_chart and len(state['data']) > 2
    columns = list(state['data'][0].keys()) if state['data'] else []
    
    prompt = f"""Analyze data and provide response.
Question: {state['question']}
Columns: {columns}
Data ({len(state['data'])} rows): {state['data'][:15]}
Create chart: {can_chart}
OUTPUT: ChartData JSON"""
    
    try:
        state['chart'] = call_chart_agent(prompt)
    except Exception as e:
        state['error'] = f"Analysis failed: {str(e)}"
    
    return state

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

graph = workflow.compile()

class Query(BaseModel):
    question: str
    session_id: Optional[str] = None

@observe(name="chatbot-query")
def run_chatbot_query(question: str, user_role: UserRole, session_id: str) -> dict:
    denied_columns = set() if user_role.role == "admin" else SENSITIVE_COLUMNS
    conversation_history = get_session_history(session_id, limit=10)
    
    result = graph.invoke({
        "question": question,
        "user_role": user_role,
        "session_id": session_id,
        "sql_query": None,
        "filtered_sql": None,
        "filter_metadata": None,
        "validation_result": None,
        "date_valid": True,
        "error": None,
        "data": [],
        "chart": None,
        "denied_columns": denied_columns,
        "conversation_history": conversation_history
    })
    return result

@app.post("/query")
def process(q: Query):
    start_time = datetime.now()
    
    user_role = UserRole(
        role="admin",
        user_id="cda977a7-7bc5-4007-9316-dc315a0037c0",
        local_union_id="782"
    )
    
    if q.session_id:
        session_id = q.session_id
    else:
        session_id = create_chat_session(
            user_role.user_id,
            user_role.role,
            user_role.local_union_id
        )
    
    save_message(session_id, "user", q.question)
    
    try:
        result = run_chatbot_query(q.question, user_role, session_id)
        duration = (datetime.now() - start_time).total_seconds()
        
        if result['error']:
            save_message(session_id, "assistant", f"Error: {result['error']}", error=result['error'])
            return {"error": result['error'], "session_id": session_id}
        
        chart_data = result['chart'].model_dump() if result['chart'] else None
        summary = chart_data.get('summary') if chart_data else "Query completed successfully"
        
        save_message(
            session_id,
            "assistant",
            summary,
            sql_query=result['sql_query'].sql if result['sql_query'] else None,
            filtered_sql=result['filtered_sql'],
            chart_data=chart_data,
            result_count=len(result['data']),
            execution_time=duration
        )
        
        return {
            "session_id": session_id,
            "original_sql": result['sql_query'].sql if result['sql_query'] else None,
            "filtered_sql": result['filtered_sql'],
            "summary": summary,
            "chart": chart_data,
            "data": result['data'],
            "execution_time": f"{duration:.2f}s"
        }
    except Exception as e:
        save_message(session_id, "assistant", f"Error: {str(e)}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
def get_sessions(user_id: Optional[str] = None):
    if not user_id:
        user_id = "cda977a7-7bc5-4007-9316-dc315a0037c0"
    
    sessions = get_user_sessions(user_id)
    return {"sessions": sessions}

@app.get("/session/{session_id}")
def get_session(session_id: str):
    conn = get_postgres_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute(
        """SELECT message_id, role, content, sql_query, filtered_sql, chart_data, 
        result_count, execution_time, error, created_at
        FROM chat_messages WHERE session_id = %s ORDER BY created_at ASC""",
        (session_id,)
    )
    
    messages = cur.fetchall()
    cur.close()
    conn.close()
    
    return {"session_id": session_id, "messages": [dict(msg) for msg in messages]}

@app.delete("/session/{session_id}")
def remove_session(session_id: str):
    delete_session(session_id)
    return {"message": "Session deleted", "session_id": session_id}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "features": ["rbac", "chat_storage", "langfuse"],
        "db": {"clickhouse": "iw_dev", "postgres": "chat_history"}
    }

if __name__ == "__main__":
    init_chat_tables()
    uvicorn.run(app, host="0.0.0.0", port=8000)