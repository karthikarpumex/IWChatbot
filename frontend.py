import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="IronWorker Chatbot", page_icon="📊", layout="wide")

BACKEND_URL = "http://localhost:8000"

if "current_session" not in st.session_state:
    st.session_state.current_session = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "load_history" not in st.session_state:
    st.session_state.load_history = False

with st.sidebar:
    st.title("💬 Chat History")
    
    if st.button("🔄 Refresh Sessions"):
        st.rerun()
    
    try:
        response = requests.get(f"{BACKEND_URL}/sessions", timeout=5)
        if response.status_code == 200:
            sessions_data = response.json()
            sessions = sessions_data.get("sessions", [])
            
            if sessions:
                st.caption(f"📋 {len(sessions)} conversation(s)")
                
                for session in sessions:
                    session_id = session["session_id"]
                    first_question = session.get("first_question", "New conversation")
                    if first_question:
                        first_question = first_question[:50]
                    else:
                        first_question = "Empty conversation"
                    
                    message_count = session.get("message_count", 0)
                    last_message = session.get("last_message_at", "")
                    
                    try:
                        dt = datetime.fromisoformat(last_message.replace('Z', '+00:00'))
                        time_str = dt.strftime("%b %d, %I:%M %p")
                    except:
                        time_str = "Unknown"
                    
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        if st.button(
                            f"💬 {first_question}",
                            key=f"session_{session_id}",
                            help=f"{message_count} messages • {time_str}"
                        ):
                            st.session_state.current_session = session_id
                            st.session_state.load_history = True
                            st.rerun()
                    
                    with col2:
                        if st.button("🗑️", key=f"delete_{session_id}", help="Delete"):
                            try:
                                requests.delete(f"{BACKEND_URL}/session/{session_id}")
                                if st.session_state.current_session == session_id:
                                    st.session_state.current_session = None
                                    st.session_state.chat_messages = []
                                st.rerun()
                            except:
                                st.error("Delete failed")
                    
                    st.divider()
            else:
                st.info("No conversations yet")
        
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ Backend offline")
    except Exception as e:
        st.error(f"Error: {e}")
    
    st.divider()
    if st.button("➕ New Chat", type="primary"):
        st.session_state.current_session = None
        st.session_state.chat_messages = []
        st.rerun()

st.title("📊 IronWorker Analytics")

if st.session_state.load_history and st.session_state.current_session:
    try:
        response = requests.get(
            f"{BACKEND_URL}/session/{st.session_state.current_session}",
            timeout=5
        )
        if response.status_code == 200:
            history = response.json()
            messages = history.get("messages", [])
            
            st.session_state.chat_messages = []
            for msg in messages:
                st.session_state.chat_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "sql_query": msg.get("sql_query"),
                    "filtered_sql": msg.get("filtered_sql"),
                    "chart_data": msg.get("chart_data"),
                    "result_count": msg.get("result_count"),
                    "execution_time": msg.get("execution_time"),
                    "error": msg.get("error"),
                    "created_at": msg.get("created_at")
                })
        
        st.session_state.load_history = False
    except Exception as e:
        st.error(f"Failed to load history: {e}")
        st.session_state.load_history = False

for i, message in enumerate(st.session_state.chat_messages):
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])
            
            if message.get("filtered_sql"):
                with st.expander("🔍 SQL Query"):
                    st.code(message["filtered_sql"], language="sql")
            
            chart_data = message.get("chart_data")
            if chart_data and chart_data.get("chart_type"):
                try:
                    chart_type = chart_data.get("chart_type")
                    x_values = chart_data.get("x_values", [])
                    y_values = chart_data.get("y_values", [])
                    
                    if x_values and y_values:
                        if chart_type == "bar":
                            fig = go.Figure(data=[
                                go.Bar(x=x_values, y=y_values, marker_color='steelblue')
                            ])
                        elif chart_type == "line":
                            fig = go.Figure(data=[
                                go.Scatter(x=x_values, y=y_values, mode='lines+markers')
                            ])
                        elif chart_type == "pie":
                            fig = go.Figure(data=[
                                go.Pie(labels=x_values, values=y_values, hole=0.3)
                            ])
                        else:
                            fig = go.Figure(data=[
                                go.Scatter(x=x_values, y=y_values, mode='markers')
                            ])
                        
                        fig.update_layout(
                            title=chart_data.get("title", ""),
                            xaxis_title=chart_data.get("x_label", ""),
                            yaxis_title=chart_data.get("y_label", ""),
                            height=400,
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {e}")
            
            if message.get("result_count") is not None or message.get("execution_time"):
                metadata = []
                if message.get("result_count") is not None:
                    metadata.append(f"📊 {message['result_count']} rows")
                if message.get("execution_time"):
                    metadata.append(f"⏱️ {message['execution_time']:.2f}s")
                st.caption(" • ".join(metadata))

user_input = st.chat_input("Ask a question about members, training, certifications...")

if user_input:
    st.session_state.chat_messages.append({
        "role": "user",
        "content": user_input
    })
    
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                payload = {"question": user_input}
                
                if st.session_state.current_session:
                    payload["session_id"] = st.session_state.current_session
                
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get("session_id"):
                        st.session_state.current_session = data["session_id"]
                    
                    if data.get("error"):
                        error_msg = f"❌ Error: {data['error']}"
                        st.error(error_msg)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "error": data["error"]
                        })
                    else:
                        summary = data.get("summary", "Query completed successfully")
                        st.write(summary)
                        
                        if data.get("filtered_sql"):
                            with st.expander("🔍 SQL Query"):
                                st.code(data["filtered_sql"], language="sql")
                        
                        chart_data = data.get("chart")
                        if chart_data and chart_data.get("chart_type"):
                            try:
                                x_values = chart_data.get("x_values", [])
                                y_values = chart_data.get("y_values", [])
                                
                                if x_values and y_values:
                                    chart_type = chart_data["chart_type"]
                                    
                                    if chart_type == "bar":
                                        fig = go.Figure(data=[
                                            go.Bar(x=x_values, y=y_values, marker_color='steelblue')
                                        ])
                                    elif chart_type == "line":
                                        fig = go.Figure(data=[
                                            go.Scatter(x=x_values, y=y_values, mode='lines+markers')
                                        ])
                                    elif chart_type == "pie":
                                        fig = go.Figure(data=[
                                            go.Pie(labels=x_values, values=y_values, hole=0.3)
                                        ])
                                    else:
                                        fig = go.Figure(data=[
                                            go.Scatter(x=x_values, y=y_values, mode='markers')
                                        ])
                                    
                                    fig.update_layout(
                                        title=chart_data.get("title", ""),
                                        xaxis_title=chart_data.get("x_label", ""),
                                        yaxis_title=chart_data.get("y_label", ""),
                                        height=400,
                                        template="plotly_white"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Chart error: {e}")
                        
                        if data.get("data"):
                            with st.expander(f"📄 Data ({len(data['data'])} rows)"):
                                df = pd.DataFrame(data["data"])
                                st.dataframe(df, use_container_width=True)
                                
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    "⬇️ Download CSV",
                                    csv,
                                    "data.csv",
                                    "text/csv",
                                    key=f"download_{len(st.session_state.chat_messages)}"
                                )
                        else:
                            st.info("No data found for this query.")
                        
                        if data.get("execution_time"):
                            st.caption(f"⏱️ Completed in {data['execution_time']}")
                        
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": summary,
                            "sql_query": data.get("original_sql"),
                            "filtered_sql": data.get("filtered_sql"),
                            "chart_data": chart_data,
                            "result_count": len(data.get("data", [])),
                            "execution_time": float(data.get("execution_time", "0").replace("s", ""))
                        })
                
                else:
                    error_msg = f"❌ Server error: {response.status_code}"
                    st.error(error_msg)
                    
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "error": f"HTTP {response.status_code}"
                    })
            
            except requests.exceptions.ConnectionError:
                error_msg = "⚠️ Backend not running! Start with: `python chat_storage_backend.py`"
                st.error(error_msg)
                
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "error": "Connection refused"
                })
            
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "error": str(e)
                })
    
    st.rerun()

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🔒 Role: Admin")

with col2:
    if st.session_state.current_session:
        st.caption(f"💬 Session: {st.session_state.current_session[:8]}...")
    else:
        st.caption("💬 New conversation")

with col3:
    st.caption(f"📊 Messages: {len(st.session_state.chat_messages)}")