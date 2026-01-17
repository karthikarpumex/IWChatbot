import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import uuid
import time

st.set_page_config(page_title="IronWorker Analytics", page_icon="📊", layout="wide")

BACKEND_URL = "http://localhost:8000"

# Initialize simple session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "current_conversation_title" not in st.session_state:
    st.session_state.current_conversation_title = None

if "session_loaded" not in st.session_state:
    st.session_state.session_loaded = False

if "new_chat_started" not in st.session_state:
    st.session_state.new_chat_started = False

# Load conversation history from backend on first load
def load_session_from_backend(session_id):
    """Load conversation history from backend database"""
    try:
        response = requests.get(f"{BACKEND_URL}/session/{session_id}", timeout=5)
        if response.status_code == 200:
            response_data = response.json()
            # Handle new backend response structure
            if response_data.get("is_success") and response_data.get("data"):
                session_data = response_data["data"]
                messages = []  # Initialize messages here
                if session_data.get("messages"):
                    for msg in session_data["messages"]:
                        # Backend now returns rich message data
                        msg_type = msg.get("type", "user")
                        if msg_type == "user":
                            messages.append({
                                "role": "user", 
                                "content": msg["content"]
                            })
                        elif msg_type == "assistant":
                            # Build rich assistant message with all metadata
                            assistant_msg = {
                                "role": "assistant", 
                                "content": msg["content"]
                            }
                            
                            # Add rich metadata if available
                            if msg.get("chart_data"):
                                assistant_msg["chart_data"] = msg["chart_data"]
                            if msg.get("sql_query"):
                                assistant_msg["sql_query"] = msg["sql_query"]
                            if msg.get("filtered_sql"):
                                assistant_msg["filtered_sql"] = msg["filtered_sql"]
                            if msg.get("data"):
                                assistant_msg["data"] = msg["data"]
                            if msg.get("execution_time"):
                                assistant_msg["execution_time"] = msg["execution_time"]
                                
                            messages.append(assistant_msg)
                        # Skip "summary" type messages for display
                st.session_state.chat_messages = messages
                st.session_state.current_conversation_title = messages[0]["content"][:50] if messages else None
                return True
        return False
    except Exception as e:
        st.error(f"Failed to load session: {e}")
        return False

def get_available_sessions():
    """Get list of available sessions from backend with caching"""
    try:
        # Use cached sessions if available and fresh (within 5 seconds)
        if "cached_sessions" in st.session_state and "cache_time" in st.session_state:
            if time.time() - st.session_state.cache_time < 5:
                return st.session_state.cached_sessions
        
        # Fetch fresh data with user_id parameter
        response = requests.get(f"{BACKEND_URL}/sessions?user_id=cda977a7-7bc5-4007-9316-dc315a0037c0", timeout=5)
        if response.status_code == 200:
            response_data = response.json()
            # Handle new backend response structure
            if response_data.get("is_success") and response_data.get("data"):
                data = response_data["data"]
                sessions = data.get("sessions", [])
                # Cache the results
                st.session_state.cached_sessions = sessions
                st.session_state.cache_time = time.time()
                return sessions
        return []
    except Exception:
        return []

def delete_session(session_id):
    """Delete a session from backend"""
    try:
        response = requests.delete(f"{BACKEND_URL}/session/{session_id}", timeout=5)
        if response.status_code == 200:
            response_data = response.json()
            # Handle new backend response structure
            return response_data.get("is_success", False)
        return False
    except Exception:
        return False

# Load session on first run or if not loaded (but skip if new chat just started)
if not st.session_state.session_loaded and not st.session_state.new_chat_started:
    # Try to load current session from backend
    if load_session_from_backend(st.session_state.session_id):
        st.session_state.session_loaded = True
    else:
        # If no history for current session, try to get latest session
        available_sessions = get_available_sessions()
        if available_sessions:
            # Use most recent session
            latest_session = available_sessions[0]  # Sessions are ordered by last_message DESC
            st.session_state.session_id = latest_session["session_id"]
            load_session_from_backend(st.session_state.session_id)
        st.session_state.session_loaded = True

# Reset new chat flag after session loading logic
if st.session_state.new_chat_started:
    st.session_state.new_chat_started = False

# Sidebar with business-focused navigation
with st.sidebar:
    st.title("🔧 IronWorker Chatbot")
    st.markdown("### Chat Dashboard")
    
    # New chat button
    new_chat_clicked = st.button("💬 New Chat", type="primary", use_container_width=True)
    
    if new_chat_clicked:
        # Save current conversation if it has messages
        if st.session_state.chat_messages:
            # Get title from first user message
            first_user_msg = next((msg["content"] for msg in st.session_state.chat_messages if msg.get("role") == "user"), "New Conversation")
            title = first_user_msg[:50] + "..." if len(first_user_msg) > 50 else first_user_msg
            
            # Save to history
            st.session_state.conversation_history.append({
                "id": st.session_state.session_id,
                "title": title,
                "messages": st.session_state.chat_messages.copy(),
                "timestamp": f"{len(st.session_state.conversation_history) + 1}"
            })
        
        # Start new conversation
        st.session_state.chat_messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.current_conversation_title = None
        st.session_state.session_loaded = True  # Mark as loaded to prevent auto-loading
        st.session_state.new_chat_started = True  # Flag to indicate new chat
        
        # Force page refresh with clearer feedback
        st.success("🆕 Starting new chat session...")
        st.info("✨ Chat cleared! Ask me anything about IronWorker data...")
        time.sleep(0.5)  # Brief pause for visual feedback
        st.rerun()
    
    st.divider()
    
    # Backend Session Management
    available_sessions = get_available_sessions()
    
    if available_sessions:
        st.markdown("### 🗂️ Recent Sessions")
        for session in available_sessions[:5]:  # Show last 5 sessions
            session_id = session["session_id"]
            title = session.get("first_question", "New Conversation")
            msg_count = session.get("message_count", 0)
            
            # Truncate title
            display_title = title[:30] + "..." if len(title) > 30 else title
            
            # Create columns for session button and delete button
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Render session without green indicator for current session
                if session_id == st.session_state.session_id:
                    st.markdown(f"**{display_title}** ({msg_count} msgs)")
                else:
                    if st.button(f"📄 {display_title} ({msg_count} msgs)", key=f"load_session_{session_id}", use_container_width=True):
                        # Load selected session
                        st.session_state.session_id = session_id
                        if load_session_from_backend(session_id):
                            st.session_state.session_loaded = True
                            st.rerun()
            
            with col2:
                # Don't allow deleting current active session
                if session_id != st.session_state.session_id:
                    if st.button("🗿️", key=f"delete_{session_id}", help="Delete session", use_container_width=True):
                        # Delete session
                        if delete_session(session_id):
                            st.success(f"Session deleted!")
                            # Clear any cached session data
                            if "cached_sessions" in st.session_state:
                                del st.session_state.cached_sessions
                            # Force immediate refresh
                            st.rerun()
    
    st.divider()
    
    # Conversation History
    if st.session_state.conversation_history:
        st.markdown("### Recent Chats")
        
        # Remove duplicates and show last 5 conversations
        seen_ids = set()
        unique_conversations = []
        for conv in reversed(st.session_state.conversation_history):
            if conv['id'] not in seen_ids:
                seen_ids.add(conv['id'])
                unique_conversations.append(conv)
            if len(unique_conversations) >= 5:
                break
        
        for i, conv in enumerate(unique_conversations):
            if st.button(f"📝 {conv['title']}", key=f"load_conv_{i}_{conv['id']}", use_container_width=True):
                # Save current conversation before switching if it has messages
                if st.session_state.chat_messages and st.session_state.session_id != conv['id']:
                    first_user_msg = next((msg["content"] for msg in st.session_state.chat_messages if msg.get("role") == "user"), "New Conversation")
                    title = first_user_msg[:50] + "..." if len(first_user_msg) > 50 else first_user_msg
                    
                    # Update or add current conversation
                    existing_conv = next((c for c in st.session_state.conversation_history if c['id'] == st.session_state.session_id), None)
                    if existing_conv:
                        existing_conv['messages'] = st.session_state.chat_messages.copy()
                        existing_conv['title'] = title
                    else:
                        st.session_state.conversation_history.append({
                            "id": st.session_state.session_id,
                            "title": title,
                            "messages": st.session_state.chat_messages.copy(),
                            "timestamp": f"{len(st.session_state.conversation_history) + 1}"
                        })
                
                # Load selected conversation
                st.session_state.chat_messages = conv['messages'].copy()
                st.session_state.session_id = conv['id']
                st.session_state.current_conversation_title = conv['title']
                st.rerun()
        
        st.divider()

st.title("📊 IronWorker Chatbot")

# Simple welcome message for new users
if not st.session_state.chat_messages:
    st.info("👋 Welcome! Ask me anything about union members, training, certifications, or employment data. I'll provide detailed analytics and visualizations.")

for i, message in enumerate(st.session_state.chat_messages):
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            chart_data = message.get("chart_data")
            summary = chart_data.get("summary") if chart_data else None
            chart_type = chart_data.get("chart_type") if chart_data else None

            # Show summary unless it's a 'no chart needed' message and no chart is shown
            suppress_phrases = [
                "a chart is not necessary",
                "no chart is necessary",
                "not necessary to represent this information",
                "no chart needed"
            ]
            show_summary = True
            if summary and chart_type is None:
                for phrase in suppress_phrases:
                    if phrase in summary.lower():
                        show_summary = False
                        break
            if summary and show_summary:
                st.write(summary)

            if chart_data and chart_type:
                try:
                    if chart_type == "table":
                        st.subheader(chart_data.get("title", "Data Table"))
                        if message.get("data"):
                            df = pd.DataFrame(message["data"])
                            st.dataframe(df, use_container_width=True, height=400)
                            csv = df.to_csv(index=False)
                            st.download_button(
                                "⬇️ Download Table Data",
                                csv,
                                f"{chart_data.get('title', 'data').replace(' ', '_').lower()}.csv",
                                "text/csv",
                                key=f"table_download_{id(message)}"
                            )
                    elif chart_type == "multi_line":
                        series_data = chart_data.get("series_data", {})
                        x_values = chart_data.get("x_values", [])
                        if series_data and x_values:
                            fig = go.Figure()
                            for series_name, y_values in series_data.items():
                                fig.add_trace(go.Scatter(
                                    x=x_values,
                                    y=y_values,
                                    mode='lines+markers',
                                    name=series_name,
                                    line=dict(width=3),
                                    marker=dict(size=6)
                                ))
                            fig.update_layout(
                                title=chart_data.get("title", "Multi-Line Chart"),
                                xaxis_title=chart_data.get("x_label", ""),
                                yaxis_title=chart_data.get("y_label", ""),
                                height=500,
                                template="plotly_white",
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                ),
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"chart_multiline_{i}_{id(message)}")
                    else:
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
                            st.plotly_chart(fig, use_container_width=True, key=f"chart_history_{i}_{id(message)}")
                except Exception as e:
                    st.error(f"Chart error: {e}")
            # Only show execution time, not row count
            if message.get("execution_time"):
                st.caption(f"⏱️ {message['execution_time']:.2f}s")

user_input = st.chat_input("Ask a question about members, training, certifications...")

if user_input:
    # Add user message to current conversation
    st.session_state.chat_messages.append({
        "role": "user",
        "content": user_input
    })
    
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                payload = {
                    "question": user_input,
                    "session_id": st.session_state.session_id
                }
                
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Handle new backend response structure
                    if not response_data.get("is_success"):
                        error_msg = f"❌ Error: {response_data.get('message', 'Unknown error')}"
                        st.error(error_msg)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "error": response_data.get("message")
                        })
                    else:
                        # Extract data from the wrapped response
                        data = response_data.get("data", {})
                        if data.get("error"):
                            error_msg = f"❌ Error: {data['error']}"
                            st.error(error_msg)
                            st.session_state.chat_messages.append({
                                "role": "assistant",
                                "content": error_msg,
                                "error": data["error"]
                            })
                        else:
                            # Backend returns content, not summary
                            summary = data.get("content", "Query completed successfully")
                            st.write(summary)
                            
                            # SQL query hidden for client demo
                            # if data.get("filtered_sql"):
                            #     with st.expander("🔍 SQL Query"):
                            #         st.code(data["filtered_sql"], language="sql")
                            
                            # Backend returns chart_data instead of chart
                            chart_data = data.get("chart_data")
                            if chart_data and chart_data.get("chart_type"):
                                try:
                                    chart_type = chart_data["chart_type"]
                                    
                                    if chart_type == "table":
                                        # Display data in enhanced table format
                                        st.subheader(chart_data.get("title", "Data Table"))
                                        if data.get("data"):
                                            df = pd.DataFrame(data["data"])
                                            st.dataframe(df, use_container_width=True, height=400)
                                            
                                            # Add download option
                                            csv = df.to_csv(index=False)
                                        st.download_button(
                                            "⬇️ Download Table Data",
                                            csv,
                                            f"{chart_data.get('title', 'data').replace(' ', '_').lower()}.csv",
                                            "text/csv",
                                            key=f"new_table_download_{len(st.session_state.chat_messages)}"
                                        )
                                    elif chart_type == "multi_line":
                                        # Handle multi-line charts with series_data
                                        series_data = chart_data.get("series_data", {})
                                        x_values = chart_data.get("x_values", [])
                                        
                                        if series_data and x_values:
                                            fig = go.Figure()
                                            
                                            # Add each series as a separate line
                                            for series_name, y_values in series_data.items():
                                                fig.add_trace(go.Scatter(
                                                    x=x_values,
                                                    y=y_values,
                                                    mode='lines+markers',
                                                    name=series_name,
                                                    line=dict(width=3),
                                                    marker=dict(size=6)
                                                ))
                                            
                                            fig.update_layout(
                                                title=chart_data.get("title", "Multi-Line Chart"),
                                                xaxis_title=chart_data.get("x_label", ""),
                                                yaxis_title=chart_data.get("y_label", ""),
                                                height=500,
                                                template="plotly_white",
                                                legend=dict(
                                                    orientation="h",
                                                    yanchor="bottom",
                                                    y=1.02,
                                                    xanchor="right",
                                                    x=1
                                                ),
                                                hovermode='x unified'
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True, key=f"chart_new_multiline_{len(st.session_state.chat_messages)}")
                                    else:
                                        # Handle regular chart types
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
                                        
                                        st.plotly_chart(fig, use_container_width=True, key=f"chart_new_regular_{len(st.session_state.chat_messages)}")
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
                                "sql_query": data.get("sql_query"),  # Fixed field name
                                "filtered_sql": data.get("filtered_sql"),
                                "chart_data": chart_data,
                                "data": data.get("data", []),  # Add data to session state
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

# Business footer for client demo
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🔒 Secure Access")

with col2:
    st.caption("📊 Real-time Analytics")

with col3:
    if len(st.session_state.chat_messages) > 0:
        user_questions = len([msg for msg in st.session_state.chat_messages if msg.get("role") == "user"])
        st.caption(f"💬 {user_questions} questions answered")
    else:
        st.caption("💬 Ready to help")