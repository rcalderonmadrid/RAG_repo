import streamlit as st
import os
from pathlib import Path
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="🥀🛰️💻 RAG Intelligence Hub",
    page_icon="🥀🛰️💻",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/rcalderonmadrid/RAG_repo',
        'Report a bug': 'https://github.com/rcalderonmadrid/RAG_repo/issues',
        'About': "# RAG Intelligence Hub\nAdvanced Document Intelligence & Conversational AI System"
    }
)

from rag_engine import RAGEngine
from session_init import init_session_state, ensure_session_state
import utils

def apply_custom_css():
    """Apply custom CSS for better UX/UI"""
    st.markdown("""
    <style>
        /* Import modern fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Main app styling */
        .main {
            font-family: 'Inter', sans-serif;
        }

        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }

        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .main-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.2rem;
            opacity: 0.9;
        }

        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #f8f9ff 0%, #ffffff 100%);
        }

        /* Card styling */
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
            margin-bottom: 1rem;
        }

        .chat-message {
            padding: 1rem 1.5rem;
            border-radius: 12px;
            margin: 0.5rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: 20%;
        }

        .assistant-message {
            background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
            border-left: 4px solid #667eea;
            margin-right: 20%;
        }

        /* Status indicators */
        .status-success {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 500;
        }

        .status-warning {
            background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 500;
        }

        .status-error {
            background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 500;
        }

        /* Button styling */
        .stButton > button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }

        /* File uploader styling */
        .stFileUploader {
            border: 2px dashed #667eea;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
        }

        .stFileUploader:hover {
            border-color: #764ba2;
            background: rgba(102, 126, 234, 0.05);
        }

        /* Progress bars */
        .stProgress > div > div > div {
            background: linear-gradient(45deg, #667eea, #764ba2);
        }

        /* Selectbox and input styling */
        .stSelectbox, .stTextInput, .stTextArea {
            border-radius: 8px;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
            border-radius: 12px;
            padding: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            font-weight: 500;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
            border-radius: 8px;
            font-weight: 500;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 2rem;
            }

            .user-message, .assistant-message {
                margin-left: 0;
                margin-right: 0;
            }
        }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar navigation and system status"""
    with st.sidebar:
        st.markdown("### 🧭 Navigation")

        page = st.selectbox(
            "Choose a page:",
            ["📊 Dashboard", "💬 Chat", "📁 Documents", "💾 Conversations", "📈 Analytics", "⚙️ Settings"],
            key="page_selector"
        )

        st.divider()

        st.markdown("### 📊 System Status")

        # Initialize RAG engine if not already done
        if st.session_state.rag_engine is None:
            if st.button("🚀 Initialize RAG System", type="primary"):
                with st.spinner("Initializing RAG system..."):
                    try:
                        st.session_state.rag_engine = RAGEngine(st.session_state.config)
                        st.success("✅ RAG system initialized!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error initializing RAG: {str(e)}")

        # System status indicators
        if st.session_state.rag_engine:
            st.markdown('<div class="status-success">🟢 RAG System: Online</div>', unsafe_allow_html=True)

            # Document count
            doc_count = st.session_state.doc_manager.get_document_count()
            st.metric("📚 Documents", doc_count)

            # Conversation count
            conv_count = st.session_state.conv_manager.get_conversation_count()
            st.metric("💬 Conversations", conv_count)

        else:
            st.markdown('<div class="status-warning">🟡 RAG System: Offline</div>', unsafe_allow_html=True)

        st.divider()

        # Quick actions
        st.markdown("### ⚡ Quick Actions")

        if st.button("🔄 Refresh System"):
            st.rerun()

        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")

        # System info
        st.markdown("### ℹ️ System Info")
        with st.expander("View Details"):
            st.json({
                "LLM Model": st.session_state.config.llm_model,
                "Embedding Model": st.session_state.config.embedding_model,
                "Vector DB": st.session_state.config.persist_directory,
                "Temperature": st.session_state.config.temperature
            })

        return page.split(" ", 1)[1]

def render_dashboard():
    """Render main dashboard"""
    st.markdown("""
    <div class="main-header">
        <h1>🥀🛰️💻 RAG Intelligence Hub</h1>
        <p>Advanced Document Intelligence & Conversational AI Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        doc_count = st.session_state.doc_manager.get_document_count()
        st.metric("📚 Total Documents", doc_count, delta=None)

    with col2:
        conv_count = st.session_state.conv_manager.get_conversation_count()
        st.metric("💬 Conversations", conv_count, delta=None)

    with col3:
        if st.session_state.rag_engine:
            chunk_count = st.session_state.rag_engine.get_chunk_count()
            st.metric("🧩 Document Chunks", chunk_count, delta=None)
        else:
            st.metric("🧩 Document Chunks", "N/A")

    with col4:
        total_queries = len(st.session_state.conversation_history)
        st.metric("🔍 Total Queries", total_queries, delta=None)

    st.divider()

    # Recent activity and quick start
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🕒 Recent Activity")

        if st.session_state.conversation_history:
            recent_conversations = st.session_state.conv_manager.get_recent_conversations(5)
            for conv in recent_conversations:
                with st.expander(f"💬 {conv['title'][:50]}..."):
                    st.write(f"**Date:** {conv['timestamp']}")
                    st.write(f"**Messages:** {len(conv['messages'])}")
                    if st.button(f"🔗 Open", key=f"open_{conv['id']}"):
                        st.session_state.current_conversation = conv['messages']
                        st.switch_page("pages/chat.py")
        else:
            st.info("No recent conversations found. Start chatting to see activity here!")

    with col2:
        st.markdown("### 🚀 Quick Start")

        if not st.session_state.rag_engine:
            st.warning("Initialize the RAG system first!")
            if st.button("🔧 Go to Settings", key="quick_settings"):
                st.switch_page("pages/settings.py")
        else:
            st.success("System ready! Choose an action below:")

            if st.button("💬 Start Chatting", type="primary", key="quick_chat"):
                st.switch_page("pages/chat.py")

            if st.button("📁 Manage Documents", key="quick_docs"):
                st.switch_page("pages/documents.py")

            if st.button("📈 View Analytics", key="quick_analytics"):
                st.switch_page("pages/analytics.py")

    # System performance chart
    if st.session_state.conversation_history:
        st.markdown("### 📊 Usage Analytics")

        # Create sample usage data
        dates = [datetime.now().strftime("%Y-%m-%d") for _ in range(7)]
        queries = [len([c for c in st.session_state.conversation_history if c.get('date') == date]) for date in dates]

        fig = px.line(
            x=dates[-7:],
            y=queries[-7:],
            title="Query Volume (Last 7 Days)",
            labels={'x': 'Date', 'y': 'Number of Queries'}
        )
        fig.update_traces(line=dict(color='#667eea', width=3))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, width="stretch")

def main():
    """Main application entry point"""
    init_session_state()
    apply_custom_css()

    # Render sidebar and get selected page
    page = render_sidebar()

    # Route to appropriate page
    if page == "Dashboard":
        render_dashboard()
    elif page == "Chat":
        exec(open("pages/chat.py").read())
    elif page == "Documents":
        exec(open("pages/documents.py").read())
    elif page == "Conversations":
        exec(open("pages/conversations.py").read())
    elif page == "Analytics":
        exec(open("pages/analytics.py").read())
    elif page == "Settings":
        exec(open("pages/settings.py").read())

if __name__ == "__main__":
    main()