import streamlit as st
import time
from datetime import datetime
import uuid
from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from session_init import ensure_session_state

# Ensure session state is initialized
ensure_session_state()

def render_message(message: Dict[str, Any], key: str):
    """Render a single message in the chat"""
    role = message.get('role', 'user')
    content = message.get('content', '')
    timestamp = message.get('timestamp', datetime.now().isoformat())
    processing_time = message.get('processing_time', 0)

    if role == 'user':
        with st.container():
            st.markdown(
                f"""
                <div class="chat-message user-message">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <strong>👤 You</strong>
                        <small style="opacity: 0.7;">{utils.format_timestamp(timestamp, 'short')}</small>
                    </div>
                    <div>{content}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        with st.container():
            st.markdown(
                f"""
                <div class="chat-message assistant-message">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <strong>🤖 Assistant</strong>
                        <small style="opacity: 0.7;">
                            {utils.format_timestamp(timestamp, 'short')}
                            {f" • {utils.format_duration(processing_time)}" if processing_time > 0 else ""}
                        </small>
                    </div>
                    <div>{content}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Show sources if available
            sources = message.get('sources', [])
            if sources and st.session_state.config.get('enable_source_citations', True):
                with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                    for i, source in enumerate(sources[:5]):  # Limit to 5 sources
                        metadata = source.get('metadata', {})
                        content_preview = source.get('content', '')

                        st.markdown(f"**Source {i+1}:**")
                        if metadata.get('source'):
                            st.markdown(f"📄 {metadata['source']}")
                        if metadata.get('page'):
                            st.markdown(f"📃 Page {metadata['page']}")

                        st.markdown(f"```\n{utils.truncate_text(content_preview, 200)}\n```")
                        st.divider()

def render_chat_interface():
    """Render the main chat interface"""
    st.markdown("## 💬 Chat with Your Documents")

    # Check if RAG system is initialized
    if not st.session_state.rag_engine:
        st.error("🚨 RAG system not initialized! Please go to Settings to configure the system.")
        if st.button("🔧 Go to Settings"):
            st.switch_page("pages/settings.py")
        return

    # Check for and process pending documents
    try:
        from session_init import process_pending_documents
        processed_count = process_pending_documents()
        if processed_count and isinstance(processed_count, int) and processed_count > 0:
            st.success(f"✅ Processed {processed_count} pending document(s) automatically!")
            st.rerun()
    except Exception as e:
        st.warning(f"Note: Could not auto-process pending documents: {str(e)}")

    # Check if documents are loaded
    doc_count = st.session_state.doc_manager.get_document_count()
    if doc_count == 0:
        st.warning("📭 No documents uploaded yet! Upload some documents to start chatting.")
        if st.button("📁 Go to Documents"):
            st.switch_page("pages/documents.py")
        return

    # System status indicator
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"📚 **{doc_count} documents loaded** • 🧩 **{st.session_state.rag_engine.get_chunk_count()} chunks**")
    with col2:
        if st.button("🔄 Refresh", help="Refresh the system"):
            st.rerun()
    with col3:
        if st.button("💾 Save Chat", help="Save current conversation"):
            save_current_conversation()
            st.success("Conversation saved!")

    st.divider()

    # Chat history display
    chat_container = st.container()

    # Display conversation history
    if st.session_state.current_conversation:
        with chat_container:
            for i, message in enumerate(st.session_state.current_conversation):
                render_message(message, f"msg_{i}")

    # Quick action buttons
    if st.session_state.current_conversation:
        st.markdown("### ⚡ Quick Actions")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.current_conversation = []
                st.rerun()

        with col2:
            if st.button("📤 Export Chat"):
                export_conversation()

        with col3:
            if st.button("🔍 Search Chat"):
                st.session_state.show_search = not st.session_state.get('show_search', False)
                st.rerun()

        with col4:
            if st.button("🔖 Bookmark"):
                bookmark_conversation()

    # Search within conversation (if enabled)
    if st.session_state.get('show_search', False):
        st.markdown("### 🔍 Search in Conversation")
        search_query = st.text_input("Search messages:", placeholder="Enter search terms...")

        if search_query:
            matching_messages = []
            for i, message in enumerate(st.session_state.current_conversation):
                if search_query.lower() in message.get('content', '').lower():
                    matching_messages.append((i, message))

            if matching_messages:
                st.success(f"Found {len(matching_messages)} matching messages:")
                for msg_index, message in matching_messages:
                    with st.expander(f"Message {msg_index + 1} - {message.get('role', 'user').title()}"):
                        highlighted_content = utils.highlight_text(
                            message.get('content', ''), search_query
                        )
                        st.markdown(highlighted_content, unsafe_allow_html=True)
            else:
                st.info("No matching messages found.")

    # Chat input area
    st.markdown("### 💭 Ask a Question")

    # Suggested questions
    if not st.session_state.current_conversation:
        st.markdown("**💡 Suggested questions:**")
        suggestions = generate_suggested_questions()

        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions[:4]):
            with cols[i % 2]:
                if st.button(
                    suggestion,
                    key=f"suggestion_{i}",
                    help="Click to use this question",
                    width="stretch"
                ):
                    st.session_state.temp_question = suggestion
                    st.rerun()

    # Main chat input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])

        with col1:
            # Use temporary question if set from suggestions
            default_value = st.session_state.get('temp_question', '')
            if 'temp_question' in st.session_state:
                del st.session_state.temp_question

            user_question = st.text_input(
                "Your question:",
                value=default_value,
                placeholder="Type your question here and press Enter or click Send...",
                help="Ask anything about your uploaded documents",
                key="user_question_input"
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing

            # Advanced options
            with st.expander("🔧 Options"):
                include_sources = st.checkbox(
                    "Show sources",
                    value=st.session_state.config.get('enable_source_citations', True),
                    help="Include source citations in the response"
                )

                retrieval_k = st.slider(
                    "Number of sources",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.config.get('retrieval_k', 3),
                    help="How many document chunks to retrieve"
                )

                temperature = st.slider(
                    "Creativity",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.config.get('temperature', 0.1),
                    step=0.1,
                    help="Higher values make responses more creative but less focused"
                )

            submit_button = st.form_submit_button(
                "💬 Send",
                type="primary",
                width="stretch"
            )

    # Process the question
    if submit_button and user_question and user_question.strip():
        # Clear any temporary questions from session state
        if 'temp_question' in st.session_state:
            del st.session_state.temp_question

        # Add a success message to indicate the form was submitted
        st.success(f"✅ Processing your question: \"{user_question.strip()[:50]}{'...' if len(user_question.strip()) > 50 else ''}\"")

        process_user_question(
            user_question.strip(),
            include_sources=include_sources,
            retrieval_k=retrieval_k,
            temperature=temperature
        )
    elif submit_button and not user_question.strip():
        st.warning("⚠️ Please enter a question before submitting.")

def generate_suggested_questions() -> List[str]:
    """Generate suggested questions based on loaded documents"""
    # This could be enhanced to analyze document content for better suggestions
    general_suggestions = [
        "What are the main topics covered in the documents?",
        "Can you summarize the key findings?",
        "What methodology was used in this research?",
        "What are the limitations mentioned?",
        "What conclusions can be drawn?",
        "Are there any recommendations provided?"
    ]

    # Get document-specific suggestions if possible
    doc_stats = st.session_state.doc_manager.get_storage_stats()
    file_types = doc_stats.get('file_types', {})

    if '.pdf' in file_types:
        general_suggestions.extend([
            "What are the main sections of this document?",
            "What figures or tables are mentioned?"
        ])

    return general_suggestions

def process_user_question(question: str, include_sources: bool = True, retrieval_k: int = 3, temperature: float = 0.1):
    """Process user question and generate response"""

    # Check if RAG engine is available
    if not hasattr(st.session_state, 'rag_engine') or not st.session_state.rag_engine:
        st.error("🚨 RAG engine is not available. Please go to Settings to initialize the system.")
        return

    # Add user message to conversation
    user_message = {
        'role': 'user',
        'content': question,
        'timestamp': datetime.now().isoformat(),
        'id': str(uuid.uuid4())
    }

    st.session_state.current_conversation.append(user_message)

    # Show processing indicator
    with st.spinner("🤔 Thinking..."):
        try:
            # Store original temperature for restoration
            original_temp = st.session_state.rag_engine.llm.temperature

            # Update LLM temperature for this query
            st.session_state.rag_engine.llm.temperature = temperature

            # Get response from RAG engine with custom retrieval_k
            start_time = time.time()
            response = st.session_state.rag_engine.answer_question(question, retrieval_k=retrieval_k)
            processing_time = time.time() - start_time

            if response['success']:
                # Create assistant message
                assistant_message = {
                    'role': 'assistant',
                    'content': response['answer'],
                    'timestamp': datetime.now().isoformat(),
                    'processing_time': processing_time,
                    'sources': response.get('sources', []) if include_sources else [],
                    'id': str(uuid.uuid4())
                }

                st.session_state.current_conversation.append(assistant_message)

                # Add to conversation history
                st.session_state.conversation_history.append({
                    'question': question,
                    'answer': response['answer'],
                    'timestamp': datetime.now().isoformat(),
                    'processing_time': processing_time
                })

            else:
                # Error message
                error_message = {
                    'role': 'assistant',
                    'content': f"❌ **Error:** {response['answer']}",
                    'timestamp': datetime.now().isoformat(),
                    'processing_time': processing_time,
                    'sources': [],
                    'id': str(uuid.uuid4())
                }

                st.session_state.current_conversation.append(error_message)

        except Exception as e:
            # Handle any unexpected errors
            processing_time = time.time() - start_time if 'start_time' in locals() else 0
            error_message = {
                'role': 'assistant',
                'content': f"❌ **Unexpected Error:** {str(e)}",
                'timestamp': datetime.now().isoformat(),
                'processing_time': processing_time,
                'sources': [],
                'id': str(uuid.uuid4())
            }
            st.session_state.current_conversation.append(error_message)
            st.error(f"Error processing question: {str(e)}")

        finally:
            # Restore original temperature
            try:
                if 'original_temp' in locals():
                    st.session_state.rag_engine.llm.temperature = original_temp
            except Exception as e:
                st.error(f"Error restoring temperature: {str(e)}")

    # Rerun to show the new messages
    st.rerun()

def save_current_conversation():
    """Save current conversation to conversation manager"""
    if not st.session_state.current_conversation:
        return

    # Generate conversation title from first user message
    first_user_msg = None
    for msg in st.session_state.current_conversation:
        if msg.get('role') == 'user':
            first_user_msg = msg.get('content', '')
            break

    title = utils.truncate_text(first_user_msg or "Untitled Conversation", 50)

    # Save conversation
    conv_id = st.session_state.conv_manager.create_conversation(title)

    for message in st.session_state.current_conversation:
        st.session_state.conv_manager.add_message(conv_id, message)

    return conv_id

def export_conversation():
    """Export current conversation"""
    if not st.session_state.current_conversation:
        st.warning("No conversation to export!")
        return

    # Save conversation first
    conv_id = save_current_conversation()

    # Show export options
    with st.modal("📤 Export Conversation"):
        st.markdown("### Choose Export Format")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📄 JSON", width="stretch"):
                export_data = st.session_state.conv_manager.export_conversation(conv_id, 'json')
                if export_data['success']:
                    st.download_button(
                        "Download JSON",
                        data=json.dumps(export_data['data'], indent=2),
                        file_name=export_data['filename'],
                        mime="application/json"
                    )

        with col2:
            if st.button("📝 Text", width="stretch"):
                export_data = st.session_state.conv_manager.export_conversation(conv_id, 'txt')
                if export_data['success']:
                    st.download_button(
                        "Download Text",
                        data=export_data['data'],
                        file_name=export_data['filename'],
                        mime="text/plain"
                    )

        with col3:
            if st.button("📋 Markdown", width="stretch"):
                export_data = st.session_state.conv_manager.export_conversation(conv_id, 'md')
                if export_data['success']:
                    st.download_button(
                        "Download Markdown",
                        data=export_data['data'],
                        file_name=export_data['filename'],
                        mime="text/markdown"
                    )

def bookmark_conversation():
    """Bookmark current conversation"""
    if not st.session_state.current_conversation:
        st.warning("No conversation to bookmark!")
        return

    # Save and bookmark conversation
    conv_id = save_current_conversation()
    st.session_state.conv_manager.bookmark_conversation(conv_id, True)
    st.success("🔖 Conversation bookmarked!")

# Import required modules at the top of the file
import utils
import json

# Render the chat interface
render_chat_interface()