import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from session_init import ensure_session_state
import utils

# Ensure session state is initialized
ensure_session_state()

def render_conversation_list():
    """Render list of saved conversations"""
    st.markdown("### 💬 Conversation History")

    conversations = st.session_state.conv_manager.get_conversation_list(limit=100)

    if not conversations:
        st.info("📭 No saved conversations yet. Start chatting to build your conversation history!")
        if st.button("💬 Start Chatting"):
            st.switch_page("pages/chat.py")
        return

    # Controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sort_options = ["updated_at", "created_at", "message_count", "title"]
        sort_by = st.selectbox("Sort by:", sort_options, key="conv_sort")

    with col2:
        sort_order = st.selectbox("Order:", ["Newest first", "Oldest first"], key="conv_order")

    with col3:
        time_filter = st.selectbox("Time filter:", ["All time", "Last 7 days", "Last 30 days", "Last 90 days"], key="conv_time")

    with col4:
        search_query = st.text_input("🔍 Search:", placeholder="Search conversations...", key="conv_search")

    # Apply filters
    filtered_conversations = conversations

    # Time filter
    if time_filter != "All time":
        days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
        cutoff_date = datetime.now() - timedelta(days=days_map[time_filter])

        filtered_conversations = [
            conv for conv in filtered_conversations
            if datetime.fromisoformat(conv.get('updated_at', '')) > cutoff_date
        ]

    # Search filter
    if search_query:
        search_results = st.session_state.conv_manager.search_conversations(search_query)
        search_ids = {conv['id'] for conv in search_results}
        filtered_conversations = [conv for conv in filtered_conversations if conv['id'] in search_ids]

    # Sort
    reverse = sort_order == "Newest first"
    try:
        filtered_conversations.sort(key=lambda x: x.get(sort_by, ''), reverse=reverse)
    except:
        pass  # Keep original order if sorting fails

    # Bulk actions
    st.markdown("### ⚡ Bulk Actions")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📤 Export All", help="Export all conversations"):
            export_all_conversations(filtered_conversations)

    with col2:
        if st.button("🔖 Show Bookmarked", help="Show only bookmarked conversations"):
            show_bookmarked_conversations()

    with col3:
        if st.button("🧹 Cleanup Old", help="Delete old conversations"):
            cleanup_old_conversations()

    with col4:
        if st.button("📊 Show Analytics", help="Show conversation analytics"):
            show_conversation_analytics()

    # Display conversations
    st.markdown(f"### 💭 Conversations ({len(filtered_conversations)})")

    for conv in filtered_conversations:
        render_conversation_card(conv)

def render_conversation_card(conv):
    """Render a single conversation card"""
    with st.container():
        # Load full conversation for additional details
        full_conv = st.session_state.conv_manager.load_conversation(conv['id'])
        is_bookmarked = full_conv.get('metadata', {}).get('bookmarked', False) if full_conv else False

        st.markdown(
            f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div style="flex-grow: 1;">
                        <h4 style="margin: 0; color: #1f2937;">
                            {'🔖' if is_bookmarked else '💬'} {conv.get('title', 'Untitled Conversation')}
                        </h4>
                        <div style="margin: 0.5rem 0; color: #6b7280;">
                            <span>{utils.format_timestamp(conv.get('updated_at', ''))}</span>
                            • <span>{conv.get('message_count', 0)} messages</span>
                            • <span>Created {utils.format_timestamp(conv.get('created_at', ''))}</span>
                        </div>
                        <p style="margin: 0.5rem 0; color: #4b5563; font-style: italic;">
                            {conv.get('summary', 'No preview available')}
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Action buttons
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            if st.button("👁️ View", key=f"view_{conv['id']}", help="View conversation"):
                view_conversation(conv['id'])

        with col2:
            if st.button("💬 Continue", key=f"continue_{conv['id']}", help="Continue conversation"):
                continue_conversation(conv['id'])

        with col3:
            bookmark_label = "📖" if is_bookmarked else "🔖"
            bookmark_help = "Remove bookmark" if is_bookmarked else "Bookmark conversation"
            if st.button(bookmark_label, key=f"bookmark_{conv['id']}", help=bookmark_help):
                toggle_bookmark(conv['id'], not is_bookmarked)

        with col4:
            if st.button("📝 Rename", key=f"rename_{conv['id']}", help="Rename conversation"):
                rename_conversation(conv['id'], conv.get('title', ''))

        with col5:
            if st.button("📤 Export", key=f"export_{conv['id']}", help="Export conversation"):
                export_conversation(conv['id'])

        with col6:
            if st.button("🗑️ Delete", key=f"delete_{conv['id']}", help="Delete conversation", type="secondary"):
                delete_conversation(conv['id'], conv.get('title', 'Untitled'))

        st.divider()

def view_conversation(conversation_id):
    """View full conversation in a modal"""
    conversation = st.session_state.conv_manager.load_conversation(conversation_id)

    if not conversation:
        st.error("Conversation not found!")
        return

    with st.modal(f"💬 {conversation.get('title', 'Untitled Conversation')}"):
        # Conversation metadata
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Messages", len(conversation.get('messages', [])))

        with col2:
            created_date = utils.format_timestamp(conversation.get('created_at', ''), 'date')
            st.metric("Created", created_date)

        with col3:
            avg_time = conversation.get('metadata', {}).get('average_response_time', 0)
            st.metric("Avg Response", f"{avg_time:.1f}s" if avg_time > 0 else "N/A")

        st.divider()

        # Messages
        st.markdown("### 💭 Messages")

        messages_container = st.container()
        with messages_container:
            for i, message in enumerate(conversation.get('messages', [])):
                role = message.get('role', 'user')
                content = message.get('content', '')
                timestamp = message.get('timestamp', '')
                processing_time = message.get('processing_time', 0)

                if role == 'user':
                    st.markdown(
                        f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; margin-left: 20%;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                <strong>👤 You</strong>
                                <small>{utils.format_timestamp(timestamp, 'short')}</small>
                            </div>
                            <div>{content}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
                                    border-left: 4px solid #667eea; padding: 1rem; border-radius: 10px;
                                    margin: 0.5rem 0; margin-right: 20%;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                <strong>🤖 Assistant</strong>
                                <small>{utils.format_timestamp(timestamp, 'short')}
                                     {f" • {utils.format_duration(processing_time)}" if processing_time > 0 else ""}</small>
                            </div>
                            <div>{content}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

def continue_conversation(conversation_id):
    """Continue a conversation in the chat interface"""
    conversation = st.session_state.conv_manager.load_conversation(conversation_id)

    if not conversation:
        st.error("Conversation not found!")
        return

    # Load messages into current conversation
    st.session_state.current_conversation = conversation.get('messages', [])

    st.success("Conversation loaded! Redirecting to chat...")
    time.sleep(1)
    st.switch_page("pages/chat.py")

def toggle_bookmark(conversation_id, bookmarked):
    """Toggle conversation bookmark status"""
    success = st.session_state.conv_manager.bookmark_conversation(conversation_id, bookmarked)

    if success:
        action = "Bookmarked" if bookmarked else "Removed bookmark from"
        st.success(f"✅ {action} conversation!")
    else:
        st.error("❌ Failed to update bookmark status!")

    time.sleep(1)
    st.rerun()

def rename_conversation(conversation_id, current_title):
    """Rename a conversation"""
    with st.modal("📝 Rename Conversation"):
        st.markdown(f"**Current title:** {current_title}")

        with st.form("rename_form"):
            new_title = st.text_input("New title:", value=current_title, max_chars=100)

            if st.form_submit_button("💾 Save", type="primary"):
                if new_title and new_title != current_title:
                    success = st.session_state.conv_manager.update_conversation_title(conversation_id, new_title)

                    if success:
                        st.success("✅ Conversation renamed!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to rename conversation!")
                elif not new_title:
                    st.error("Title cannot be empty!")

def export_conversation(conversation_id):
    """Export a single conversation"""
    conversation = st.session_state.conv_manager.load_conversation(conversation_id)

    if not conversation:
        st.error("Conversation not found!")
        return

    with st.modal("📤 Export Conversation"):
        st.markdown(f"### Export: {conversation.get('title', 'Untitled')}")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📄 JSON", width="stretch", help="Export as JSON"):
                export_data = st.session_state.conv_manager.export_conversation(conversation_id, 'json')
                if export_data['success']:
                    st.download_button(
                        "📥 Download JSON",
                        data=json.dumps(export_data['data'], indent=2),
                        file_name=export_data['filename'],
                        mime="application/json"
                    )

        with col2:
            if st.button("📝 Text", width="stretch", help="Export as plain text"):
                export_data = st.session_state.conv_manager.export_conversation(conversation_id, 'txt')
                if export_data['success']:
                    st.download_button(
                        "📥 Download Text",
                        data=export_data['data'],
                        file_name=export_data['filename'],
                        mime="text/plain"
                    )

        with col3:
            if st.button("📋 Markdown", width="stretch", help="Export as Markdown"):
                export_data = st.session_state.conv_manager.export_conversation(conversation_id, 'md')
                if export_data['success']:
                    st.download_button(
                        "📥 Download Markdown",
                        data=export_data['data'],
                        file_name=export_data['filename'],
                        mime="text/markdown"
                    )

def export_all_conversations(conversations):
    """Export all conversations as a single file"""
    if not conversations:
        st.warning("No conversations to export!")
        return

    with st.modal("📤 Export All Conversations"):
        st.markdown(f"### Export {len(conversations)} Conversations")

        export_format = st.selectbox("Format:", ["JSON", "CSV Summary"], key="export_all_format")

        if st.button("📥 Generate Export", type="primary"):
            if export_format == "JSON":
                # Export all as JSON
                all_conversations_data = []
                for conv_info in conversations:
                    full_conv = st.session_state.conv_manager.load_conversation(conv_info['id'])
                    if full_conv:
                        all_conversations_data.append(full_conv)

                export_data = {
                    'export_info': {
                        'timestamp': datetime.now().isoformat(),
                        'total_conversations': len(all_conversations_data),
                        'format_version': '1.0'
                    },
                    'conversations': all_conversations_data
                }

                st.download_button(
                    "📥 Download All Conversations (JSON)",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"all_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

            else:  # CSV Summary
                # Export summary as CSV
                csv_data = []
                for conv_info in conversations:
                    full_conv = st.session_state.conv_manager.load_conversation(conv_info['id'])
                    if full_conv:
                        csv_data.append({
                            'Title': full_conv.get('title', ''),
                            'Created': utils.format_timestamp(full_conv.get('created_at', ''), 'date'),
                            'Updated': utils.format_timestamp(full_conv.get('updated_at', ''), 'date'),
                            'Messages': len(full_conv.get('messages', [])),
                            'Bookmarked': full_conv.get('metadata', {}).get('bookmarked', False),
                            'Avg Response Time': f"{full_conv.get('metadata', {}).get('average_response_time', 0):.1f}s"
                        })

                df = pd.DataFrame(csv_data)
                csv_string = df.to_csv(index=False)

                st.download_button(
                    "📥 Download Conversation Summary (CSV)",
                    data=csv_string,
                    file_name=f"conversation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def show_bookmarked_conversations():
    """Show only bookmarked conversations"""
    bookmarked = st.session_state.conv_manager.get_bookmarked_conversations()

    with st.modal("🔖 Bookmarked Conversations"):
        if bookmarked:
            st.markdown(f"### {len(bookmarked)} Bookmarked Conversations")

            for conv in bookmarked:
                with st.expander(f"🔖 {conv.get('title', 'Untitled')}"):
                    st.markdown(f"**Created:** {utils.format_timestamp(conv.get('created_at', ''))}")
                    st.markdown(f"**Messages:** {conv.get('message_count', 0)}")
                    st.markdown(f"**Summary:** {conv.get('summary', 'No preview available')}")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"👁️ View", key=f"view_bookmarked_{conv['id']}"):
                            view_conversation(conv['id'])

                    with col2:
                        if st.button(f"💬 Continue", key=f"continue_bookmarked_{conv['id']}"):
                            continue_conversation(conv['id'])
        else:
            st.info("📭 No bookmarked conversations yet!")

def cleanup_old_conversations():
    """Clean up old conversations"""
    with st.modal("🧹 Cleanup Old Conversations"):
        st.markdown("### Delete Old Conversations")
        st.warning("This will delete conversations older than the specified days. Bookmarked conversations will be preserved.")

        days = st.number_input("Delete conversations older than (days):", min_value=1, max_value=365, value=90)

        if st.button("🗑️ Delete Old Conversations", type="primary"):
            result = st.session_state.conv_manager.cleanup_old_conversations(days)

            if result['success']:
                st.success(f"✅ Deleted {result['deleted_count']} old conversations")
            else:
                st.error(f"❌ Cleanup failed: {result.get('error', 'Unknown error')}")

def show_conversation_analytics():
    """Show conversation analytics"""
    analytics = st.session_state.conv_manager.get_conversation_analytics()

    with st.modal("📊 Conversation Analytics"):
        if analytics:
            st.markdown("### 📈 Usage Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Conversations", analytics.get('total_conversations', 0))

            with col2:
                st.metric("Total Messages", analytics.get('total_messages', 0))

            with col3:
                avg_length = analytics.get('average_conversation_length', 0)
                st.metric("Avg Length", f"{avg_length} messages")

            with col4:
                avg_time = analytics.get('average_processing_time', 0)
                st.metric("Avg Response Time", f"{avg_time:.1f}s")

            # Activity chart
            messages_per_day = analytics.get('messages_per_day', {})
            if messages_per_day:
                st.markdown("### 📅 Daily Activity")

                # Convert to DataFrame for plotting
                dates = list(messages_per_day.keys())
                counts = list(messages_per_day.values())

                chart_data = pd.DataFrame({
                    'Date': dates,
                    'Messages': counts
                })

                st.line_chart(chart_data.set_index('Date'))

            # Most active day
            most_active = analytics.get('most_active_day')
            if most_active:
                st.markdown(f"**🏆 Most active day:** {most_active[0]} ({most_active[1]} messages)")

        else:
            st.info("📭 No analytics data available yet!")

def delete_conversation(conversation_id, title):
    """Delete conversation with confirmation"""
    with st.modal(f"🗑️ Delete Conversation"):
        st.warning(f"Are you sure you want to delete **{title}**?")
        st.markdown("This action cannot be undone.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("❌ Cancel", width="stretch"):
                st.rerun()

        with col2:
            if st.button("🗑️ Delete", type="primary", width="stretch"):
                success = st.session_state.conv_manager.delete_conversation(conversation_id)

                if success:
                    st.success("✅ Conversation deleted!")
                else:
                    st.error("❌ Failed to delete conversation!")

                time.sleep(1)
                st.rerun()

def render_conversations_page():
    """Main conversations page renderer"""
    st.markdown("# 💾 Conversation History")
    st.markdown("View, manage, and export your chat conversations.")

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)

    total_conversations = st.session_state.conv_manager.get_conversation_count()
    bookmarked = st.session_state.conv_manager.get_bookmarked_conversations()
    analytics = st.session_state.conv_manager.get_conversation_analytics()

    with col1:
        st.metric("Total Conversations", total_conversations)

    with col2:
        st.metric("Bookmarked", len(bookmarked))

    with col3:
        total_messages = analytics.get('total_messages', 0)
        st.metric("Total Messages", total_messages)

    with col4:
        avg_length = analytics.get('average_conversation_length', 0)
        st.metric("Avg Length", f"{avg_length:.1f} msgs")

    st.divider()

    # Main conversation list
    render_conversation_list()

# Import required modules
import time

# Render the conversations page
render_conversations_page()