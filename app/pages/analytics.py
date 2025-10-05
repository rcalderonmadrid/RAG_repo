import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from session_init import ensure_session_state
import utils

# Ensure session state is initialized
ensure_session_state()

def render_overview_metrics():
    """Render overview metrics dashboard"""
    st.markdown("### 📊 System Overview")

    # Get basic stats
    doc_stats = st.session_state.doc_manager.get_storage_stats()
    conv_analytics = st.session_state.conv_manager.get_conversation_analytics()

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📚 Total Documents",
            doc_stats['total_documents'],
            help="Number of uploaded documents"
        )

    with col2:
        st.metric(
            "💬 Conversations",
            conv_analytics.get('total_conversations', 0),
            help="Total chat conversations"
        )

    with col3:
        if st.session_state.rag_engine:
            chunk_count = st.session_state.rag_engine.get_chunk_count()
        else:
            chunk_count = 0
        st.metric(
            "🧩 Document Chunks",
            chunk_count,
            help="Total processed document chunks"
        )

    with col4:
        st.metric(
            "💾 Storage Used",
            doc_stats['readable_size'],
            help="Total storage space used"
        )

    # Secondary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_messages = conv_analytics.get('total_messages', 0)
        st.metric(
            "📨 Total Messages",
            total_messages,
            help="Total messages across all conversations"
        )

    with col2:
        avg_response_time = conv_analytics.get('average_processing_time', 0)
        st.metric(
            "⏱️ Avg Response Time",
            f"{avg_response_time:.2f}s",
            help="Average RAG processing time"
        )

    with col3:
        avg_conv_length = conv_analytics.get('average_conversation_length', 0)
        st.metric(
            "📏 Avg Conversation Length",
            f"{avg_conv_length:.1f} msgs",
            help="Average messages per conversation"
        )

    with col4:
        # Calculate document processing success rate
        documents = st.session_state.doc_manager.get_document_list()
        processed = len([d for d in documents if d.get('processing_status') == 'completed'])
        success_rate = (processed / len(documents) * 100) if documents else 0
        st.metric(
            "✅ Processing Success Rate",
            f"{success_rate:.1f}%",
            help="Percentage of successfully processed documents"
        )

def render_document_analytics():
    """Render document-related analytics"""
    st.markdown("### 📁 Document Analytics")

    documents = st.session_state.doc_manager.get_document_list()

    if not documents:
        st.info("📭 No documents to analyze yet!")
        return

    # Document type distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Document Types")

        # Count by file type
        type_counts = {}
        for doc in documents:
            file_type = doc.get('file_extension', 'unknown')
            type_counts[file_type] = type_counts.get(file_type, 0) + 1

        if type_counts:
            # Create pie chart
            fig = px.pie(
                values=list(type_counts.values()),
                names=list(type_counts.keys()),
                title="Distribution by File Type"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("#### 📈 Processing Status")

        # Count by processing status
        status_counts = {}
        for doc in documents:
            status = doc.get('processing_status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1

        if status_counts:
            # Create bar chart
            fig = px.bar(
                x=list(status_counts.keys()),
                y=list(status_counts.values()),
                title="Documents by Processing Status",
                color=list(status_counts.values()),
                color_continuous_scale="Viridis"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

    # Document size analysis
    st.markdown("#### 📏 Document Size Analysis")

    # Create size distribution chart
    sizes = [doc.get('file_size', 0) for doc in documents]
    size_labels = [f"{doc.get('filename', 'Unknown')[:20]}..." for doc in documents]

    if sizes:
        fig = px.histogram(
            x=sizes,
            nbins=20,
            title="Document Size Distribution",
            labels={'x': 'File Size (bytes)', 'y': 'Number of Documents'}
        )
        st.plotly_chart(fig, width="stretch")

        # Top largest documents
        doc_sizes = [(doc.get('filename', 'Unknown'), doc.get('file_size', 0)) for doc in documents]
        doc_sizes.sort(key=lambda x: x[1], reverse=True)

        st.markdown("**🏆 Largest Documents:**")
        for i, (filename, size) in enumerate(doc_sizes[:5]):
            st.markdown(f"{i+1}. **{filename[:30]}...** - {utils.format_file_size(size)}")

def render_conversation_analytics():
    """Render conversation-related analytics"""
    st.markdown("### 💬 Conversation Analytics")

    analytics = st.session_state.conv_manager.get_conversation_analytics()

    if not analytics or analytics.get('total_conversations', 0) == 0:
        st.info("📭 No conversation data to analyze yet!")
        return

    # Activity over time
    messages_per_day = analytics.get('messages_per_day', {})
    if messages_per_day:
        st.markdown("#### 📅 Daily Activity")

        # Convert to DataFrame
        dates = sorted(messages_per_day.keys())
        counts = [messages_per_day[date] for date in dates]

        df = pd.DataFrame({
            'Date': pd.to_datetime(dates),
            'Messages': counts
        })

        # Create line chart
        fig = px.line(
            df,
            x='Date',
            y='Messages',
            title='Daily Message Volume',
            markers=True
        )
        fig.update_traces(line=dict(color='#667eea', width=3))
        st.plotly_chart(fig, width="stretch")

        # Activity patterns
        col1, col2 = st.columns(2)

        with col1:
            # Most active day
            most_active = analytics.get('most_active_day')
            if most_active:
                st.metric(
                    "🏆 Most Active Day",
                    most_active[0],
                    f"{most_active[1]} messages"
                )

        with col2:
            # Average daily messages
            if dates:
                total_days = len(dates)
                total_messages = sum(counts)
                avg_daily = total_messages / total_days
                st.metric(
                    "📊 Daily Average",
                    f"{avg_daily:.1f}",
                    "messages per day"
                )

    # Response time analysis
    conversations = st.session_state.conv_manager.get_conversation_list()
    response_times = []

    for conv_info in conversations:
        full_conv = st.session_state.conv_manager.load_conversation(conv_info['id'])
        if full_conv:
            for message in full_conv.get('messages', []):
                if message.get('role') == 'assistant':
                    rt = message.get('processing_time', 0)
                    if rt > 0:
                        response_times.append(rt)

    if response_times:
        st.markdown("#### ⏱️ Response Time Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Response time distribution
            fig = px.histogram(
                x=response_times,
                nbins=20,
                title="Response Time Distribution",
                labels={'x': 'Response Time (seconds)', 'y': 'Frequency'}
            )
            fig.update_traces(marker_color='#667eea')
            st.plotly_chart(fig, width="stretch")

        with col2:
            # Response time statistics
            st.markdown("**📈 Response Time Stats:**")
            st.markdown(f"• **Average:** {sum(response_times) / len(response_times):.2f}s")
            st.markdown(f"• **Median:** {sorted(response_times)[len(response_times)//2]:.2f}s")
            st.markdown(f"• **Min:** {min(response_times):.2f}s")
            st.markdown(f"• **Max:** {max(response_times):.2f}s")
            st.markdown(f"• **Total Samples:** {len(response_times)}")

def render_system_performance():
    """Render system performance metrics"""
    st.markdown("### ⚡ System Performance")

    # RAG system health check
    if st.session_state.rag_engine:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏥 System Health")

            if st.button("🔍 Run Health Check", help="Check system components"):
                with st.spinner("Running health check..."):
                    health = st.session_state.rag_engine.health_check()

                # Display health status
                components = [
                    ('LLM', health.get('llm_available', False)),
                    ('Embeddings', health.get('embeddings_available', False)),
                    ('Vector Store', health.get('vectorstore_available', False)),
                    ('Documents', health.get('documents_loaded', False)),
                    ('RAG Chain', health.get('chain_ready', False))
                ]

                for component, status in components:
                    status_icon = "✅" if status else "❌"
                    st.markdown(f"{status_icon} **{component}**: {'Healthy' if status else 'Issue'}")

                # Overall status
                overall = health.get('overall_status', 'unknown')
                status_colors = {
                    'healthy': '🟢',
                    'partial': '🟡',
                    'unhealthy': '🔴'
                }
                st.markdown(f"**Overall Status:** {status_colors.get(overall, '⚪')} {overall.upper()}")

        with col2:
            st.markdown("#### 📊 Performance Metrics")

            # Calculate performance metrics
            documents = st.session_state.doc_manager.get_document_list()
            processed_docs = [d for d in documents if d.get('processing_status') == 'completed']

            # Processing efficiency
            if documents:
                efficiency = len(processed_docs) / len(documents) * 100
                st.metric("Processing Efficiency", f"{efficiency:.1f}%")

            # Average chunk count per document
            if processed_docs:
                total_chunks = sum(doc.get('chunk_count', 0) for doc in processed_docs)
                avg_chunks = total_chunks / len(processed_docs)
                st.metric("Avg Chunks per Doc", f"{avg_chunks:.1f}")

            # Storage efficiency
            if st.session_state.rag_engine:
                chunk_count = st.session_state.rag_engine.get_chunk_count()
                if chunk_count > 0:
                    doc_count = len(processed_docs)
                    if doc_count > 0:
                        chunks_per_doc = chunk_count / doc_count
                        st.metric("Vector DB Efficiency", f"{chunks_per_doc:.1f} chunks/doc")

    else:
        st.warning("🚨 RAG system not initialized! Cannot show performance metrics.")

def render_usage_insights():
    """Render usage insights and recommendations"""
    st.markdown("### 💡 Usage Insights & Recommendations")

    documents = st.session_state.doc_manager.get_document_list()
    conversations = st.session_state.conv_manager.get_conversation_list()
    conv_analytics = st.session_state.conv_manager.get_conversation_analytics()

    insights = []
    recommendations = []

    # Document insights
    if not documents:
        insights.append("📭 No documents uploaded yet")
        recommendations.append("Upload some documents to start using the RAG system")
    else:
        unprocessed_docs = [d for d in documents if d.get('processing_status') != 'completed']
        if unprocessed_docs:
            insights.append(f"⚠️ {len(unprocessed_docs)} documents not yet processed")
            recommendations.append("Process pending documents to improve search results")

        # Check for document diversity
        file_types = set(doc.get('file_extension', 'unknown') for doc in documents)
        if len(file_types) == 1:
            insights.append("📄 All documents are the same file type")
            recommendations.append("Consider adding diverse document types for richer content")

    # Conversation insights
    if not conversations:
        insights.append("💬 No conversations saved yet")
        recommendations.append("Start chatting and save conversations to build history")
    else:
        avg_length = conv_analytics.get('average_conversation_length', 0)
        if avg_length < 3:
            insights.append("📏 Conversations are relatively short")
            recommendations.append("Try asking follow-up questions for deeper insights")

        # Check for inactive periods
        messages_per_day = conv_analytics.get('messages_per_day', {})
        if messages_per_day:
            recent_activity = any(
                datetime.fromisoformat(date) > datetime.now() - timedelta(days=7)
                for date in messages_per_day.keys()
            )
            if not recent_activity:
                insights.append("😴 No recent activity detected")
                recommendations.append("Continue using the system regularly for best results")

    # Performance insights
    if st.session_state.rag_engine:
        chunk_count = st.session_state.rag_engine.get_chunk_count()
        if chunk_count > 10000:
            insights.append("🐌 Large number of chunks may impact performance")
            recommendations.append("Consider optimizing chunk size or document selection")

        avg_response_time = conv_analytics.get('average_processing_time', 0)
        if avg_response_time > 10:
            insights.append("⏰ Response times are slower than optimal")
            recommendations.append("Check system resources or consider smaller model")

    # Display insights
    if insights:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔍 Current Insights")
            for insight in insights:
                st.markdown(f"• {insight}")

        with col2:
            st.markdown("#### 🎯 Recommendations")
            for recommendation in recommendations:
                st.markdown(f"• {recommendation}")

    # Usage tips
    st.markdown("#### 💡 Usage Tips")

    tips = [
        "🎯 **Be Specific**: Ask detailed questions for better responses",
        "📚 **Use Context**: Reference specific documents or sections",
        "🔄 **Iterate**: Follow up with clarifying questions",
        "🔖 **Bookmark**: Save important conversations for later reference",
        "📊 **Review Sources**: Check citations to verify information",
        "🏷️ **Tag Documents**: Add tags to organize your document library",
        "📈 **Monitor Performance**: Regular health checks ensure optimal operation"
    ]

    for tip in tips:
        st.markdown(f"• {tip}")

def render_export_analytics():
    """Render analytics export functionality"""
    st.markdown("### 📤 Export Analytics")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Export Usage Report", help="Export comprehensive usage report"):
            generate_usage_report()

    with col2:
        if st.button("📈 Export Performance Metrics", help="Export system performance data"):
            generate_performance_report()

    with col3:
        if st.button("📋 Export System Summary", help="Export system status summary"):
            generate_system_summary()

def generate_usage_report():
    """Generate comprehensive usage report"""
    # Gather all analytics data
    doc_stats = st.session_state.doc_manager.get_storage_stats()
    conv_analytics = st.session_state.conv_manager.get_conversation_analytics()
    documents = st.session_state.doc_manager.get_document_list()
    conversations = st.session_state.conv_manager.get_conversation_list()

    report_data = {
        'report_info': {
            'generated_at': datetime.now().isoformat(),
            'report_type': 'usage_report',
            'version': '1.0'
        },
        'document_analytics': {
            'total_documents': doc_stats['total_documents'],
            'total_size': doc_stats['readable_size'],
            'file_types': doc_stats.get('file_types', {}),
            'processing_status': {}
        },
        'conversation_analytics': conv_analytics,
        'system_status': {}
    }

    # Add processing status breakdown
    status_counts = {}
    for doc in documents:
        status = doc.get('processing_status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
    report_data['document_analytics']['processing_status'] = status_counts

    # Add system status
    if st.session_state.rag_engine:
        health = st.session_state.rag_engine.health_check()
        report_data['system_status'] = health

    # Create downloadable report
    import json
    report_json = json.dumps(report_data, indent=2)

    st.download_button(
        "📥 Download Usage Report",
        data=report_json,
        file_name=f"usage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        help="Download detailed usage analytics"
    )

def generate_performance_report():
    """Generate performance metrics report"""
    # Collect performance data
    conversations = st.session_state.conv_manager.get_conversation_list()
    response_times = []

    for conv_info in conversations:
        full_conv = st.session_state.conv_manager.load_conversation(conv_info['id'])
        if full_conv:
            for message in full_conv.get('messages', []):
                if message.get('role') == 'assistant':
                    rt = message.get('processing_time', 0)
                    if rt > 0:
                        response_times.append({
                            'timestamp': message.get('timestamp', ''),
                            'response_time': rt,
                            'conversation_id': conv_info['id']
                        })

    # Create CSV data
    df = pd.DataFrame(response_times)

    if not df.empty:
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📥 Download Performance Data",
            data=csv_data,
            file_name=f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No performance data available to export")

def generate_system_summary():
    """Generate system status summary"""
    summary = f"""# RAG System Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## System Overview
- Documents: {st.session_state.doc_manager.get_document_count()}
- Conversations: {st.session_state.conv_manager.get_conversation_count()}
- RAG Engine: {'Initialized' if st.session_state.rag_engine else 'Not Initialized'}

## Configuration
- LLM Model: {st.session_state.config.llm_model}
- Embedding Model: {st.session_state.config.embedding_model}
- Chunk Size: {st.session_state.config.chunk_size}
- Temperature: {st.session_state.config.temperature}

## Storage
{st.session_state.doc_manager.get_storage_stats()['readable_size']} used across {st.session_state.doc_manager.get_document_count()} documents

## Recent Activity
{st.session_state.conv_manager.get_conversation_analytics().get('total_messages', 0)} total messages
{st.session_state.conv_manager.get_conversation_analytics().get('average_processing_time', 0):.2f}s average response time

---
Generated by RAG Intelligence Hub
"""

    st.download_button(
        "📥 Download System Summary",
        data=summary,
        file_name=f"system_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )

def render_analytics_page():
    """Main analytics page renderer"""
    st.markdown("# 📈 Analytics & Insights")
    st.markdown("Comprehensive analytics and performance insights for your RAG system.")

    # Tab layout for different analytics views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📁 Documents",
        "💬 Conversations",
        "⚡ Performance",
        "📤 Export"
    ])

    with tab1:
        render_overview_metrics()
        render_usage_insights()

    with tab2:
        render_document_analytics()

    with tab3:
        render_conversation_analytics()

    with tab4:
        render_system_performance()

    with tab5:
        render_export_analytics()

# Render the analytics page
render_analytics_page()