import streamlit as st
import json
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from session_init import ensure_session_state
import utils

# Ensure session state is initialized
ensure_session_state()

def render_model_settings():
    """Render model configuration settings"""
    st.markdown("### 🤖 Model Settings")

    with st.form("model_settings_form"):
        col1, col2 = st.columns(2)

        with col1:
            # LLM Model settings
            st.markdown("#### 🧠 Language Model")

            available_models = st.session_state.config.get_available_models()
            llm_models = available_models.get('llm_models', [])

            if llm_models:
                current_llm = st.session_state.config.llm_model
                llm_index = llm_models.index(current_llm) if current_llm in llm_models else 0

                new_llm_model = st.selectbox(
                    "LLM Model:",
                    llm_models,
                    index=llm_index,
                    help="Choose the language model for generating responses"
                )
            else:
                new_llm_model = st.text_input(
                    "LLM Model:",
                    value=st.session_state.config.llm_model,
                    help="Enter the LLM model name (ensure it's available in Ollama)"
                )

            new_temperature = st.slider(
                "Temperature:",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.config.temperature,
                step=0.1,
                help="Higher values make responses more creative but less focused"
            )

            new_max_tokens = st.number_input(
                "Max Tokens:",
                min_value=100,
                max_value=8192,
                value=st.session_state.config.get('max_tokens', 2000),
                help="Maximum number of tokens in response"
            )

        with col2:
            # Embedding Model settings
            st.markdown("#### 🔍 Embedding Model")

            embedding_models = available_models.get('embedding_models', [])

            if embedding_models:
                current_embedding = st.session_state.config.embedding_model
                embedding_index = embedding_models.index(current_embedding) if current_embedding in embedding_models else 0

                new_embedding_model = st.selectbox(
                    "Embedding Model:",
                    embedding_models,
                    index=embedding_index,
                    help="Choose the embedding model for document vectorization"
                )
            else:
                new_embedding_model = st.text_input(
                    "Embedding Model:",
                    value=st.session_state.config.embedding_model,
                    help="Enter the embedding model name (ensure it's available in Ollama)"
                )

            new_ollama_url = st.text_input(
                "Ollama URL:",
                value=st.session_state.config.get('ollama_url', 'http://localhost:11434'),
                help="URL of the Ollama service"
            )

        # Save button
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("💾 Save Model Settings", type="primary"):
                updates = {
                    'llm_model': new_llm_model,
                    'embedding_model': new_embedding_model,
                    'temperature': new_temperature,
                    'max_tokens': new_max_tokens,
                    'ollama_url': new_ollama_url
                }

                if st.session_state.config.update(updates):
                    st.success("✅ Model settings saved successfully!")

                    # Reinitialize RAG engine if it exists
                    if st.session_state.rag_engine:
                        with st.spinner("Reinitializing RAG engine with new settings..."):
                            try:
                                from rag_engine import RAGEngine
                                st.session_state.rag_engine = RAGEngine(st.session_state.config)
                                st.success("🔄 RAG engine reinitialized!")
                            except Exception as e:
                                st.error(f"❌ Failed to reinitialize RAG engine: {str(e)}")

                    st.rerun()
                else:
                    st.error("❌ Failed to save settings!")

        with col2:
            if st.form_submit_button("🔍 Test Connection"):
                with st.spinner("Testing Ollama connection..."):
                    # Temporarily update the config for testing
                    temp_config = st.session_state.config
                    temp_config.ollama_url = new_ollama_url
                    connection_result = temp_config.test_ollama_connection()

                if connection_result['connected']:
                    st.success(f"✅ Connected! Found {connection_result['model_count']} models")
                else:
                    st.error(f"❌ Connection failed: {connection_result['error']}")

def render_rag_settings():
    """Render RAG-specific settings"""
    st.markdown("### 🔍 RAG Settings")

    with st.form("rag_settings_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📄 Document Processing")

            new_chunk_size = st.number_input(
                "Chunk Size:",
                min_value=100,
                max_value=4000,
                value=st.session_state.config.chunk_size,
                help="Size of text chunks for processing (characters)"
            )

            new_chunk_overlap = st.number_input(
                "Chunk Overlap:",
                min_value=0,
                max_value=500,
                value=st.session_state.config.chunk_overlap,
                help="Overlap between consecutive chunks (characters)"
            )

            new_retrieval_k = st.number_input(
                "Retrieval K:",
                min_value=1,
                max_value=20,
                value=st.session_state.config.retrieval_k,
                help="Number of document chunks to retrieve for each query"
            )

        with col2:
            st.markdown("#### 🗂️ System Settings")

            new_persist_directory = st.text_input(
                "Vector DB Directory:",
                value=st.session_state.config.persist_directory,
                help="Directory to store the vector database"
            )

            new_max_file_size = st.number_input(
                "Max File Size (MB):",
                min_value=1,
                max_value=500,
                value=int(st.session_state.config.max_file_size_mb),
                help="Maximum file size for uploads"
            )

            new_conversation_retention = st.number_input(
                "Conversation Retention (days):",
                min_value=1,
                max_value=365,
                value=st.session_state.config.get('conversation_retention_days', 90),
                help="How long to keep conversations before cleanup"
            )

        # Advanced settings
        st.markdown("#### ⚙️ Advanced Settings")

        col1, col2, col3 = st.columns(3)

        with col1:
            new_enable_citations = st.checkbox(
                "Enable Source Citations",
                value=st.session_state.config.get('enable_source_citations', True),
                help="Include source citations in responses"
            )

            new_streaming = st.checkbox(
                "Streaming Responses",
                value=st.session_state.config.get('streaming_responses', True),
                help="Enable streaming response generation"
            )

        with col2:
            new_enable_analytics = st.checkbox(
                "Enable Analytics",
                value=st.session_state.config.get('enable_analytics', True),
                help="Collect usage analytics and performance metrics"
            )

            new_show_processing_time = st.checkbox(
                "Show Processing Time",
                value=st.session_state.config.get('show_processing_time', True),
                help="Display response generation time in chat"
            )

        with col3:
            new_enable_export = st.checkbox(
                "Enable Conversation Export",
                value=st.session_state.config.get('enable_conversation_export', True),
                help="Allow exporting conversations in various formats"
            )

            new_enable_preview = st.checkbox(
                "Enable Document Preview",
                value=st.session_state.config.get('enable_document_preview', True),
                help="Enable document content preview functionality"
            )

        # Save button
        if st.form_submit_button("💾 Save RAG Settings", type="primary"):
            updates = {
                'chunk_size': new_chunk_size,
                'chunk_overlap': new_chunk_overlap,
                'retrieval_k': new_retrieval_k,
                'persist_directory': new_persist_directory,
                'max_file_size_mb': new_max_file_size,
                'conversation_retention_days': new_conversation_retention,
                'enable_source_citations': new_enable_citations,
                'streaming_responses': new_streaming,
                'enable_analytics': new_enable_analytics,
                'show_processing_time': new_show_processing_time,
                'enable_conversation_export': new_enable_export,
                'enable_document_preview': new_enable_preview
            }

            # Validate settings
            validation = st.session_state.config.validate_settings()
            if not validation['valid']:
                st.error("❌ Invalid settings:")
                for error in validation['errors']:
                    st.error(f"• {error}")
                return

            if st.session_state.config.update(updates):
                st.success("✅ RAG settings saved successfully!")

                # Show warnings if any
                for warning in validation.get('warnings', []):
                    st.warning(f"⚠️ {warning}")

                st.rerun()
            else:
                st.error("❌ Failed to save settings!")

def render_ui_settings():
    """Render UI/UX settings"""
    st.markdown("### 🎨 UI/UX Settings")

    with st.form("ui_settings_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎨 Appearance")

            new_theme = st.selectbox(
                "Theme:",
                ["light", "dark", "auto"],
                index=["light", "dark", "auto"].index(st.session_state.config.theme),
                help="Choose the application theme"
            )

            new_items_per_page = st.number_input(
                "Items per Page:",
                min_value=5,
                max_value=100,
                value=st.session_state.config.items_per_page,
                help="Number of items to show per page in lists"
            )

        with col2:
            st.markdown("#### 🔧 Behavior")

            new_auto_save = st.checkbox(
                "Auto-save Conversations",
                value=st.session_state.config.get('auto_save', True),
                help="Automatically save conversations"
            )

            new_enable_search_highlighting = st.checkbox(
                "Enable Search Highlighting",
                value=st.session_state.config.get('enable_search_highlighting', True),
                help="Highlight search terms in results"
            )

        # Save button
        if st.form_submit_button("💾 Save UI Settings", type="primary"):
            updates = {
                'theme': new_theme,
                'items_per_page': new_items_per_page,
                'auto_save': new_auto_save,
                'enable_search_highlighting': new_enable_search_highlighting
            }

            if st.session_state.config.update(updates):
                st.success("✅ UI settings saved successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to save settings!")

def render_import_export():
    """Render configuration import/export functionality"""
    st.markdown("### 📤 Import/Export Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📤 Export Configuration")
        st.markdown("Export your current configuration to backup or share with others.")

        if st.button("📥 Export Config", type="primary"):
            config_data = st.session_state.config.export_config()
            config_json = json.dumps(config_data, indent=2)

            st.download_button(
                "📥 Download Configuration",
                data=config_json,
                file_name=f"rag_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Download current configuration as JSON file"
            )

    with col2:
        st.markdown("#### 📥 Import Configuration")
        st.markdown("Import configuration from a previously exported file.")

        uploaded_config = st.file_uploader(
            "Choose configuration file:",
            type=['json'],
            help="Select a JSON configuration file to import"
        )

        if uploaded_config is not None:
            if st.button("🔄 Import Configuration", type="primary"):
                try:
                    config_data = json.loads(uploaded_config.read())
                    result = st.session_state.config.import_config(config_data)

                    if result['success']:
                        st.success("✅ Configuration imported successfully!")

                        # Show warnings if any
                        for warning in result.get('warnings', []):
                            st.warning(f"⚠️ {warning}")

                        st.rerun()
                    else:
                        st.error(f"❌ Import failed: {result['message']}")
                        for error in result.get('errors', []):
                            st.error(f"• {error}")

                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON file!")
                except Exception as e:
                    st.error(f"❌ Import failed: {str(e)}")

def render_system_management():
    """Render system management tools"""
    st.markdown("### 🔧 System Management")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🚀 RAG System")

        if not st.session_state.rag_engine:
            if st.button("🔧 Initialize RAG System", type="primary"):
                with st.spinner("Initializing RAG system..."):
                    try:
                        from rag_engine import RAGEngine
                        st.session_state.rag_engine = RAGEngine(st.session_state.config)
                        st.success("✅ RAG system initialized successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to initialize RAG system: {str(e)}")
        else:
            if st.button("🔄 Restart RAG System"):
                with st.spinner("Restarting RAG system..."):
                    try:
                        from rag_engine import RAGEngine
                        st.session_state.rag_engine = RAGEngine(st.session_state.config)
                        st.success("✅ RAG system restarted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to restart RAG system: {str(e)}")

            if st.button("🏥 Run Health Check"):
                with st.spinner("Running system health check..."):
                    health = st.session_state.rag_engine.health_check()

                st.markdown("**System Health Status:**")
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

    with col2:
        st.markdown("#### 🗑️ Data Management")

        if st.button("🧹 Cleanup Old Conversations"):
            with st.modal("🧹 Cleanup Confirmation"):
                st.warning("This will delete conversations older than the retention period.")

                days = st.number_input(
                    "Delete conversations older than (days):",
                    min_value=1,
                    max_value=365,
                    value=st.session_state.config.get('conversation_retention_days', 90)
                )

                if st.button("🗑️ Confirm Cleanup", type="primary"):
                    result = st.session_state.conv_manager.cleanup_old_conversations(days)

                    if result['success']:
                        st.success(f"✅ Deleted {result['deleted_count']} old conversations")
                    else:
                        st.error(f"❌ Cleanup failed: {result.get('error', 'Unknown error')}")

        if st.button("🗂️ Cleanup Orphaned Files"):
            with st.spinner("Cleaning up orphaned files..."):
                result = st.session_state.doc_manager.cleanup_orphaned_files()

            if result['success']:
                st.success(f"✅ Removed {result['removed_count']} orphaned files")
            else:
                st.error(f"❌ Cleanup failed: {result.get('error', 'Unknown error')}")

        if st.button("🔄 Reset to Defaults"):
            with st.modal("⚠️ Reset Configuration"):
                st.warning("This will reset all settings to default values. This action cannot be undone.")

                if st.button("🔄 Confirm Reset", type="primary"):
                    if st.session_state.config.reset_to_defaults():
                        st.success("✅ Configuration reset to defaults!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to reset configuration!")

def render_system_info():
    """Render system information"""
    st.markdown("### ℹ️ System Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Current Configuration")

        config_summary = {
            'LLM Model': st.session_state.config.llm_model,
            'Embedding Model': st.session_state.config.embedding_model,
            'Temperature': st.session_state.config.temperature,
            'Chunk Size': st.session_state.config.chunk_size,
            'Chunk Overlap': st.session_state.config.chunk_overlap,
            'Retrieval K': st.session_state.config.retrieval_k,
            'Max File Size': f"{st.session_state.config.max_file_size_mb}MB",
            'Vector DB Path': st.session_state.config.persist_directory
        }

        for key, value in config_summary.items():
            st.markdown(f"**{key}:** {value}")

    with col2:
        st.markdown("#### 📈 System Stats")

        # Document stats
        doc_stats = st.session_state.doc_manager.get_storage_stats()
        st.markdown(f"**Documents:** {doc_stats['total_documents']}")
        st.markdown(f"**Storage Used:** {doc_stats['readable_size']}")

        # Conversation stats
        conv_count = st.session_state.conv_manager.get_conversation_count()
        st.markdown(f"**Conversations:** {conv_count}")

        # RAG system stats
        if st.session_state.rag_engine:
            chunk_count = st.session_state.rag_engine.get_chunk_count()
            st.markdown(f"**Document Chunks:** {chunk_count}")
            st.markdown("**RAG Status:** ✅ Initialized")
        else:
            st.markdown("**RAG Status:** ❌ Not Initialized")

        # Ollama connection
        connection_status = st.session_state.config.test_ollama_connection()
        status_icon = "✅" if connection_status['connected'] else "❌"
        st.markdown(f"**Ollama:** {status_icon} {connection_status['status']}")

def render_settings_page():
    """Main settings page renderer"""
    st.markdown("# ⚙️ Settings & Configuration")
    st.markdown("Configure your RAG system settings and preferences.")

    # Tab layout for different settings categories
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🤖 Models",
        "🔍 RAG",
        "🎨 UI/UX",
        "📤 Import/Export",
        "🔧 System",
        "ℹ️ Info"
    ])

    with tab1:
        render_model_settings()

    with tab2:
        render_rag_settings()

    with tab3:
        render_ui_settings()

    with tab4:
        render_import_export()

    with tab5:
        render_system_management()

    with tab6:
        render_system_info()

# Render the settings page
render_settings_page()