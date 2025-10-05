import streamlit as st
import pandas as pd
from datetime import datetime
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from session_init import ensure_session_state
import utils

# Ensure session state is initialized
ensure_session_state()

def render_document_upload():
    """Render document upload interface"""
    st.markdown("### 📤 Upload New Documents")

    with st.form("upload_form"):
        uploaded_files = st.file_uploader(
            "Choose documents to upload",
            type=st.session_state.config.supported_file_types,
            accept_multiple_files=True,
            help=f"Supported formats: {', '.join(st.session_state.config.supported_file_types)}"
        )

        col1, col2 = st.columns(2)
        with col1:
            description = st.text_area(
                "Description (optional)",
                placeholder="Brief description of the documents...",
                height=100
            )

        with col2:
            # Upload options
            st.markdown("**Upload Options:**")
            auto_process = st.checkbox(
                "Auto-process after upload",
                value=True,
                help="Automatically process documents for RAG after upload"
            )

            overwrite_duplicates = st.checkbox(
                "Overwrite duplicates",
                value=False,
                help="Replace existing documents with the same content hash"
            )

        submit_upload = st.form_submit_button(
            "🚀 Upload Documents",
            type="primary",
            width="stretch"
        )

    if submit_upload and uploaded_files:
        process_file_uploads(uploaded_files, description, auto_process, overwrite_duplicates)

def process_file_uploads(uploaded_files, description, auto_process, overwrite_duplicates):
    """Process uploaded files"""
    upload_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        progress_bar.progress((i + 1) / len(uploaded_files))

        # Upload file
        result = st.session_state.doc_manager.upload_file(uploaded_file, description)

        if result['success']:
            upload_results.append({
                'file': uploaded_file.name,
                'status': '✅ Success',
                'message': result['message'],
                'hash': result['file_hash'],
                'size': utils.format_file_size(result['file_info']['file_size'])
            })

            # Auto-process if enabled
            if auto_process:
                process_document_for_rag(result['file_hash'], result['file_path'])

        elif result.get('duplicate'):
            if overwrite_duplicates:
                # Handle duplicate overwrite logic here
                upload_results.append({
                    'file': uploaded_file.name,
                    'status': '🔄 Replaced',
                    'message': 'Duplicate file replaced',
                    'hash': result['file_hash'],
                    'size': 'N/A'
                })
            else:
                upload_results.append({
                    'file': uploaded_file.name,
                    'status': '⚠️ Duplicate',
                    'message': result['message'],
                    'hash': result.get('file_hash', 'N/A'),
                    'size': 'N/A'
                })
        else:
            upload_results.append({
                'file': uploaded_file.name,
                'status': '❌ Failed',
                'message': result['message'],
                'hash': 'N/A',
                'size': 'N/A'
            })

        time.sleep(0.1)  # Small delay for better UX

    progress_bar.progress(1.0)
    status_text.text("Upload complete!")

    # Display results
    if upload_results:
        st.markdown("### 📊 Upload Results")
        df = pd.DataFrame(upload_results)
        st.dataframe(df, width="stretch")

        # Summary
        successful = len([r for r in upload_results if r['status'].startswith('✅')])
        st.success(f"Successfully uploaded {successful} out of {len(uploaded_files)} files!")

    time.sleep(1)
    st.rerun()

def process_document_for_rag(file_hash, file_path):
    """Process document for RAG system"""
    if not st.session_state.rag_engine:
        st.warning("RAG system not initialized. Document uploaded but not processed.")
        return

    try:
        # Update status to processing
        st.session_state.doc_manager.update_document_status(file_hash, 'processing')

        # Load document
        doc_result = st.session_state.rag_engine.load_document(file_path)

        if doc_result['success']:
            # Split into chunks
            chunks = st.session_state.rag_engine.split_documents(doc_result['pages'])

            # Add to RAG system
            if st.session_state.rag_engine.add_documents(chunks, file_hash):
                # Update vectorstore
                st.session_state.rag_engine.create_vectorstore()

                # Update status
                st.session_state.doc_manager.update_document_status(
                    file_hash, 'completed', len(chunks)
                )
            else:
                st.session_state.doc_manager.update_document_status(
                    file_hash, 'error', 0, 'Failed to add to RAG system'
                )
        else:
            st.session_state.doc_manager.update_document_status(
                file_hash, 'error', 0, doc_result['error']
            )

    except Exception as e:
        st.session_state.doc_manager.update_document_status(
            file_hash, 'error', 0, str(e)
        )

def render_document_list():
    """Render list of uploaded documents"""
    st.markdown("### 📚 Document Library")

    # Get document list
    documents = st.session_state.doc_manager.get_document_list()

    if not documents:
        st.info("📭 No documents uploaded yet. Upload some documents to get started!")
        return

    # Controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sort_options = ["upload_timestamp", "filename", "file_size"]
        sort_by = st.selectbox("Sort by:", sort_options, key="doc_sort")

    with col2:
        sort_order = st.selectbox("Order:", ["Newest first", "Oldest first"], key="doc_order")
        reverse = sort_order == "Newest first"

    with col3:
        filter_type = st.selectbox("Filter by type:", ["All"] + list(st.session_state.config.supported_file_types), key="doc_filter")

    with col4:
        search_query = st.text_input("🔍 Search:", placeholder="Search documents...", key="doc_search")

    # Apply filters
    filtered_docs = documents

    if filter_type != "All":
        filtered_docs = [doc for doc in filtered_docs if doc.get('file_extension') == filter_type]

    if search_query:
        filtered_docs = [
            doc for doc in filtered_docs
            if search_query.lower() in doc.get('filename', '').lower() or
               search_query.lower() in doc.get('description', '').lower()
        ]

    # Sort documents
    if sort_by in ['upload_timestamp', 'file_size', 'filename']:
        filtered_docs.sort(key=lambda x: x.get(sort_by, ''), reverse=reverse)

    # Bulk actions
    st.markdown("### ⚡ Bulk Actions")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔄 Reprocess All", help="Reprocess all documents for RAG"):
            reprocess_all_documents(filtered_docs)

    with col2:
        if st.button("🧹 Cleanup Orphans", help="Remove orphaned files"):
            cleanup_result = st.session_state.doc_manager.cleanup_orphaned_files()
            if cleanup_result['success']:
                st.success(f"Removed {cleanup_result['removed_count']} orphaned files")
            else:
                st.error(f"Cleanup failed: {cleanup_result.get('error', 'Unknown error')}")

    with col3:
        if st.button("📊 Refresh Stats", help="Refresh document statistics"):
            st.rerun()

    with col4:
        if st.button("📤 Export List", help="Export document list"):
            export_document_list(filtered_docs)

    # Display documents
    st.markdown(f"### 📄 Documents ({len(filtered_docs)})")

    for doc in filtered_docs:
        render_document_card(doc)

def render_document_card(doc):
    """Render a single document card"""
    with st.container():
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div style="flex-grow: 1;">
                        <h4 style="margin: 0; color: #1f2937;">
                            📄 {doc.get('filename', 'Unknown')}
                        </h4>
                        <div style="margin: 0.5rem 0; color: #6b7280;">
                            <span>{doc.get('status_display', 'Unknown')}</span>
                            • <span>{utils.format_file_size(doc.get('file_size', 0))}</span>
                            • <span>{utils.format_timestamp(doc.get('upload_timestamp', ''))}</span>
                            {f" • {doc.get('chunk_count', 0)} chunks" if doc.get('chunk_count', 0) > 0 else ""}
                        </div>
                        {f"<p style='margin: 0.5rem 0; font-style: italic;'>{doc.get('description', '')}</p>" if doc.get('description') else ""}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Action buttons
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if st.button("👁️ Preview", key=f"preview_{doc['file_hash']}", help="Preview document"):
                preview_document(doc['file_hash'])

        with col2:
            if st.button("📝 Edit", key=f"edit_{doc['file_hash']}", help="Edit metadata"):
                edit_document_metadata(doc['file_hash'])

        with col3:
            if st.button("🔄 Reprocess", key=f"reprocess_{doc['file_hash']}", help="Reprocess for RAG"):
                reprocess_document(doc['file_hash'])

        with col4:
            if st.button("📤 Export", key=f"export_{doc['file_hash']}", help="Export document"):
                export_document(doc['file_hash'])

        with col5:
            if st.button("🗑️ Delete", key=f"delete_{doc['file_hash']}", help="Delete document", type="secondary"):
                delete_document(doc['file_hash'])

        st.divider()

def preview_document(file_hash):
    """Preview document content"""
    with st.modal("👁️ Document Preview"):
        content_result = st.session_state.doc_manager.get_document_content(file_hash)

        if content_result['success']:
            doc_info = content_result['document_info']

            st.markdown(f"**📄 {doc_info['filename']}**")

            # Document metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Size", utils.format_file_size(doc_info['file_size']))
            with col2:
                st.metric("Type", doc_info['file_extension'])
            with col3:
                st.metric("Status", doc_info.get('processing_status', 'Unknown'))

            # Content preview
            st.markdown("**📖 Content Preview:**")
            content_preview = content_result['content_preview']

            if content_preview:
                st.text_area(
                    "Preview",
                    value=content_preview,
                    height=400,
                    disabled=True
                )

                if content_result.get('truncated'):
                    st.info("Preview truncated. Full content available in the document.")
            else:
                st.warning("No content preview available for this document type.")

        else:
            st.error(f"Failed to preview document: {content_result['message']}")

def edit_document_metadata(file_hash):
    """Edit document metadata"""
    with st.modal("📝 Edit Document Metadata"):
        documents = st.session_state.doc_manager.get_document_list()
        doc = next((d for d in documents if d['file_hash'] == file_hash), None)

        if not doc:
            st.error("Document not found!")
            return

        st.markdown(f"**📄 Editing: {doc['filename']}**")

        with st.form("edit_metadata_form"):
            new_description = st.text_area(
                "Description",
                value=doc.get('description', ''),
                help="Update document description"
            )

            new_tags = st.text_input(
                "Tags",
                value=", ".join(doc.get('tags', [])),
                help="Comma-separated tags"
            )

            if st.form_submit_button("💾 Save Changes", type="primary"):
                # Update description
                if new_description != doc.get('description', ''):
                    doc['description'] = new_description

                # Update tags
                if new_tags:
                    tags = [tag.strip() for tag in new_tags.split(',') if tag.strip()]
                    st.session_state.doc_manager.add_document_tags(file_hash, tags)

                st.success("Metadata updated successfully!")
                time.sleep(1)
                st.rerun()

def reprocess_document(file_hash):
    """Reprocess a single document"""
    documents = st.session_state.doc_manager.get_document_list()
    doc = next((d for d in documents if d['file_hash'] == file_hash), None)

    if not doc:
        st.error("Document not found!")
        return

    with st.spinner(f"Reprocessing {doc['filename']}..."):
        process_document_for_rag(file_hash, doc['file_path'])

    st.success(f"✅ Reprocessed {doc['filename']}")
    time.sleep(1)
    st.rerun()

def reprocess_all_documents(documents):
    """Reprocess all documents"""
    if not documents:
        st.warning("No documents to reprocess!")
        return

    with st.spinner("Reprocessing all documents..."):
        progress_bar = st.progress(0)

        for i, doc in enumerate(documents):
            process_document_for_rag(doc['file_hash'], doc['file_path'])
            progress_bar.progress((i + 1) / len(documents))

        progress_bar.progress(1.0)

    st.success(f"✅ Reprocessed {len(documents)} documents")
    time.sleep(1)
    st.rerun()

def export_document(file_hash):
    """Export document"""
    documents = st.session_state.doc_manager.get_document_list()
    doc = next((d for d in documents if d['file_hash'] == file_hash), None)

    if not doc:
        st.error("Document not found!")
        return

    # Read and provide download
    try:
        with open(doc['file_path'], 'rb') as f:
            file_data = f.read()

        st.download_button(
            label=f"📥 Download {doc['filename']}",
            data=file_data,
            file_name=doc['filename'],
            mime=doc.get('mime_type', 'application/octet-stream')
        )
    except Exception as e:
        st.error(f"Export failed: {str(e)}")

def export_document_list(documents):
    """Export document list as CSV"""
    if not documents:
        st.warning("No documents to export!")
        return

    # Create DataFrame
    export_data = []
    for doc in documents:
        export_data.append({
            'Filename': doc.get('filename', ''),
            'Type': doc.get('file_extension', ''),
            'Size': utils.format_file_size(doc.get('file_size', 0)),
            'Upload Date': utils.format_timestamp(doc.get('upload_timestamp', ''), 'date'),
            'Status': doc.get('processing_status', ''),
            'Chunks': doc.get('chunk_count', 0),
            'Description': doc.get('description', ''),
            'Tags': ', '.join(doc.get('tags', []))
        })

    df = pd.DataFrame(export_data)

    # Provide download
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Document List (CSV)",
        data=csv_data,
        file_name=f"document_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def delete_document(file_hash):
    """Delete document with confirmation"""
    documents = st.session_state.doc_manager.get_document_list()
    doc = next((d for d in documents if d['file_hash'] == file_hash), None)

    if not doc:
        st.error("Document not found!")
        return

    # Confirmation dialog
    with st.modal(f"🗑️ Delete Document"):
        st.warning(f"Are you sure you want to delete **{doc['filename']}**?")
        st.markdown("This action cannot be undone.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("❌ Cancel", width="stretch"):
                st.rerun()

        with col2:
            if st.button("🗑️ Delete", type="primary", width="stretch"):
                # Remove from RAG system
                if st.session_state.rag_engine:
                    st.session_state.rag_engine.remove_document(file_hash)

                # Delete from document manager
                result = st.session_state.doc_manager.delete_document(file_hash)

                if result['success']:
                    st.success(f"✅ {result['message']}")
                else:
                    st.error(f"❌ {result['message']}")

                time.sleep(1)
                st.rerun()

def render_storage_stats():
    """Render storage statistics"""
    st.markdown("### 📊 Storage Statistics")

    stats = st.session_state.doc_manager.get_storage_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Documents", stats['total_documents'])

    with col2:
        st.metric("Total Size", stats['readable_size'])

    with col3:
        processed_docs = len([d for d in st.session_state.doc_manager.get_document_list()
                            if d.get('processing_status') == 'completed'])
        st.metric("Processed", processed_docs)

    with col4:
        if st.session_state.rag_engine:
            chunk_count = st.session_state.rag_engine.get_chunk_count()
            st.metric("Total Chunks", chunk_count)
        else:
            st.metric("Total Chunks", "N/A")

    # File type breakdown
    if stats['file_types']:
        st.markdown("#### 📁 File Types")

        # Create a simple chart data
        chart_data = []
        for file_type, type_stats in stats['file_types'].items():
            chart_data.append({
                'Type': file_type,
                'Count': type_stats['count'],
                'Size': utils.format_file_size(type_stats['size'])
            })

        df = pd.DataFrame(chart_data)
        st.dataframe(df, width="stretch")

def render_documents_page():
    """Main documents page renderer"""
    st.markdown("# 📁 Document Management")
    st.markdown("Upload, manage, and process your documents for RAG.")

    # Tab layout
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "📚 Library", "📊 Statistics"])

    with tab1:
        render_document_upload()

    with tab2:
        render_document_list()

    with tab3:
        render_storage_stats()

# Render the documents page
render_documents_page()