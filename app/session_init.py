import streamlit as st
from rag_engine import RAGEngine
from document_manager import DocumentManager
from conversation_manager import ConversationManager
from config import Config

def init_session_state():
    """Initialize session state variables - can be called from any page"""
    if 'config' not in st.session_state:
        st.session_state.config = Config()
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'doc_manager' not in st.session_state:
        st.session_state.doc_manager = DocumentManager()
    if 'conv_manager' not in st.session_state:
        st.session_state.conv_manager = ConversationManager()
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True

def ensure_session_state():
    """Ensure session state is initialized - safe to call multiple times"""
    if 'initialized' not in st.session_state:
        init_session_state()

def process_pending_documents():
    """Process any pending documents when RAG engine is available"""
    if not hasattr(st.session_state, 'rag_engine') or not st.session_state.rag_engine:
        return False

    if not hasattr(st.session_state, 'doc_manager'):
        return False

    try:
        # Get all documents
        documents = st.session_state.doc_manager.get_all_documents()
        pending_docs = []

        for doc_id, metadata in documents.items():
            if not metadata.get('processed', False) and metadata.get('processing_status') == 'pending':
                pending_docs.append({
                    'file_hash': doc_id,
                    'file_path': metadata['file_path']
                })

        if pending_docs:
            # Import the function here to avoid circular imports
            from pages.documents import process_document_for_rag

            for doc in pending_docs:
                try:
                    process_document_for_rag(doc['file_hash'], doc['file_path'])
                except Exception as e:
                    st.error(f"Error processing document {doc['file_hash']}: {str(e)}")

            return len(pending_docs)

        return 0

    except Exception as e:
        st.error(f"Error processing pending documents: {str(e)}")
        return False