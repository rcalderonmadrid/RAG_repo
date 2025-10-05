import os
import logging
import time
import tempfile
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import Document

import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    """Enhanced RAG engine with document management and improved functionality"""

    def __init__(self, config):
        self.config = config
        self.documents = []
        self.vectorstore = None
        self.llm = None
        self.chain = None
        self.document_metadata = {}

        self._initialize_llm()
        self._initialize_embeddings()
        logger.info("RAG engine initialized successfully")

    def _initialize_llm(self):
        """Initialize the LLM with streaming capability"""
        try:
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            self.llm = ChatOllama(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                callback_manager=callback_manager
            )
            logger.info(f"LLM initialized: {self.config.llm_model}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise

    def _initialize_embeddings(self):
        """Initialize embeddings model"""
        try:
            self.embeddings = OllamaEmbeddings(model=self.config.embedding_model)
            logger.info(f"Embeddings initialized: {self.config.embedding_model}")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            raise

    def load_document(self, file_path: str, file_type: str = None) -> Dict[str, Any]:
        """Load a single document and return metadata"""
        try:
            if file_type is None:
                file_type = Path(file_path).suffix.lower()

            # Choose appropriate loader based on file type
            if file_type == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_type == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_type in ['.docx', '.doc']:
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            pages = loader.load()

            # Calculate document hash for deduplication
            doc_hash = self._calculate_document_hash(file_path)

            # Store metadata
            metadata = {
                'file_path': file_path,
                'file_name': Path(file_path).name,
                'file_type': file_type,
                'page_count': len(pages),
                'doc_hash': doc_hash,
                'upload_time': time.time(),
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }

            self.document_metadata[doc_hash] = metadata
            logger.info(f"Document loaded: {metadata['file_name']} ({len(pages)} pages)")

            return {
                'success': True,
                'pages': pages,
                'metadata': metadata
            }

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'metadata': None
            }

    def _calculate_document_hash(self, file_path: str) -> str:
        """Calculate hash of document for deduplication"""
        hash_md5 = hashlib.md5()
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def add_documents(self, documents: List[Document], doc_hash: str = None):
        """Add documents to the collection"""
        if doc_hash and doc_hash in [doc.metadata.get('doc_hash') for doc in self.documents]:
            logger.warning(f"Document with hash {doc_hash} already exists. Skipping.")
            return False

        # Add doc_hash to metadata if provided
        if doc_hash:
            for doc in documents:
                doc.metadata['doc_hash'] = doc_hash

        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents to collection")
        return True

    def remove_document(self, doc_hash: str) -> bool:
        """Remove a document by its hash"""
        try:
            # Remove from documents list
            initial_count = len(self.documents)
            self.documents = [doc for doc in self.documents if doc.metadata.get('doc_hash') != doc_hash]
            removed_count = initial_count - len(self.documents)

            # Remove from metadata
            if doc_hash in self.document_metadata:
                del self.document_metadata[doc_hash]

            # Rebuild vectorstore if documents were removed
            if removed_count > 0:
                self._rebuild_vectorstore()
                logger.info(f"Removed document with hash {doc_hash} ({removed_count} chunks)")
                return True
            else:
                logger.warning(f"No document found with hash {doc_hash}")
                return False

        except Exception as e:
            logger.error(f"Error removing document {doc_hash}: {e}")
            return False

    def split_documents(self, pages: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = text_splitter.split_documents(pages)
        logger.info(f"Created {len(chunks)} document chunks")
        return chunks

    def create_vectorstore(self) -> bool:
        """Create or update the vector database"""
        try:
            if not self.documents:
                logger.warning("No documents to create vectorstore")
                return False

            # Remove existing database
            if os.path.exists(self.config.persist_directory):
                import shutil
                logger.info(f"Removing existing vectorstore at {self.config.persist_directory}")
                shutil.rmtree(self.config.persist_directory, ignore_errors=True)

            # Create new vectorstore
            logger.info("Creating new vectorstore...")
            self.vectorstore = Chroma.from_documents(
                documents=self.documents,
                embedding=self.embeddings,
                persist_directory=self.config.persist_directory
            )
            self.vectorstore.persist()

            # Setup chain
            self._setup_chain()

            logger.info(f"Vectorstore created successfully with {len(self.documents)} document chunks")
            return True

        except Exception as e:
            logger.error(f"Error creating vectorstore: {e}")
            return False

    def _rebuild_vectorstore(self):
        """Rebuild the vectorstore after document changes"""
        if self.documents:
            return self.create_vectorstore()
        else:
            # No documents left, clear vectorstore
            if os.path.exists(self.config.persist_directory):
                import shutil
                shutil.rmtree(self.config.persist_directory, ignore_errors=True)
            self.vectorstore = None
            self.chain = None

    def load_existing_vectorstore(self) -> bool:
        """Load existing vectorstore from disk"""
        try:
            if os.path.exists(self.config.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.config.persist_directory,
                    embedding_function=self.embeddings
                )
                self._setup_chain()
                logger.info("Loaded existing vectorstore")
                return True
            else:
                logger.info("No existing vectorstore found")
                return False
        except Exception as e:
            logger.error(f"Error loading vectorstore: {e}")
            return False

    def _setup_chain(self):
        """Set up the RAG chain for question answering"""
        if not self.vectorstore:
            logger.warning("No vectorstore available for chain setup")
            return

        # Create retriever with search parameters
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.retrieval_k}
        )

        # Define the prompt template
        template = """
        ### INSTRUCTIONS:
        You are an AI expert assistant for document analysis and question answering.
        Base your answers strictly on the provided context from the uploaded documents.
        Be professional, accurate, and helpful in your responses.

        Guidelines:
        1. Read the question and context thoroughly before answering
        2. Begin with a friendly tone and acknowledge the user's question
        3. Provide detailed, helpful responses based on the context
        4. Use precise terminology and include relevant details
        5. When helpful, provide step-by-step explanations or examples
        6. Reference sources inline when possible (e.g., [Document Title, Page X])
        7. If you cannot find the answer in the context, state this clearly

        Additional constraints:
        - Do not invent information not present in the context
        - Keep responses well-structured and easy to read
        - Use appropriate formatting (bullet points, sections) when helpful
        - If there are conflicting statements in documents, acknowledge this

        ### Question: {question}
        ### Context: {context}
        ### Response:
        """

        prompt = PromptTemplate.from_template(template)

        # Create the chain
        self.chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        logger.info("RAG chain setup complete")

    def answer_question(self, question: str, retrieval_k: Optional[int] = None) -> Dict[str, Any]:
        """Answer a question using the RAG chain"""
        if not self.chain:
            return {
                'success': False,
                'answer': "RAG system not properly initialized. Please upload documents first.",
                'sources': [],
                'processing_time': 0
            }

        start_time = time.time()
        logger.info(f"Processing question: {question}")

        # Use provided retrieval_k or default from config
        k_value = retrieval_k if retrieval_k is not None else self.config.retrieval_k

        try:
            # Get context documents for source tracking
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k_value}
            )
            context_docs = retriever.get_relevant_documents(question)

            # Generate answer
            answer = self.chain.invoke(question)

            # Extract source information
            sources = []
            for doc in context_docs:
                source_info = {
                    'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'metadata': doc.metadata
                }
                sources.append(source_info)

            processing_time = time.time() - start_time

            return {
                'success': True,
                'answer': answer,
                'sources': sources,
                'processing_time': processing_time
            }

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error answering question: {e}")
            return {
                'success': False,
                'answer': f"Error processing your question: {str(e)}",
                'sources': [],
                'processing_time': processing_time
            }

    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search documents for relevant chunks"""
        if not self.vectorstore:
            return []

        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            results = []
            for doc in docs:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'relevance_score': getattr(doc, '_distance', None)
                })
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded documents"""
        stats = {
            'total_documents': len(self.document_metadata),
            'total_chunks': len(self.documents),
            'document_types': {},
            'total_size': 0,
            'documents': []
        }

        for doc_hash, metadata in self.document_metadata.items():
            # Count document types
            file_type = metadata.get('file_type', 'unknown')
            stats['document_types'][file_type] = stats['document_types'].get(file_type, 0) + 1

            # Sum file sizes
            stats['total_size'] += metadata.get('file_size', 0)

            # Add document info
            stats['documents'].append({
                'name': metadata.get('file_name', 'Unknown'),
                'type': file_type,
                'pages': metadata.get('page_count', 0),
                'size': metadata.get('file_size', 0),
                'upload_time': metadata.get('upload_time', 0),
                'doc_hash': doc_hash
            })

        return stats

    def get_chunk_count(self) -> int:
        """Get the number of document chunks"""
        return len(self.documents)

    def is_initialized(self) -> bool:
        """Check if the RAG system is properly initialized"""
        return self.vectorstore is not None and self.chain is not None

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the RAG system"""
        status = {
            'llm_available': False,
            'embeddings_available': False,
            'vectorstore_available': False,
            'documents_loaded': False,
            'chain_ready': False,
            'overall_status': 'unhealthy'
        }

        try:
            # Check LLM
            if self.llm:
                test_response = self.llm.invoke("Hello")
                status['llm_available'] = bool(test_response)

            # Check embeddings
            if self.embeddings:
                test_embedding = self.embeddings.embed_query("test")
                status['embeddings_available'] = bool(test_embedding)

            # Check vectorstore
            status['vectorstore_available'] = self.vectorstore is not None

            # Check documents
            status['documents_loaded'] = len(self.documents) > 0

            # Check chain
            status['chain_ready'] = self.chain is not None

            # Overall status
            if all([status['llm_available'], status['embeddings_available'],
                   status['vectorstore_available'], status['chain_ready']]):
                status['overall_status'] = 'healthy'
            elif any([status['llm_available'], status['embeddings_available']]):
                status['overall_status'] = 'partial'

        except Exception as e:
            logger.error(f"Health check error: {e}")
            status['error'] = str(e)

        return status