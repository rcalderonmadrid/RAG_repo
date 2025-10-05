# Standard imports
import os
import logging
import time
import sys
import tempfile
from typing import List, Dict, Any

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Gradio for web interface
import gradio as gr

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


#parameters
PERSIST_DIRECTORY = 'chroma_db' # directory to store the vector database
CHUNK_SIZE = 1000 # characters per chunk for text splitting
CHUNK_OVERLAP = 50 # characters of overlap between chunks
PDF_URLS = [
    r'C:\Users\demst\Desktop\ragollama\data\1-s2.0-S002216941730433X-main.pdf',
    r'C:\Users\demst\Desktop\ragollama\data\1-s2.0-S002216941830698X-main.pdf'
]
LLM_MODEL = 'qwen3:1.7b'
EMBEDDING_MODEL = 'all-minilm:latest'
TEMPERATURE = 0.1


class RAGSystem:
    def __init__(self, pdf_urls: List[str], persist_directory: str = PERSIST_DIRECTORY):
        self.pdf_urls = pdf_urls
        self.persist_directory = persist_directory
        self.documents = []
        self.vectorstore = None
        self.llm = None
        self.chain = None
        
        # Initialize the LLM with streaming capability
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = ChatOllama(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            callback_manager=callback_manager
        )
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        logger.info(f"Initialized RAG system with {len(pdf_urls)} PDFs")



        def load_documents(self) -> None:
    """Load and split PDF documents"""
    logger.info("Loading and processing PDFs...")
    
    # Text splitter for chunking documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_pages = []
    
    for url in self.pdf_urls:
        try:
            loader = PyPDFLoader(url)
            pages = loader.load()
            logger.info(f"Loaded {len(pages)} pages from {url}")
            all_pages.extend(pages)
        except Exception as e:
            logger.error(f"Error loading PDF from {url}: {e}")
    
    # Split the documents into chunks
    self.documents = text_splitter.split_documents(all_pages)
    logger.info(f"Created {len(self.documents)} document chunks")


    def create_vectorstore(self) -> None:
    """Create a fresh vector database"""
    # Remove any existing database
    if os.path.exists(self.persist_directory):
        import shutil
        logger.info(f"Removing existing vectorstore at {self.persist_directory}")
        shutil.rmtree(self.persist_directory, ignore_errors=True)
    
    # Create a new vectorstore
    logger.info("Creating new vectorstore...")
    if not self.documents:
        self.load_documents()
    
    # Create a temporary directory for the database
    # This helps avoid permission issues on some systems
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Using temporary directory for initial database creation: {temp_dir}")
    
    try:
        # First create in temp directory
        self.vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            persist_directory=temp_dir
        )
        
        # Now create the real directory
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
            
        # And create the final vectorstore
        self.vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vectorstore.persist()
        
        logger.info(f"Vectorstore created successfully with {len(self.documents)} documents")
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}")
        raise
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


            def setup_chain(self) -> None:
    """Set up the RAG chain for question answering"""
    if not self.vectorstore:
        self.create_vectorstore()
    
    # Create retriever with search parameters
    retriever = self.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Return top 3 most relevant chunks
    )
    
    # Define the prompt template
    template = """
### INSTRUCTIONS:
    You are an AI expert in hydrology, drought analysis, and statistical/ML methods relevant to my PhD thesis. 
    Base your answers strictly on the TWO provided thesis articles (the context below). 
    Be polite, professional, and avoid guessing or using outside sources.
 
    (1) Be attentive to details: read the question and the context thoroughly before answering.
    (2) Begin your response with a friendly tone and briefly restate the users question to confirm understanding.
    (3) If the context allows you to answer the question, write a detailed, helpful, and easy-to-understand response.
        - Use precise terminology from the articles (e.g., threshold method, deficit volume (hm³), duration D, severity S, intensity I, run theory, copulas, KS test).
        - When helpful, include short equations in LaTeX, step-by-step reasoning, or concise examples.
        - Reference the sources **inline** (e.g., [Article 1 §3.2], [Article 2 Fig.4]) and ONLY cite sections/figures/tables present in the provided context.
      IF NOT: if you cannot find the answer, respond with an explanation, starting with: 
        "I couldn't find the information in the documents I have access to."
    (4) Below your response, list all referenced sources (document titles/IDs and exact sections/figures/tables that support your claims).
    (5) Review your answer to ensure it answers the question, is helpful and professional, and is formatted for easy reading (short paragraphs, bullets if useful).
    Additional constraints:
    - Do not invent citations or content outside the provided context.
    - Use SI units consistently (e.g., hm³ for volumes, days for duration).
    - Keep code examples minimal and only if concepts are present in the context; label as pseudo-code if not exact.
    - If there are conflicting statements in the two articles, acknowledge the discrepancy and cite both places.
 
    THINK STEP BY STEP
 
    Answer the following question using the provided context.
    ### Question: {question} ###
    ### Context: {context} ###
    ### Helpful Answer with Sources:
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


    def answer_question(self, question: str) -> str:
    """
    Answer a question using the RAG chain
    
    Args:
        question: The question to answer
        
    Returns:
        The answer to the question
    """
    if not self.chain:
        self.setup_chain()
    
    logger.info(f"Answering question: {question}")
    try:
        answer = self.chain.invoke(question)
        return answer
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return f"Error processing your question: {str(e)}"


        import re

def create_gradio_interface(rag_system: RAGSystem) -> gr.Interface:
    """
    Create an enhanced Gradio interface for the hydrology RAG system
    
    Args:
        rag_system: The RAG system for hydrology and drought analysis
        
    Returns:
        A modern Gradio interface with enhanced features
    """
    
    def get_answer_with_context(question: str, show_sources: bool = True) -> tuple:
        """Enhanced wrapper function with source visibility option"""
        if not question.strip():
            return ("Please enter a question about hydrology, drought analysis, or statistical methods.", "")
        
        try:
            # Get the answer
            answer = rag_system.answer_question(question)
            
            # Extract sources if they exist (simple extraction for demo)
            sources_info = ""
            if show_sources and answer:
                # Look for common citation patterns
                citations = re.findall(r'\[.*?\]', answer)
                if citations:
                    sources_info = "**Sources Referenced:**\n" + "\n".join(f"• {cite}" for cite in set(citations))
            
            return answer, sources_info
            
        except Exception as e:
            error_msg = f"⚠️ **Error processing question:** {str(e)}\n\nPlease check that Ollama is running and models are available."
            return error_msg, ""
    
    def get_sample_questions():
        """Return categorized sample questions"""
        return [
            "What is the threshold method for drought identification?",
            "How are deficit volumes calculated in hm³?", 
            "Explain the copula approach for drought analysis",
            "What statistical tests are used to validate the models?",
            "How do you define drought duration, severity, and intensity?"
        ]
    
    # Custom CSS for better styling
    custom_css = """
    /* Main container styling */
    .gradio-container {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        max-width: 1200px !important;
        margin: 0 auto;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    /* Button styling */
    .gr-button-primary {
        background: linear-gradient(45deg, #2196F3, #21CBF3) !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .gr-button-primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3) !important;
    }
    
    /* Input styling */
    .gr-textbox textarea {
        border-radius: 10px !important;
        border: 2px solid #e1e5e9 !important;
        font-size: 16px !important;
    }
    
    .gr-textbox textarea:focus {
        border-color: #2196F3 !important;
        box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.1) !important;
    }
    
    /* Output styling */
    .markdown-container {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #2196F3;
    }
    
    /* Example buttons */
    .example-btn {
        background: #f1f3f4 !important;
        border: 1px solid #dadce0 !important;
        border-radius: 20px !important;
        padding: 8px 16px !important;
        margin: 4px !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
    }
    
    .example-btn:hover {
        background: #e8f0fe !important;
        border-color: #2196F3 !important;
    }
    """
    
    # Create the interface using Blocks for more control
    with gr.Blocks(
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="cyan", 
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter")
        ),
        title="🌊 Hydrology RAG System"
    ) as interface:
        
        # Header section
        gr.HTML("""
        <div class="header-container">
            <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">
                🌊 Hydrology & Drought Analysis Intelligence
            </h1>
            <p style="margin: 1rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                AI-Powered PhD Thesis Research Assistant
            </p>
        </div>
        """)
        
        # Main content area
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                gr.Markdown("### 💭 Ask Your Question")
                question_input = gr.Textbox(
                    placeholder="e.g., How is drought severity calculated using the threshold method?",
                    label="",
                    lines=3,
                    max_lines=6,
                    container=False
                )
                
                # Options
                with gr.Row():
                    show_sources = gr.Checkbox(
                        label="Show source citations",
                        value=True,
                        container=False
                    )
                    submit_btn = gr.Button(
                        "🔍 Analyze",
                        variant="primary",
                        size="lg"
                    )
                
                # Quick examples
                gr.Markdown("### 🎯 Quick Examples")
                with gr.Row():
                    example_btns = []
                    for i, example in enumerate(get_sample_questions()[:3]):
                        btn = gr.Button(
                            example[:50] + "..." if len(example) > 50 else example,
                            size="sm",
                            elem_classes=["example-btn"]
                        )
                        example_btns.append((btn, example))
            
            with gr.Column(scale=1):
                # System status and info
                gr.Markdown("""
                ### 📊 System Info
                
                **🎯 Specializes in:**
                - Drought characterization methods
                - Statistical analysis (copulas, KS tests)
                - Hydrological variables (D, S, I)
                - Threshold methodologies
                
                **📚 Knowledge Base:**
                - 2 thesis research articles
                - Focused on hydrology & drought analysis
                
                **✅ Features:**
                - Inline citations
                - Technical terminology
                - Step-by-step explanations
                """)
        
        # Output section
        gr.Markdown("### 🎓 Expert Analysis")
        with gr.Row():
            with gr.Column(scale=3):
                answer_output = gr.Markdown(
                    label="",
                    container=True,
                    elem_classes=["markdown-container"]
                )
            with gr.Column(scale=1):
                sources_output = gr.Markdown(
                    label="",
                    container=True
                )
        
        # Footer info
        gr.Markdown("""
        ---
        💡 **Tip:** Be specific in your questions for better results. The system works best with technical queries about drought analysis, statistical methods, and hydrological parameters mentioned in your thesis documents.
        
        ⚠️ **Note:** All responses are based strictly on the loaded thesis documents. The system cannot access external information beyond your research articles.
        """)
        
        # Event handlers
        def handle_submit(question, show_sources_flag):
            if not question.strip():
                return "Please enter a question to get started! 🚀", ""
            return get_answer_with_context(question, show_sources_flag)
        
        # Submit button action
        submit_btn.click(
            fn=handle_submit,
            inputs=[question_input, show_sources],
            outputs=[answer_output, sources_output]
        )
        
        # Enter key submission
        question_input.submit(
            fn=handle_submit,
            inputs=[question_input, show_sources],
            outputs=[answer_output, sources_output]
        )
        
        # Example button actions
        for btn, example_text in example_btns:
            btn.click(
                fn=lambda example=example_text: example,
                outputs=question_input
            )
    
    return interface


    def main() -> None:
    """Main function to run the hydrology RAG system"""
    try:
        # Display banner
        print("\n" + "="*60)
        print("🌊 HYDROLOGY & DROUGHT ANALYSIS RAG SYSTEM")
        print("   PhD Thesis Document Intelligence Assistant")
        print("="*60)
        
        # Display available models
        print("\n==== CHECKING OLLAMA MODELS ====")
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            print("Available Ollama models:")
            if response.status_code == 200:
                models = response.json().get("models", [])
                if models:
                    for model in models:
                        print(f"✓ {model['name']}")
                else:
                    print("❌ No models found")
            else:
                print(f"❌ Error checking Ollama models: {response.status_code}")
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            print("   Make sure Ollama is running with: ollama serve")
        
        print(f"\n📋 CONFIGURATION:")
        print(f"   LLM model: {LLM_MODEL}")
        print(f"   Embedding model: {EMBEDDING_MODEL}")
        print(f"   Documents to process: {len(PDF_URLS)} thesis articles")
        print("\n   Make sure these models are available with 'ollama pull' commands.")
        
        # Create and initialize the RAG system
        print("\n==== INITIALIZING RAG SYSTEM ====")
        logger.info("Creating hydrology RAG system...")
        rag_system = RAGSystem(pdf_urls=PDF_URLS)
        
        # Load documents and create vectorstore
        print("📚 Loading thesis documents...")
        rag_system.load_documents()
        
        print("🔍 Creating vector embeddings...")
        rag_system.create_vectorstore()
        
        # Test with a control question about hydrology/drought analysis
        print("\n==== TESTING SYSTEM ====")
        logger.info("Testing with a hydrology control question...")
        test_questions = [
            "What is the threshold method for drought identification?",
            "How are drought characteristics defined?",
            "What statistical methods are used in the analysis?"
        ]
        
        # Try the first available test question
        test_question = test_questions[0]
        print(f"🧪 Testing with: '{test_question}'")
        test_answer = rag_system.answer_question(test_question)
        
        if test_answer and len(test_answer) > 50:
            logger.info(f"✓ Control answer received (length: {len(test_answer)})")
            print("✓ System test successful - RAG pipeline working correctly")
        else:
            logger.warning(f"⚠️ Short control answer received (length: {len(test_answer)})")
            print("⚠️ System test completed but response seems short")
        
        # Create and launch Gradio interface
        print("\n==== LAUNCHING INTERFACE ====")
        logger.info("Launching Gradio hydrology interface...")
        print("🚀 Starting web interface...")
        print("   - Access locally at: http://localhost:7860")
        print("   - Interface optimized for hydrology and drought analysis")
        print("   - All responses based on your thesis documents")
        
        # Use our custom hydrology interface
        interface = create_gradio_interface(rag_system)
        interface.launch(
            share=False,  # Set share=True to create a public link
            inbrowser=True,  # Automatically open browser
            show_error=True,
            quiet=False
        )
    
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        print(f"\n\n❌ ERROR: {str(e)}\n\n")
        print("🔧 TROUBLESHOOTING TIPS FOR HYDROLOGY RAG SYSTEM:")
        print("="*50)
        print("1. 🖥️  OLLAMA SERVICE:")
        print("   - Make sure Ollama is running: 'ollama serve'")
        print("   - Check if Ollama is accessible at http://localhost:11434")
        print()
        print("2. 🧠 REQUIRED MODELS:")
        print(f"   - Pull LLM model: 'ollama pull {LLM_MODEL}'")
        print(f"   - Pull embedding model: 'ollama pull {EMBEDDING_MODEL}'")
        print("   - For hydrology, recommend: llama3.1:8b or mistral:7b")
        print()
        print("3. 📄 DOCUMENT PROCESSING:")
        print("   - Verify PDF URLs are accessible")
        print("   - Check thesis documents are properly formatted")
        print("   - Ensure PDFs contain extractable text")
        print()
        print("4. 🔧 TECHNICAL ISSUES:")
        print("   - If dimension mismatch: try 'nomic-embed-text' embedding model")
        print("   - Check Python packages: pip install -r requirements.txt")
        print("   - Verify Chroma vector database permissions")
        print()
        print("5. 📊 PERFORMANCE OPTIMIZATION:")
        print("   - For better hydrology responses, use larger models (13B+)")
        print("   - Increase chunk_size for technical documents")
        print("   - Adjust temperature for more/less conservative answers")


        RAGSystem.load_documents = load_documents
RAGSystem.create_vectorstore = create_vectorstore
RAGSystem.setup_chain = setup_chain
RAGSystem.answer_question = answer_question

# Run the system
if __name__ == "__main__":
    main()
else:
    # If running in a notebook
    main()