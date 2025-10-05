# 🌊 RAG Intelligence Hub

An advanced Retrieval Augmented Generation (RAG) system built with Streamlit, offering comprehensive document management, conversational AI, and analytics capabilities.

## ✨ Features

### 📊 **Dashboard**
- System overview with key metrics
- Recent activity tracking
- Quick access to all features
- Usage analytics at a glance

### 💬 **Intelligent Chat Interface**
- Natural language querying of your documents
- Real-time response generation with processing time display
- Source citations and reference tracking
- Conversation saving and bookmarking
- Advanced query options (temperature, retrieval count)
- Search within conversations
- Suggested questions based on document content

### 📁 **Document Management**
- **Multi-format support**: PDF, TXT, DOCX, DOC, MD
- **Drag-and-drop upload** with file validation
- **Document preview** with content extraction
- **Bulk operations**: Upload multiple files, reprocess all documents
- **Smart organization**: Tags, descriptions, search functionality
- **Processing status tracking**: Monitor document processing progress
- **Storage analytics**: File type distribution, size analysis
- **Cleanup tools**: Remove orphaned files, manage storage

### 💾 **Conversation History**
- **Complete conversation management**: Save, view, search, and organize chats
- **Export capabilities**: JSON, TXT, Markdown formats
- **Bookmarking system**: Mark important conversations
- **Bulk operations**: Export all conversations, cleanup old chats
- **Advanced search**: Find conversations by content or metadata
- **Analytics**: Response time tracking, conversation patterns

### 📈 **Analytics & Insights**
- **System performance**: Health checks, processing efficiency
- **Usage patterns**: Daily activity, response time analysis
- **Document insights**: Type distribution, processing success rates
- **Conversation analytics**: Message volume, engagement patterns
- **Performance metrics**: Response times, system health
- **Export capabilities**: Generate comprehensive usage reports

### ⚙️ **Advanced Settings**
- **Model configuration**: LLM and embedding model selection
- **RAG parameters**: Chunk size, overlap, retrieval count
- **UI customization**: Theme, display preferences
- **System management**: Health checks, cleanup tools
- **Import/Export**: Backup and restore configurations
- **Real-time validation**: Settings verification and warnings

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Python 3.8+**: Ensure you have Python installed
3. **Git**: For cloning the repository

### Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd ragollama/app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and start Ollama**:
   ```bash
   # Install Ollama (follow instructions for your OS)
   ollama serve
   ```

4. **Pull required models**:
   ```bash
   # Language model (choose one)
   ollama pull qwen2.5:1.5b        # Lightweight, fast
   ollama pull llama3.1:8b         # Balanced performance
   ollama pull mistral:7b          # Good for analysis

   # Embedding model
   ollama pull all-minilm:latest   # Recommended
   # or
   ollama pull nomic-embed-text    # Alternative
   ```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. **Initial Setup**
1. Navigate to **Settings** → **Models**
2. Verify Ollama connection (click "Test Connection")
3. Configure your preferred LLM and embedding models
4. Adjust RAG parameters if needed
5. Click "Initialize RAG System"

### 2. **Upload Documents**
1. Go to **Documents** → **Upload**
2. Drag and drop or select your files
3. Add descriptions and tags (optional)
4. Enable "Auto-process after upload"
5. Click "Upload Documents"

### 3. **Start Chatting**
1. Navigate to **Chat**
2. Use suggested questions or ask your own
3. Adjust settings using the Options panel
4. Save important conversations
5. Export conversations as needed

### 4. **Manage Your Data**
- **Documents**: Review, edit metadata, reprocess files
- **Conversations**: Search, bookmark, export chat history
- **Analytics**: Monitor usage patterns and performance
- **Settings**: Fine-tune system parameters

## 🔧 Configuration

### Model Settings
- **LLM Model**: Choose your language model (affects response quality)
- **Temperature**: Control response creativity (0.0-2.0)
- **Embedding Model**: Vector representation of documents
- **Max Tokens**: Maximum response length

### RAG Parameters
- **Chunk Size**: Document segmentation size (default: 1000)
- **Chunk Overlap**: Overlap between segments (default: 50)
- **Retrieval K**: Number of relevant chunks to retrieve (default: 3)

### Advanced Options
- **Source Citations**: Include document references
- **Streaming**: Real-time response generation
- **Analytics**: Usage tracking and metrics
- **Auto-save**: Automatic conversation persistence

## 🏗️ Architecture

```
RAG Intelligence Hub
├── app.py                 # Main Streamlit application
├── rag_engine.py         # Core RAG functionality
├── document_manager.py   # Document upload/management
├── conversation_manager.py # Chat history management
├── config.py             # Configuration management
├── utils.py              # Utility functions
└── pages/                # Individual page components
    ├── chat.py           # Chat interface
    ├── documents.py      # Document management
    ├── conversations.py  # Conversation history
    ├── analytics.py      # Analytics dashboard
    └── settings.py       # Settings and configuration
```

## 📊 Data Storage

### File Organization
```
ragollama/app/
├── uploaded_documents/    # User-uploaded files
├── conversations/         # Saved chat history
├── chroma_db/            # Vector database
└── config.json          # Application configuration
```

### Data Persistence
- **Documents**: Stored with metadata and processing status
- **Conversations**: JSON format with full message history
- **Vector Database**: ChromaDB for document embeddings
- **Configuration**: JSON file for all settings

## 🔒 Security & Privacy

- **Local Processing**: All data stays on your machine
- **No External APIs**: Uses local Ollama models only
- **Document Security**: Files stored locally with hash verification
- **Privacy Protection**: No data transmitted to external services

## 🛠️ Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   - Ensure Ollama is running: `ollama serve`
   - Check URL in settings (default: http://localhost:11434)
   - Verify firewall settings

2. **Model Not Found**
   - Pull required models: `ollama pull <model-name>`
   - Check available models: `ollama list`
   - Verify model names in settings

3. **Document Processing Failed**
   - Check file format (PDF, TXT, DOCX supported)
   - Verify file isn't password-protected
   - Check file size limits (default: 50MB)

4. **Slow Performance**
   - Use smaller models for faster responses
   - Reduce chunk size or retrieval count
   - Monitor system resources

### Health Checks
Use the built-in health check in **Settings** → **System** to diagnose issues.

## 🆚 Migration from Gradio

This application replaces the original Gradio-based RAG system with enhanced features:

### Key Improvements
- **Multi-page interface** vs single page
- **Document management** vs fixed document list
- **Conversation history** vs no persistence
- **Analytics dashboard** vs no insights
- **Advanced configuration** vs basic settings
- **Export capabilities** vs no export options
- **Real-time status** vs minimal feedback

### Migration Steps
1. Export your documents from the old system
2. Upload them to the new document manager
3. Configure models and settings as needed
4. Start using the enhanced features

## 📈 Performance Tips

### Optimization Strategies
- **Model Selection**: Balance quality vs speed based on your needs
- **Chunk Configuration**: Optimize for your document types
- **Retrieval Tuning**: Adjust K value based on document complexity
- **System Resources**: Monitor RAM and CPU usage
- **Regular Maintenance**: Clean up old conversations and orphaned files

### Recommended Configurations

**For Speed** (Development/Testing):
- LLM: `qwen2.5:1.5b`
- Temperature: 0.1
- Chunk Size: 800
- Retrieval K: 3

**For Quality** (Production):
- LLM: `llama3.1:8b` or `mistral:7b`
- Temperature: 0.1
- Chunk Size: 1000
- Retrieval K: 5

**For Specialized Content**:
- Increase chunk size for technical documents
- Higher retrieval K for complex queries
- Lower temperature for factual responses

## 🤝 Contributing

This is a complete RAG solution. For enhancements:

1. **Feature Requests**: Document specific needs
2. **Bug Reports**: Include system info and steps to reproduce
3. **Performance Issues**: Share configuration and document types
4. **Model Integration**: Test with new Ollama models

## 📄 License

This project is designed for educational and research purposes. Please ensure compliance with Ollama's terms of service and any document copyright restrictions.

## 🙏 Acknowledgments

- **Streamlit**: Modern web app framework
- **LangChain**: RAG implementation framework
- **Ollama**: Local LLM serving platform
- **ChromaDB**: Vector database solution
- **Plotly**: Interactive analytics visualizations

---

**🌊 RAG Intelligence Hub** - Your complete document intelligence solution!