# 🤖 Multi-Tenant Chatbot Platform

A powerful RAG-based chatbot platform that allows users to create personalized AI assistants by uploading documents and building knowledge bases. Each user can create multiple bots with isolated knowledge bases, perfect for different use cases.

## ✨ Features

- **📁 Document Upload**: Support for PDF, DOCX, TXT, and MD files
- **🌐 URL Scraping**: Upload content from websites or sitemaps
- **🧠 Smart RAG Pipeline**: Optimized retrieval with citations and confidence scores
- **👥 Multi-Tenant**: Each user can create multiple bots with separate knowledge bases
- **🎨 Modern UI**: Beautiful TypeScript/React frontend with dark theme
- **⚡ Fast Processing**: Document processing in seconds, not minutes
- **📊 Analytics**: View chunk usage, confidence scores, and source citations
- **🔒 Data Isolation**: Complete separation between users and bots
- **🔄 Real-time Updates**: Live knowledge base statistics and refresh
- **📱 Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Start Guide

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd fusor-ai
```

### Step 2: Set Up Python Environment

**Create and activate virtual environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**Install Python dependencies:**
```bash
pip install -r requirements.txt
```

### Step 3: Set Up Frontend (TypeScript/React)

**Navigate to frontend directory:**
```bash
cd Frontend
```

**Install Node.js dependencies:**
```bash
npm install
```

**Go back to project root:**
```bash
cd ..
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

**Get your OpenAI API key:**
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy and paste it in your `.env` file

### Step 5: Start Qdrant Vector Database

**Option A: Using Docker (Recommended)**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Option B: Using pip**
```bash
pip install qdrant-client
python -m qdrant_client.http
```

### Step 6: Launch the Platform

**🚀 Start everything at once (Recommended):**
```bash
python start_full_platform.py
```

**Or start services separately:**
```bash
# Terminal 1: Start FastAPI backend
python main.py

# Terminal 2: Start TypeScript frontend
cd Frontend
npm run dev
```

### Step 7: Access the Platform

- **🎨 Main Interface**: http://localhost:3000 (TypeScript Frontend)
- **📡 API Documentation**: http://localhost:8000/docs
- **🔧 API Endpoints**: http://localhost:8000
- **🗄️ Qdrant Dashboard**: http://localhost:6333/dashboard

## 📖 How to Use

### Creating Your First Knowledge Base

1. **Open the Platform**
   - Go to http://localhost:3000
   - You'll see the "Upload Documents" tab

2. **Enter Your Details**
   - **Username**: Choose a unique username (e.g., `john_doe`)
   - **Bot Name**: Name your chatbot (e.g., `support_bot`)

3. **Upload Documents**
   - Drag and drop files into the upload zone, or click to browse
   - Supported formats: PDF, DOCX, TXT, MD
   - You can upload multiple files at once
   - Or enter a website URL to scrape content

4. **Process Documents**
   - Click "Create Knowledge Base"
   - Wait for processing to complete (usually 2-5 seconds)
   - You'll see a success message with chunk count

### Chatting with Your Bot

1. **Go to Chat Tab**
   - Click "Chat with Bot" tab
   - Select your knowledge base from the dropdown
   - Or enter your username and bot name manually

2. **Start Chatting**
   - Type your question in the message box
   - Press Enter or click "Send"
   - Get responses with citations and chunk usage info

3. **View Knowledge Base Info**
   - See your current knowledge base in the sidebar
   - View all available knowledge bases
   - Check chunk usage and statistics

## 🎯 Example Usage

### Creating Multiple Bots

**User: `alice`**
- **Bot 1**: `customer_support` (for customer service documents)
- **Bot 2**: `technical_docs` (for technical documentation)
- **Bot 3**: `faq_bot` (for frequently asked questions)

Each bot has its own isolated knowledge base!

### Sample Queries

- "What is this document about?"
- "Can you summarize the main points?"
- "What are the key features mentioned?"
- "How do I use this system?"
- "What technologies are used?"

## 🔧 Configuration

The platform uses a centralized configuration system in `config.py`:

### Key Settings

```python
# Chunking Configuration
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 150  # Overlap between chunks

# Retrieval Configuration
SCORE_THRESHOLD = 0.3  # Minimum relevance score
DEFAULT_TOP_K = 5  # Number of chunks to retrieve

# Embedding Model
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
```

### Tuning for Your Use Case

- **Higher CHUNK_SIZE**: Better context, fewer chunks
- **Lower SCORE_THRESHOLD**: More permissive, more results
- **Higher DEFAULT_TOP_K**: More context, higher costs

## 🧪 Testing

### Test Your Knowledge Base

Run the test script to verify everything works:

```bash
python test.py
```

This will test your knowledge base with sample queries and show:
- Response quality
- Citation accuracy
- Confidence scores
- Chunk usage

### API Testing

Test the API directly:

```bash
# Test knowledge bases
curl http://localhost:8000/knowledge-bases

# Test query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this document about?",
    "user_id": "your_username",
    "bot_id": "your_bot_name"
  }'
```

## 📊 Performance

### Expected Performance

- **Document Processing**: 2-5 seconds per document
- **Query Response**: 1-3 seconds
- **Memory Usage**: ~500MB for 1000 documents
- **Concurrent Users**: 10-100 users supported

### Optimization Tips

1. **Chunk Size**: 1000 characters is optimal for most content
2. **Score Threshold**: 0.3 works well for mixed content
3. **Batch Upload**: Upload multiple documents together
4. **Regular Cleanup**: Remove unused knowledge bases

## 🛠️ Troubleshooting

### Common Issues

**"No relevant context found"**
- Check if documents were uploaded successfully
- Try lowering the score threshold in `config.py`
- Verify username and bot name are correct

**"Connection error"**
- Ensure FastAPI backend is running on port 8000
- Check if Qdrant is running on port 6333
- Verify your `.env` file has the correct API key
- Make sure frontend is running on port 3000

**"Module not found"**
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt`
- Check Python version (3.8+ required)
- For frontend issues, run `cd Frontend && npm install`

**Slow document processing**
- The system uses fast fallback processing
- Large documents may take longer
- Check your internet connection for API calls

### Getting Help

1. **Check the logs** in your terminal
2. **Verify all services are running**:
   - FastAPI: http://localhost:8000
   - Frontend: http://localhost:3000
   - Qdrant: http://localhost:6333
3. **Test with the test script**: `python test.py`

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React/TS      │    │   FastAPI       │    │   Qdrant        │
│   Frontend      │◄──►│   Backend       │◄──►│   Vector DB     │
│   (Port 3000)   │    │   (Port 8000)   │    │   (Port 6333)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   OpenAI API    │
                       │   (Embeddings   │
                       │   + Chat)       │
                       └─────────────────┘
```

## 📁 Project Structure

```
fusor-ai/
├── 📄 main.py                 # FastAPI application
├── 📄 data_ingestion.py       # Document processing
├── 📄 search_engine.py        # RAG pipeline
├── 📄 gradio_interface.py     # Legacy Gradio interface
├── 📄 config.py              # Configuration
├── 📄 test.py                # Testing script
├── 📄 start_full_platform.py # Full platform startup script
├── 📄 requirements.txt       # Python dependencies
├── 📄 .env                   # Environment variables
├── 📁 Frontend/              # TypeScript/React frontend
│   ├── 📄 package.json       # Node.js dependencies
│   ├── 📄 vite.config.ts     # Vite configuration
│   ├── 📄 tailwind.config.js # Tailwind CSS config
│   ├── 📁 components/        # React components
│   ├── 📁 services/          # API services
│   └── 📁 styles/            # CSS styles
└── 📁 qdrant_storage/        # Vector database storage
```

## 🔮 Future Enhancements

- **Hybrid Search**: Vector + keyword search
- **Advanced Reranking**: Better result ordering
- **Real-time Collaboration**: Multiple users per bot
- **Analytics Dashboard**: Usage statistics
- **Custom Embeddings**: Support for other models
- **API Rate Limiting**: Production-ready scaling

## 📝 License

This project is part of a Final Year Project (FYP) for academic purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the logs
3. Test with the provided test script
4. Create an issue in the repository

---

**Built with ❤️ using FastAPI, Gradio, Qdrant, and OpenAI**

**Ready to create your first AI chatbot? Start with Step 1 above!** 🚀
