# 🚀 Multi-Tenant Chatbot Platform - Full Stack Integration

This project now includes both Python FastAPI backend and TypeScript React frontend, providing a complete full-stack solution for document ingestion and AI chatbot interactions.

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

## ✨ Features

### Frontend (TypeScript/React)
- **Modern UI**: Dark theme with Tailwind CSS matching your design requirements
- **Document Upload**: Drag-and-drop file upload with progress indicators
- **URL Processing**: Support for single pages and sitemap.xml processing
- **Real-time Chat**: Interactive chat interface with message history
- **Knowledge Base Management**: View and manage all knowledge bases
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Type Safety**: Full TypeScript support with proper type definitions

### Backend (Python/FastAPI)
- **REST API**: Complete RESTful API for all frontend operations
- **WebSocket Support**: Real-time chat capabilities
- **CORS Enabled**: Properly configured for frontend integration
- **Document Processing**: Support for PDF, DOCX, TXT, MD files
- **URL Processing**: Single page and sitemap processing
- **Multi-tenant**: User and bot isolation
- **Vector Search**: Advanced RAG pipeline with Qdrant

## 🚀 Quick Start

### Option 1: Start Everything at Once (Recommended)

```bash
# Make sure you're in the project root
python start_full_platform.py
```

This will:
1. ✅ Check virtual environment
2. 📦 Install frontend dependencies (if needed)
3. 🐍 Start Python FastAPI backend
4. ⚛️ Start TypeScript React frontend
5. 🌐 Open both services automatically

### Option 2: Start Services Separately

**Terminal 1 - Python Backend:**
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux

# Start FastAPI backend
python main.py
```

**Terminal 2 - TypeScript Frontend:**
```bash
# Navigate to frontend directory
cd Frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

## 🌐 Access Points

- **🎨 Frontend**: http://localhost:3000
- **🔧 Backend API**: http://localhost:8000
- **📚 API Documentation**: http://localhost:8000/docs
- **🗄️ Qdrant Dashboard**: http://localhost:6333/dashboard

## 📁 Project Structure

```
fusor-ai/
├── 📄 main.py                     # FastAPI application
├── 📄 data_ingestion.py           # Document processing
├── 📄 search_engine.py            # RAG pipeline
├── 📄 gradio_interface.py         # Gradio interface (legacy)
├── 📄 start_full_platform.py      # Full-stack startup script
├── 📄 config.py                  # Configuration
├── 📄 requirements.txt           # Python dependencies
├── 📁 Frontend/                  # TypeScript React frontend
│   ├── 📄 package.json           # Node.js dependencies
│   ├── 📄 tsconfig.json          # TypeScript configuration
│   ├── 📄 vite.config.ts         # Vite configuration
│   ├── 📄 tailwind.config.js     # Tailwind CSS configuration
│   ├── 📄 App.tsx                # Main React component
│   ├── 📁 components/            # React components
│   │   ├── 📄 UploadDocuments.tsx # Document upload component
│   │   └── 📄 ChatWithBot.tsx    # Chat interface component
│   ├── 📁 services/              # API service layer
│   │   └── 📄 api.ts             # API client and types
│   └── 📁 styles/                # CSS styles
│       └── 📄 globals.css         # Global styles
└── 📁 qdrant_storage/            # Vector database storage
```

## 🔧 API Endpoints

### Document Management
- `POST /ingest` - Upload documents or process URLs
- `GET /knowledge-bases` - List all knowledge bases
- `GET /knowledge-bases/{user_id}/{bot_id}/stats` - Get KB statistics
- `DELETE /knowledge-bases/{user_id}/{bot_id}` - Delete knowledge base

### Chat & Query
- `POST /query` - Query chatbot
- `WebSocket /ws/chat/{user_id}/{bot_id}` - Real-time chat

## 🎨 Frontend Features

### Upload Documents Tab
- **File Upload**: Drag-and-drop with file validation
- **URL Processing**: Single page or sitemap processing
- **Progress Tracking**: Real-time upload status
- **Error Handling**: Comprehensive error messages

### Chat with Bot Tab
- **Knowledge Base Selection**: Dropdown with all available KBs
- **Real-time Chat**: Message history with citations
- **Statistics**: Live KB statistics in sidebar
- **Responsive Design**: Works on all screen sizes

## 🛠️ Development

### Frontend Development
```bash
cd Frontend
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
npm run type-check   # TypeScript type checking
npm run lint         # ESLint checking
```

### Backend Development
```bash
# Activate virtual environment
venv\Scripts\activate

# Run with auto-reload
python main.py

# Or use uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🔒 Environment Variables

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## 📦 Dependencies

### Python Backend
- FastAPI - Web framework
- Qdrant - Vector database
- LangChain - AI/ML framework
- OpenAI - AI models
- Uvicorn - ASGI server

### TypeScript Frontend
- React 18 - UI framework
- TypeScript - Type safety
- Vite - Build tool
- Tailwind CSS - Styling
- Lucide React - Icons

## 🚀 Production Deployment

### Frontend Build
```bash
cd Frontend
npm run build
# Output in Frontend/dist/
```

### Backend Deployment
```bash
# Install production dependencies
pip install -r requirements.txt

# Run with production server
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 🔄 Migration from Gradio

The TypeScript frontend provides the same functionality as the Gradio interface but with:
- ✅ Better performance
- ✅ Modern UI/UX
- ✅ Mobile responsiveness
- ✅ Real-time features
- ✅ Type safety
- ✅ Better error handling

You can still use the Gradio interface by running:
```bash
python gradio_interface.py
```

## 🐛 Troubleshooting

### Common Issues

**Frontend won't start:**
```bash
cd Frontend
npm install
npm run dev
```

**Backend connection errors:**
- Check if FastAPI is running on port 8000
- Verify CORS settings in main.py
- Check Qdrant is running on port 6333

**File upload issues:**
- Check file size limits
- Verify file types are supported
- Check backend logs for errors

**Chat not working:**
- Verify knowledge base exists
- Check username/bot name are correct
- Verify OpenAI API key is set

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the API documentation at http://localhost:8000/docs
3. Check the browser console for frontend errors
4. Check the terminal for backend errors

---

**Built with ❤️ using FastAPI, React, TypeScript, and Tailwind CSS**
