import { useState } from 'react';
import { UploadDocuments } from './components/UploadDocuments';
import { ChatWithBot } from './components/ChatWithBot';

export default function App() {
  const [activeTab, setActiveTab] = useState<'upload' | 'chat'>('upload');

  return (
    <div className="min-h-screen bg-[#0a0a0b] text-white">
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-white mb-2">Multi-Tenant Chatbot Platform</h1>
          <p className="text-zinc-400 text-sm">
            Upload documents to create knowledge bases and chat with your personalized AI assistants.
          </p>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-8 border-b border-zinc-800">
          <button
            onClick={() => setActiveTab('upload')}
            className={`px-4 py-3 text-sm transition-colors relative ${
              activeTab === 'upload'
                ? 'text-white'
                : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            Upload Documents
            {activeTab === 'upload' && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-indigo-500" />
            )}
          </button>
          <button
            onClick={() => setActiveTab('chat')}
            className={`px-4 py-3 text-sm transition-colors relative ${
              activeTab === 'chat'
                ? 'text-white'
                : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            Chat with Bot
            {activeTab === 'chat' && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-indigo-500" />
            )}
          </button>
        </div>

        {/* Content */}
        {activeTab === 'upload' ? <UploadDocuments /> : <ChatWithBot />}
      </div>

      {/* Footer */}
      <div className="border-t border-zinc-800 mt-16">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <p className="text-zinc-500 text-sm">Built with Gradio and FastAPI</p>
        </div>
      </div>
    </div>
  );
}
