import { useState } from 'react';
import { UploadDocuments } from './components/UploadDocuments';
import { ChatWithBot } from './components/ChatWithBot';

export default function App() {
  const [activeTab, setActiveTab] = useState<'upload' | 'chat'>('upload');

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#0a0a0b', color: '#ffffff' }}>
      <div style={{ maxWidth: '1280px', margin: '0 auto', padding: '32px 24px' }}>
        {/* Header */}
        <div style={{ marginBottom: '32px' }}>
          <h1 style={{ color: '#ffffff', marginBottom: '8px', fontSize: '30px', fontWeight: 'bold' }}>
            Multi-Tenant Chatbot Platform
          </h1>
          <p style={{ color: '#a1a1aa', fontSize: '14px' }}>
            Upload documents to create knowledge bases and chat with your personalized AI assistants.
          </p>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', gap: '8px', marginBottom: '32px', borderBottom: '1px solid #27272a' }}>
          <button
            onClick={() => setActiveTab('upload')}
            style={{
              padding: '12px 16px',
              fontSize: '14px',
              backgroundColor: activeTab === 'upload' ? 'transparent' : 'transparent',
              color: activeTab === 'upload' ? '#ffffff' : '#71717a',
              border: 'none',
              cursor: 'pointer',
              position: 'relative',
              transition: 'color 0.2s'
            }}
          >
            Upload Documents
            {activeTab === 'upload' && (
              <div style={{
                position: 'absolute',
                bottom: 0,
                left: 0,
                right: 0,
                height: '2px',
                backgroundColor: '#4f46e5'
              }} />
            )}
          </button>
          <button
            onClick={() => setActiveTab('chat')}
            style={{
              padding: '12px 16px',
              fontSize: '14px',
              backgroundColor: 'transparent',
              color: activeTab === 'chat' ? '#ffffff' : '#71717a',
              border: 'none',
              cursor: 'pointer',
              position: 'relative',
              transition: 'color 0.2s'
            }}
          >
            Chat with Bot
            {activeTab === 'chat' && (
              <div style={{
                position: 'absolute',
                bottom: 0,
                left: 0,
                right: 0,
                height: '2px',
                backgroundColor: '#4f46e5'
              }} />
            )}
          </button>
        </div>

        {/* Content */}
        {activeTab === 'upload' ? <UploadDocuments /> : <ChatWithBot />}
      </div>

      {/* Footer */}
      <div style={{ borderTop: '1px solid #27272a', marginTop: '64px' }}>
        <div style={{ maxWidth: '1280px', margin: '0 auto', padding: '16px 24px' }}>
          <p style={{ color: '#71717a', fontSize: '14px' }}>Built with React, TypeScript, and FastAPI</p>
        </div>
      </div>
    </div>
  );
}
