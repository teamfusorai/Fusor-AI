import { useState, useEffect } from 'react';
import { Send, RefreshCw, ChevronDown, Database, BarChart3 } from 'lucide-react';
import { api, ChatMessage, KnowledgeBase, ChatWebSocket } from '../services/api';

export function ChatWithBot() {
  const [selectedKB, setSelectedKB] = useState('');
  const [username, setUsername] = useState('');
  const [botName, setBotName] = useState('');
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [currentKBStats, setCurrentKBStats] = useState<{ chunks_count: number } | null>(null);
  const [wsConnection, setWsConnection] = useState<ChatWebSocket | null>(null);

  // Load knowledge bases on component mount
  useEffect(() => {
    loadKnowledgeBases();
  }, []);

  // Update stats when username/botName changes
  useEffect(() => {
    if (username && botName) {
      loadKBStats(username, botName);
    } else {
      setCurrentKBStats(null);
    }
  }, [username, botName]);

  const loadKnowledgeBases = async () => {
    setIsRefreshing(true);
    try {
      const kbs = await api.getKnowledgeBases();
      setKnowledgeBases(kbs);
    } catch (error) {
      console.error('Failed to load knowledge bases:', error);
    } finally {
      setIsRefreshing(false);
    }
  };

  const loadKBStats = async (user: string, bot: string) => {
    try {
      const stats = await api.getKnowledgeBaseStats(user, bot);
      setCurrentKBStats(stats);
    } catch (error) {
      console.error('Failed to load KB stats:', error);
      setCurrentKBStats(null);
    }
  };

  const handleKBSelection = (kbSelection: string) => {
    setSelectedKB(kbSelection);
    if (kbSelection && kbSelection.includes(' - ')) {
      const [user, bot] = kbSelection.split(' - ');
      setUsername(user);
      setBotName(bot);
    }
  };

  const handleSend = async () => {
    if (!message.trim() || !username.trim() || !botName.trim()) {
      return;
    }

    const userMessage: ChatMessage = {
      role: 'user',
      content: message.trim()
    };

    setChatHistory(prev => [...prev, userMessage]);
      setMessage('');
    setIsLoading(true);

    try {
      const response = await api.queryChatbot(message.trim(), username, botName);
      
      const botMessage: ChatMessage = {
        role: 'assistant',
        content: response.answer,
        citations: response.citations,
        chunks_used: response.chunks_used
      };

      setChatHistory(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Failed to get response'}`
      };
      setChatHistory(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h2 style={{ color: '#ffffff', marginBottom: '24px', fontSize: '24px', fontWeight: 'bold' }}>
          Chat with Your AI Assistant
        </h2>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>
        {/* Left Column - Chat Interface */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Knowledge Base Selection and User Info */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '16px' }}>
            <div style={{ gridColumn: '1 / -1' }}>
              <label style={{ display: 'block', fontSize: '14px', color: '#d4d4d8', marginBottom: '8px' }}>
                Select Knowledge Base
              </label>
              <div style={{ position: 'relative' }}>
                <select
                  value={selectedKB}
                  onChange={(e) => handleKBSelection(e.target.value)}
                  style={{
                    width: '100%',
                    backgroundColor: 'rgba(24, 24, 27, 0.5)',
                    border: '1px solid #27272a',
                    borderRadius: '8px',
                    padding: '10px 16px',
                    color: '#ffffff',
                    fontSize: '14px',
                    appearance: 'none',
                    cursor: 'pointer'
                  }}
                >
                  <option value="">Select a knowledge base...</option>
                  {knowledgeBases.map((kb) => (
                    <option key={`${kb.user_id}-${kb.bot_id}`} value={`${kb.user_id} - ${kb.bot_id}`}>
                      {kb.user_id} - {kb.bot_id} ({kb.chunks_count} chunks)
                    </option>
                  ))}
                </select>
                <ChevronDown style={{
                  position: 'absolute',
                  right: '16px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  width: '16px',
                  height: '16px',
                  color: '#71717a',
                  pointerEvents: 'none'
                }} />
              </div>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '14px', color: '#d4d4d8', marginBottom: '8px' }}>
                Username
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter username"
                style={{
                  width: '100%',
                  backgroundColor: 'rgba(24, 24, 27, 0.5)',
                  border: '1px solid #27272a',
                  borderRadius: '8px',
                  padding: '10px 16px',
                  color: '#ffffff',
                  fontSize: '14px'
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '14px', color: '#d4d4d8', marginBottom: '8px' }}>
                Bot Name
              </label>
              <input
                type="text"
                value={botName}
                onChange={(e) => setBotName(e.target.value)}
                placeholder="Enter bot name"
                style={{
                  width: '100%',
                  backgroundColor: 'rgba(24, 24, 27, 0.5)',
                  border: '1px solid #27272a',
                  borderRadius: '8px',
                  padding: '10px 16px',
                  color: '#ffffff',
                  fontSize: '14px'
                }}
              />
            </div>

            <div style={{ display: 'flex', alignItems: 'end' }}>
              <button 
                onClick={loadKnowledgeBases}
                disabled={isRefreshing}
                style={{
                  width: '100%',
                  border: '1px solid #27272a',
                  padding: '10px 16px',
                  borderRadius: '8px',
                  backgroundColor: isRefreshing ? '#52525b' : 'rgba(24, 24, 27, 0.5)',
                  color: isRefreshing ? '#a1a1aa' : '#d4d4d8',
                  cursor: isRefreshing ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  fontSize: '14px',
                  transition: 'all 0.2s'
                }}
              >
                <RefreshCw style={{
                  width: '16px',
                  height: '16px',
                  animation: isRefreshing ? 'spin 1s linear infinite' : 'none'
                }} />
                {isRefreshing ? 'Refreshing...' : 'Refresh'}
              </button>
            </div>
          </div>

          {/* Chat History */}
          <div>
            <label style={{ display: 'block', fontSize: '14px', color: '#d4d4d8', marginBottom: '8px' }}>
              Chat History
            </label>
            <div style={{
              backgroundColor: 'rgba(24, 24, 27, 0.3)',
              border: '1px solid #27272a',
              borderRadius: '8px',
              padding: '24px',
              height: '400px',
              overflowY: 'auto'
            }}>
              {chatHistory.length === 0 ? (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                  <p style={{ color: '#71717a', fontSize: '14px' }}>Start a conversation...</p>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {chatHistory.map((msg, idx) => (
                    <div
                      key={idx}
                      style={{
                        display: 'flex',
                        justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start'
                      }}
                    >
                      <div
                        style={{
                          maxWidth: '80%',
                          padding: '10px 16px',
                          borderRadius: '8px',
                          backgroundColor: msg.role === 'user' ? '#4f46e5' : '#27272a',
                          color: msg.role === 'user' ? '#ffffff' : '#d4d4d8'
                        }}
                      >
                        <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                        {msg.citations && msg.citations.length > 0 && (
                          <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #3f3f46' }}>
                            <p style={{ fontSize: '12px', color: '#a1a1aa' }}>
                              Sources: {msg.citations.join(', ')}
                            </p>
                          </div>
                        )}
                        {msg.chunks_used && (
                          <div style={{ marginTop: '4px' }}>
                            <p style={{ fontSize: '12px', color: '#a1a1aa' }}>
                              Used {msg.chunks_used} chunks
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                  {isLoading && (
                    <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
                      <div style={{
                        backgroundColor: '#27272a',
                        color: '#d4d4d8',
                        padding: '10px 16px',
                        borderRadius: '8px'
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <div style={{
                            width: '8px',
                            height: '8px',
                            backgroundColor: '#a1a1aa',
                            borderRadius: '50%',
                            animation: 'bounce 1s infinite'
                          }} />
                          <div style={{
                            width: '8px',
                            height: '8px',
                            backgroundColor: '#a1a1aa',
                            borderRadius: '50%',
                            animation: 'bounce 1s infinite',
                            animationDelay: '0.1s'
                          }} />
                          <div style={{
                            width: '8px',
                            height: '8px',
                            backgroundColor: '#a1a1aa',
                            borderRadius: '50%',
                            animation: 'bounce 1s infinite',
                            animationDelay: '0.2s'
                          }} />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Message Input */}
          <div>
            <label style={{ display: 'block', fontSize: '14px', color: '#d4d4d8', marginBottom: '8px' }}>
              Your Message
            </label>
            <div style={{ display: 'flex', gap: '12px' }}>
              <input
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={isLoading}
                placeholder="Type your question here..."
                style={{
                  flex: 1,
                  backgroundColor: 'rgba(24, 24, 27, 0.5)',
                  border: '1px solid #27272a',
                  borderRadius: '8px',
                  padding: '10px 16px',
                  color: '#ffffff',
                  fontSize: '14px',
                  opacity: isLoading ? 0.5 : 1,
                  cursor: isLoading ? 'not-allowed' : 'text'
                }}
              />
              <button
                onClick={handleSend}
                disabled={isLoading || !message.trim() || !username.trim() || !botName.trim()}
                style={{
                  padding: '10px 24px',
                  borderRadius: '8px',
                  backgroundColor: isLoading || !message.trim() || !username.trim() || !botName.trim() ? '#52525b' : '#4f46e5',
                  color: isLoading || !message.trim() || !username.trim() || !botName.trim() ? '#a1a1aa' : '#ffffff',
                  border: 'none',
                  cursor: isLoading || !message.trim() || !username.trim() || !botName.trim() ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontSize: '14px',
                  transition: 'all 0.2s'
                }}
              >
                <Send style={{ width: '16px', height: '16px' }} />
                {isLoading ? 'Sending...' : 'Send'}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column - Knowledge Base Info */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Current Knowledge Base */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <Database style={{ width: '16px', height: '16px', color: '#10b981' }} />
              <label style={{ display: 'block', fontSize: '14px', color: '#d4d4d8' }}>
                Current Knowledge Base
              </label>
            </div>
            <div style={{
              backgroundColor: 'rgba(24, 24, 27, 0.3)',
              border: '1px solid #27272a',
              borderRadius: '8px',
              padding: '16px'
            }}>
              <div style={{ fontSize: '14px', color: '#818cf8', marginBottom: '8px' }}>Active Knowledge Base</div>
              {selectedKB ? (
                <div>
                  <p style={{ color: '#d4d4d8', fontSize: '14px' }}>{selectedKB}</p>
                  {currentKBStats && (
                    <p style={{ color: '#71717a', fontSize: '12px', marginTop: '4px' }}>
                      {currentKBStats.chunks_count} chunks available
                    </p>
                  )}
                </div>
              ) : (
                <p style={{ color: '#71717a', fontSize: '14px' }}>No knowledge base selected</p>
              )}
            </div>
          </div>

          {/* Knowledge Base Stats */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <BarChart3 style={{ width: '16px', height: '16px', color: '#f59e0b' }} />
              <label style={{ display: 'block', fontSize: '14px', color: '#d4d4d8' }}>
                Knowledge Base Stats
              </label>
            </div>
            <div style={{
              backgroundColor: 'rgba(24, 24, 27, 0.3)',
              border: '1px solid #27272a',
              borderRadius: '8px',
              padding: '16px'
            }}>
              <div style={{ fontSize: '14px', color: '#818cf8', marginBottom: '12px' }}>Statistics</div>
              {currentKBStats ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <p style={{ color: '#d4d4d8', fontSize: '14px' }}>
                    Chunks: {currentKBStats.chunks_count}
                  </p>
                  <p style={{ color: '#d4d4d8', fontSize: '14px' }}>
                    User: {username}
                  </p>
                  <p style={{ color: '#d4d4d8', fontSize: '14px' }}>
                    Bot: {botName}
                  </p>
                </div>
              ) : (
                <p style={{ color: '#71717a', fontSize: '14px' }}>No data available</p>
              )}
            </div>
          </div>

          {/* Available Knowledge Bases */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <Database style={{ width: '16px', height: '16px', color: '#3b82f6' }} />
              <label style={{ display: 'block', fontSize: '14px', color: '#d4d4d8' }}>
                Available Knowledge Bases
              </label>
            </div>
            <div style={{
              backgroundColor: 'rgba(24, 24, 27, 0.3)',
              border: '1px solid #27272a',
              borderRadius: '8px',
              padding: '16px'
            }}>
              <div style={{ fontSize: '14px', color: '#818cf8', marginBottom: '12px' }}>All Knowledge Bases</div>
              {knowledgeBases.length > 0 ? (
                <ul style={{ display: 'flex', flexDirection: 'column', gap: '8px', margin: 0, padding: 0 }}>
                  {knowledgeBases.map((kb) => (
                    <li key={`${kb.user_id}-${kb.bot_id}`} style={{
                      color: '#a1a1aa',
                      fontSize: '14px',
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '8px',
                      listStyle: 'none'
                    }}>
                      <span style={{ color: '#71717a', marginTop: '4px' }}>•</span>
                      <span>{kb.user_id} - {kb.bot_id} ({kb.chunks_count} chunks)</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p style={{ color: '#71717a', fontSize: '14px' }}>No knowledge bases available</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
