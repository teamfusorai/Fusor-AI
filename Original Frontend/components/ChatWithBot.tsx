import { useState } from 'react';
import { Send, RefreshCw, ChevronDown, Database, BarChart3 } from 'lucide-react';

export function ChatWithBot() {
  const [selectedKB, setSelectedKB] = useState('');
  const [username, setUsername] = useState('');
  const [botName, setBotName] = useState('');
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState<Array<{ role: string; content: string }>>([]);

  const knowledgeBases = [
    'anas183 - bot1',
    'anas183 - bot2',
  ];

  const handleSend = () => {
    if (message.trim()) {
      setChatHistory([...chatHistory, { role: 'user', content: message }]);
      setMessage('');
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-white mb-6">Chat with Your AI Assistant</h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Chat Interface */}
        <div className="lg:col-span-2 space-y-6">
          {/* Knowledge Base Selection and User Info */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="md:col-span-3">
              <label className="block text-sm text-zinc-300 mb-2">
                Select Knowledge Base
              </label>
              <div className="relative">
                <select
                  value={selectedKB}
                  onChange={(e) => setSelectedKB(e.target.value)}
                  className="w-full bg-zinc-900/50 border border-zinc-800 rounded-lg px-4 py-2.5 text-white appearance-none focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-transparent transition-all cursor-pointer"
                >
                  <option value="">Select a knowledge base...</option>
                  {knowledgeBases.map((kb) => (
                    <option key={kb} value={kb}>
                      {kb}
                    </option>
                  ))}
                </select>
                <ChevronDown className="absolute right-4 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500 pointer-events-none" />
              </div>
            </div>

            <div>
              <label className="block text-sm text-zinc-300 mb-2">
                Username
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter username"
                className="w-full bg-zinc-900/50 border border-zinc-800 rounded-lg px-4 py-2.5 text-white placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-transparent transition-all"
              />
            </div>

            <div>
              <label className="block text-sm text-zinc-300 mb-2">
                Bot Name
              </label>
              <input
                type="text"
                value={botName}
                onChange={(e) => setBotName(e.target.value)}
                placeholder="Enter bot name"
                className="w-full bg-zinc-900/50 border border-zinc-800 rounded-lg px-4 py-2.5 text-white placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-transparent transition-all"
              />
            </div>

            <div className="flex items-end">
              <button className="w-full bg-zinc-900/50 border border-zinc-800 hover:border-zinc-700 text-zinc-300 py-2.5 rounded-lg transition-all flex items-center justify-center gap-2">
                <RefreshCw className="w-4 h-4" />
                Refresh
              </button>
            </div>
          </div>

          {/* Chat History */}
          <div>
            <label className="block text-sm text-zinc-300 mb-2">
              Chat History
            </label>
            <div className="bg-zinc-900/30 border border-zinc-800 rounded-lg p-6 h-[400px] overflow-y-auto">
              {chatHistory.length === 0 ? (
                <div className="flex items-center justify-center h-full">
                  <p className="text-zinc-600 text-sm">Start a conversation...</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {chatHistory.map((msg, idx) => (
                    <div
                      key={idx}
                      className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`max-w-[80%] px-4 py-2.5 rounded-lg ${
                          msg.role === 'user'
                            ? 'bg-indigo-600 text-white'
                            : 'bg-zinc-800 text-zinc-100'
                        }`}
                      >
                        {msg.content}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Message Input */}
          <div>
            <label className="block text-sm text-zinc-300 mb-2">
              Your Message
            </label>
            <div className="flex gap-3">
              <input
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                placeholder="Type your question here..."
                className="flex-1 bg-zinc-900/50 border border-zinc-800 rounded-lg px-4 py-2.5 text-white placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-transparent transition-all"
              />
              <button
                onClick={handleSend}
                className="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2.5 rounded-lg transition-colors flex items-center gap-2"
              >
                <Send className="w-4 h-4" />
                Send
              </button>
            </div>
          </div>
        </div>

        {/* Right Column - Knowledge Base Info */}
        <div className="space-y-6">
          {/* Current Knowledge Base */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Database className="w-4 h-4 text-emerald-500" />
              <label className="block text-sm text-zinc-300">
                Current Knowledge Base
              </label>
            </div>
            <div className="bg-zinc-900/30 border border-zinc-800 rounded-lg p-4">
              <div className="text-sm text-indigo-400 mb-2">Active Knowledge Base</div>
              {selectedKB ? (
                <p className="text-zinc-300 text-sm">{selectedKB}</p>
              ) : (
                <p className="text-zinc-600 text-sm">No knowledge base selected</p>
              )}
            </div>
          </div>

          {/* Knowledge Base Stats */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-4 h-4 text-amber-500" />
              <label className="block text-sm text-zinc-300">
                Knowledge Base Stats
              </label>
            </div>
            <div className="bg-zinc-900/30 border border-zinc-800 rounded-lg p-4">
              <div className="text-sm text-indigo-400 mb-3">Statistics</div>
              <p className="text-zinc-600 text-sm">No data available</p>
            </div>
          </div>

          {/* Available Knowledge Bases */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Database className="w-4 h-4 text-blue-500" />
              <label className="block text-sm text-zinc-300">
                Available Knowledge Bases
              </label>
            </div>
            <div className="bg-zinc-900/30 border border-zinc-800 rounded-lg p-4">
              <div className="text-sm text-indigo-400 mb-3">All Knowledge Bases</div>
              {knowledgeBases.length > 0 ? (
                <ul className="space-y-2">
                  {knowledgeBases.map((kb) => (
                    <li key={kb} className="text-zinc-400 text-sm flex items-start gap-2">
                      <span className="text-zinc-600 mt-1">•</span>
                      <span>{kb}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-zinc-600 text-sm">No knowledge bases available</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
