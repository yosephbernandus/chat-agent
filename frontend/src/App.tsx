import { useState } from 'react'
import './App.css'

interface Source {
  content: string
  source: string
  score: number
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [mode, setMode] = useState<'direct' | 'rag'>('rag')
  const [connection, setConnection] = useState<'http' | 'sse' | 'websocket'>('http')
  const [isLoading, setIsLoading] = useState(false)
  const [docCount, setDocCount] = useState(0)

  return (
    <div className="flex flex-col h-screen bg-[#2f2f2f] text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-700 bg-[#2f2f2f] px-4 py-3">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <h1 className="text-sm font-medium text-gray-300">RAG Chat Agent</h1>
          <div className="flex gap-4 text-xs">
            {/* Mode Toggle */}
            <div className="flex items-center gap-2">
              <span className="text-gray-400">Mode:</span>
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value as 'direct' | 'rag')}
                className="bg-[#3f3f3f] border border-gray-600 text-gray-200 rounded px-2 py-1 text-xs focus:outline-none focus:border-gray-500"
              >
                <option value="direct">Direct</option>
                <option value="rag">RAG</option>
              </select>
            </div>

            {/* Connection Toggle */}
            <div className="flex items-center gap-2">
              <span className="text-gray-400">Connection:</span>
              <select
                value={connection}
                onChange={(e) => setConnection(e.target.value as 'http' | 'sse' | 'websocket')}
                className="bg-[#3f3f3f] border border-gray-600 text-gray-200 rounded px-2 py-1 text-xs focus:outline-none focus:border-gray-500"
              >
                <option value="http">HTTP</option>
                <option value="sse">SSE</option>
                <option value="websocket">WebSocket</option>
              </select>
            </div>

            <div className="text-gray-500">KB: {docCount} docs</div>
          </div>
        </div>
      </header>

      {/* Messages Area */}
      <main className="flex-1 overflow-y-auto px-4">
        <div className="max-w-3xl mx-auto py-8">
          {messages.length === 0 ? (
            <div className="text-center text-gray-500 mt-32">
              <p className="text-lg mb-2">Start a conversation</p>
              <p className="text-sm">Your AI assistant is ready to help</p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className="mb-8">
                {/* User message - right aligned, more compact */}
                {msg.role === 'user' ? (
                  <div className="flex justify-end mb-6">
                    <div className="bg-[#3f3f3f] rounded-lg px-4 py-3 max-w-2xl">
                      <p className="text-gray-100 leading-relaxed">{msg.content}</p>
                    </div>
                  </div>
                ) : (
                  // Assistant message - left aligned, full width
                  <div className="mb-6">
                    {/* Sources */}
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="mb-3">
                        <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-[#3f3f3f] border border-gray-700 rounded text-xs text-gray-400">
                          <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z"/>
                          </svg>
                          Sources: {msg.sources.map(s => s.source).join(', ')}
                        </div>
                      </div>
                    )}
                    <div className="text-gray-200 leading-relaxed whitespace-pre-wrap font-serif">
                      {msg.content}
                    </div>
                  </div>
                )}
              </div>
            ))
          )}

          {/* Loading State */}
          {isLoading && (
            <div className="mb-6">
              <div className="flex gap-2 text-gray-500 text-sm">
                <div className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                <div className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                <div className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Input Area */}
      <footer className="border-t border-gray-700 bg-[#2f2f2f] px-4 py-4">
        <div className="max-w-3xl mx-auto">
          <div className="relative">
            <div className="flex items-end gap-3 bg-[#3f3f3f] rounded-2xl border border-gray-700 px-4 py-3 focus-within:border-gray-600">
              {/* Plus button */}
              <button className="text-gray-400 hover:text-gray-300 mb-1">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
              </button>

              {/* Input */}
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !isLoading && input.trim()) {
                    console.log('Send:', input)
                  }
                }}
                placeholder="Reply..."
                disabled={isLoading}
                className="flex-1 bg-transparent text-gray-100 placeholder-gray-500 focus:outline-none disabled:text-gray-600"
              />

              {/* Send button */}
              <button
                onClick={() => {
                  console.log('Send:', input)
                }}
                disabled={isLoading || !input.trim()}
                className="bg-[#9b6b4f] hover:bg-[#b07d5f] disabled:bg-gray-700 disabled:cursor-not-allowed text-white rounded-lg px-3 py-1.5 transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                </svg>
              </button>
            </div>

            {/* Model selector (like Opus 4.5 in Claude) */}
            <div className="flex items-center justify-between mt-3 text-xs">
              <div className="text-gray-500">
                Assistant is AI and can make mistakes. Please double-check responses.
              </div>
              <select className="bg-[#3f3f3f] border border-gray-700 text-gray-400 rounded px-2 py-1 text-xs focus:outline-none focus:border-gray-600">
                <option>Ollama 3b</option>
              </select>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
