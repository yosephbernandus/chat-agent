import { useCallback, useEffect, useState, useRef } from 'react'
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

const API_BASE = 'http://localhost:8000'
const WS_BASE = 'ws://localhost:8000'

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [mode, setMode] = useState<'direct' | 'rag'>('rag')
  const [connection, setConnection] = useState<'http' | 'sse' | 'websocket'>('http')
  const [isLoading, setIsLoading] = useState(false)
  const [docCount, setDocCount] = useState(0)
  const [isWsConnected, setIsWsConnected] = useState(false)

  // useRef for WebSocket - persists across renders, doesn't trigger re-render
  const wsRef = useRef<WebSocket | null>(null)
  const currentMessageRef = useRef<Message | null>(null)
  const messageAddedRef = useRef(false)

  // Fetch document count on mount
  useEffect(() => {
    fetchDocCount()
  }, [])

  // WebSocket lifecycle management
  useEffect(() => {
    if (connection !== 'websocket') {
      // Close WebSocket if switching away
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
        setIsWsConnected(false)
      }
      // Reset message tracking refs
      currentMessageRef.current = null
      messageAddedRef.current = false
      return
    }

    // Connect to WebSocket
    const ws = new WebSocket(`${WS_BASE}/ws/chat`)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('✓ WebSocket connected')
      setIsWsConnected(true)
    }

    ws.onclose = () => {
      console.log('✗ WebSocket disconnected')
      setIsWsConnected(false)
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)

      if (data.type === 'sources') {
        // Start new message with sources (RAG mode)
        currentMessageRef.current = {
          role: 'assistant',
          content: '',
          sources: data.data
        }

        // Add initial empty message to the array
        setMessages(prev => [...prev, { ...currentMessageRef.current! }])
        messageAddedRef.current = true
      } else if (data.type === 'token') {
        // Create message if not exists (direct mode - no sources sent)
        if (!currentMessageRef.current) {
          currentMessageRef.current = {
            role: 'assistant',
            content: ''
          }
        }

        // Append token
        currentMessageRef.current.content += data.data

        // Add or update message in the array
        setMessages(prev => {
          const newMessages = [...prev]
          if (!messageAddedRef.current) {
            // First token - add new message
            newMessages.push({ ...currentMessageRef.current! })
            messageAddedRef.current = true
          } else {
            // Update last message
            newMessages[newMessages.length - 1] = { ...currentMessageRef.current! }
          }
          return newMessages
        })
      } else if (data.type === 'done') {
        // Message complete - reset for next message
        currentMessageRef.current = null
        messageAddedRef.current = false
        setIsLoading(false)
      }
    }

    // Cleanup on unmount or connection change
    return () => {
      ws.close()
      currentMessageRef.current = null
      messageAddedRef.current = false
    }
  }, [connection])

  const fetchDocCount = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/documents`)
      const data = await res.json()
      setDocCount(data.count)
    } catch (error) {
      console.error('Failed to fetch doc count:', error)
    }
  }, [])

  // HTTP Chat
  const sendHTTP = useCallback(async (message: string) => {
    const endpoint = mode === 'rag' ? '/chat/rag' : '/chat'

    try {
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      })

      const data = await res.json()

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        sources: data.sources
      }])
    } catch (error) {
      console.error('HTTP error:', error)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Error: Failed to get response'
      }])
    }
  }, [mode])

  // SSE Streaming Chat
  const sendSSE = useCallback(async (message: string) => {
    const endpoint = mode === 'rag' ? '/chat/rag/stream' : '/chat'

    try {
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      })

      if (!res.body) throw new Error('No response body')

      const reader = res.body.getReader()
      const decoder = new TextDecoder()

      let currentMessage: Message = { role: 'assistant', content: '' }
      let messageIndex = messages.length + 1

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const text = decoder.decode(value)
        const lines = text.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.slice(6)
            try {
              const data = JSON.parse(jsonStr)

              if (data.type === 'sources') {
                currentMessage.sources = data.content
              } else if (data.type === 'token') {
                currentMessage.content += data.content

                setMessages(prev => {
                  const newMessages = [...prev]
                  if (newMessages[messageIndex]) {
                    newMessages[messageIndex] = { ...currentMessage }
                  } else {
                    newMessages.push({ ...currentMessage })
                  }
                  return newMessages
                })
              } else if (data.type === 'done') {
                // Streaming complete
              }
            } catch (e) {
              // Ignore JSON parse errors for incomplete chunks
            }
          }
        }
      }
    } catch (error) {
      console.error('SSE error:', error)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Error: Failed to stream response'
      }])
    }
  }, [mode, messages.length])

  const sendMessage = useCallback(async () => {
    if (!input.trim() || isLoading) return

    const userMessage = input.trim()
    setInput('')
    setIsLoading(true)

    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])

    try {
      if (connection === 'http') {
        await sendHTTP(userMessage)
      } else if (connection === 'sse') {
        await sendSSE(userMessage)
      } else if (connection === 'websocket') {
        // WebSocket
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({
            message: userMessage,
            mode: mode
          }))
          // Don't set loading false here - wait for 'done' message
        } else {
          setMessages(prev => [...prev, {
            role: 'assistant',
            content: 'Error: WebSocket not connected'
          }])
          setIsLoading(false)
        }
      }
    } catch (error) {
      console.error('Send error:', error)
      setIsLoading(false)
    } finally {
      // Only set loading false for HTTP/SSE (WebSocket handles it in onmessage)
      if (connection !== 'websocket') {
        setIsLoading(false)
      }
    }
  }, [input, isLoading, connection, mode, sendHTTP, sendSSE])

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
            {connection === 'websocket' && (
              <div className="flex items-center gap-1 text-xs">
                <div className={`w-2 h-2 rounded-full ${isWsConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
                <span className={isWsConnected ? 'text-green-400' : 'text-red-400'}>
                  {isWsConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            )}
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
                            <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z" />
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
                    sendMessage()
                  }
                }}
                placeholder="Reply..."
                disabled={isLoading}
                className="flex-1 bg-transparent text-gray-100 placeholder-gray-500 focus:outline-none disabled:text-gray-600"
              />

              {/* Send button */}
              <button
                onClick={sendMessage}
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
