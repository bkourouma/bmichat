import React, { useState, useRef, useEffect } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  PaperAirplaneIcon,
  DocumentTextIcon,
  ClockIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline'
import toast from 'react-hot-toast'

import { chatApi } from '@/lib/api'
import { generateId, formatDuration } from '@/lib/utils'
import type { ChatMessage, ChatResponse } from '@/types'

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [currentMessage, setCurrentMessage] = useState('')
  const [sessionId] = useState(() => generateId('chat'))
  const [keywords, setKeywords] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Load chat history
  useQuery({
    queryKey: ['chat-history', sessionId],
    queryFn: () => chatApi.getHistory(sessionId),
    enabled: !!sessionId,
  })

  // Load session summary
  const { data: sessionSummary } = useQuery({
    queryKey: ['chat-session', sessionId],
    queryFn: () => chatApi.getSessionSummary(sessionId),
    enabled: !!sessionId,
    retry: false,
  })

  // Send message mutation
  const sendMessageMutation = useMutation({
    mutationFn: chatApi.sendMessage,
    onSuccess: (response: ChatResponse) => {
      // Add assistant response
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: response.message,
          timestamp: response.timestamp,
        }
      ])
      
      toast.success(`Réponse générée en ${formatDuration(response.response_time_ms)}`)
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Erreur lors de l\'envoi du message')
    },
  })

  const handleSendMessage = async () => {
    if (!currentMessage.trim() || sendMessageMutation.isPending) return

    const userMessage: ChatMessage = {
      role: 'user',
      content: currentMessage,
      timestamp: new Date().toISOString(),
    }

    // Add user message immediately
    setMessages(prev => [...prev, userMessage])
    
    // Clear input
    const messageToSend = currentMessage
    setCurrentMessage('')

    // Send to API
    try {
      await sendMessageMutation.mutateAsync({
        message: messageToSend,
        session_id: sessionId,
        keywords: keywords ? keywords.split(',').map(k => k.trim()) : undefined,
        use_history: true,
        max_context_chunks: 5,
      })
    } catch (error) {
      // Error is handled by onError callback
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const clearChat = async () => {
    try {
      await chatApi.clearSession(sessionId)
      setMessages([])
      toast.success('Conversation effacée')
    } catch (error) {
      toast.error('Erreur lors de l\'effacement de la conversation')
    }
  }

  return (
    <div className="flex h-full max-h-[calc(100vh-8rem)]">
      {/* Main chat area */}
      <div className="flex-1 flex flex-col">
        {/* Chat header */}
        <div className="bg-white border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-600 rounded-full flex items-center justify-center">
                <SparklesIcon className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-gray-900">
                  Akissi - Assistant BMI
                </h2>
                <p className="text-sm text-gray-500">
                  Posez vos questions sur les documents BMI
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              {sessionSummary && (
                <div className="text-sm text-gray-500">
                  {sessionSummary.message_count} messages • {sessionSummary.total_tokens_used} tokens
                </div>
              )}
              <button
                onClick={clearChat}
                className="btn-outline btn text-sm"
              >
                Nouvelle conversation
              </button>
            </div>
          </div>
        </div>

        {/* Messages area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 ? (
            <div className="text-center py-12">
              <SparklesIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                Commencez une conversation
              </h3>
              <p className="text-gray-500 max-w-md mx-auto">
                Posez une question sur vos documents BMI. Akissi utilisera les informations 
                disponibles pour vous donner une réponse précise et documentée.
              </p>
              
              {/* Suggested questions */}
              <div className="mt-8 space-y-2">
                <p className="text-sm font-medium text-gray-700">Questions suggérées :</p>
                <div className="space-y-2">
                  {[
                    "Comment faire une réclamation d'assurance ?",
                    "Quels sont les services BMI disponibles ?",
                    "Comment contacter le support client ?",
                  ].map((question) => (
                    <button
                      key={question}
                      onClick={() => setCurrentMessage(question)}
                      className="block w-full max-w-md mx-auto p-3 text-left bg-gray-50 hover:bg-gray-100 rounded-lg text-sm text-gray-700 transition-colors"
                    >
                      {question}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <>
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-3xl ${
                      message.role === 'user'
                        ? 'message-user'
                        : 'message-assistant'
                    }`}
                  >
                    <div className="prose-chat">
                      {message.content.split('\n').map((line, i) => (
                        <p key={i}>{line}</p>
                      ))}
                    </div>
                    
                    {message.timestamp && (
                      <div className="mt-2 text-xs opacity-70">
                        {new Date(message.timestamp).toLocaleTimeString('fr-FR')}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              
              {/* Typing indicator */}
              {sendMessageMutation.isPending && (
                <div className="flex justify-start">
                  <div className="message-assistant">
                    <div className="typing-indicator">
                      <div className="typing-dot"></div>
                      <div className="typing-dot"></div>
                      <div className="typing-dot"></div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
        <div className="bg-white border-t border-gray-200 p-6">
          {/* Keywords input */}
          <div className="mb-4">
            <input
              type="text"
              placeholder="Mots-clés optionnels (ex: assurance, réclamation, support)"
              value={keywords}
              onChange={(e) => setKeywords(e.target.value)}
              className="input w-full text-sm"
            />
          </div>
          
          {/* Message input */}
          <div className="flex space-x-4">
            <div className="flex-1">
              <textarea
                value={currentMessage}
                onChange={(e) => setCurrentMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Tapez votre message... (Entrée pour envoyer, Shift+Entrée pour nouvelle ligne)"
                className="input w-full resize-none"
                rows={3}
                disabled={sendMessageMutation.isPending}
              />
            </div>
            <button
              onClick={handleSendMessage}
              disabled={!currentMessage.trim() || sendMessageMutation.isPending}
              className="btn-primary px-6 py-3 self-end"
            >
              <PaperAirplaneIcon className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Sidebar */}
      <div className="w-80 bg-white border-l border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Informations de session
        </h3>
        
        {sessionSummary ? (
          <div className="space-y-4">
            <div className="card p-4">
              <div className="flex items-center space-x-2 mb-2">
                <ClockIcon className="w-4 h-4 text-gray-400" />
                <span className="text-sm font-medium text-gray-700">Durée</span>
              </div>
              <p className="text-lg font-semibold text-gray-900">
                {Math.round(sessionSummary.duration_minutes || 0)} min
              </p>
            </div>
            
            <div className="card p-4">
              <div className="flex items-center space-x-2 mb-2">
                <DocumentTextIcon className="w-4 h-4 text-gray-400" />
                <span className="text-sm font-medium text-gray-700">Messages</span>
              </div>
              <p className="text-lg font-semibold text-gray-900">
                {sessionSummary.message_count}
              </p>
            </div>
            
            <div className="card p-4">
              <div className="flex items-center space-x-2 mb-2">
                <SparklesIcon className="w-4 h-4 text-gray-400" />
                <span className="text-sm font-medium text-gray-700">Tokens utilisés</span>
              </div>
              <p className="text-lg font-semibold text-gray-900">
                {sessionSummary.total_tokens_used.toLocaleString('fr-FR')}
              </p>
            </div>
          </div>
        ) : (
          <div className="text-center py-8">
            <p className="text-gray-500 text-sm">
              Les statistiques apparaîtront après le premier message
            </p>
          </div>
        )}
        
        {/* Tips */}
        <div className="mt-8">
          <h4 className="text-sm font-medium text-gray-900 mb-3">
            Conseils d'utilisation
          </h4>
          <div className="space-y-3 text-sm text-gray-600">
            <div className="flex items-start space-x-2">
              <span className="text-primary-600">•</span>
              <span>Utilisez des mots-clés pour filtrer les documents pertinents</span>
            </div>
            <div className="flex items-start space-x-2">
              <span className="text-primary-600">•</span>
              <span>Posez des questions précises pour de meilleurs résultats</span>
            </div>
            <div className="flex items-start space-x-2">
              <span className="text-primary-600">•</span>
              <span>Akissi utilise uniquement les informations des documents BMI</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Chat
