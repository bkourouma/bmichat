import axios, { AxiosResponse, AxiosError } from 'axios'
import toast from 'react-hot-toast'

import type {
  ChatRequest,
  ChatResponse,
  Document,
  DocumentListResponse,
  SearchRequest,
  SearchResponse,
  AnalyticsData,
  HealthCheck,
  ChatSession,
  WidgetChatRequest,
  WidgetChatResponse,
  WidgetConfig,
} from '@/types'

// Create axios instance with default config
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api',
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    
    // Log request in development
    if (import.meta.env.DEV) {
      console.log(`🚀 ${config.method?.toUpperCase()} ${config.url}`, config.data)
    }
    
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response: AxiosResponse) => {
    // Log response in development
    if (import.meta.env.DEV) {
      console.log(`✅ ${response.config.method?.toUpperCase()} ${response.config.url}`, response.data)
    }
    
    return response
  },
  (error: AxiosError) => {
    // Log error in development
    if (import.meta.env.DEV) {
      console.error(`❌ ${error.config?.method?.toUpperCase()} ${error.config?.url}`, error.response?.data)
    }
    
    // Handle common errors
    if (error.response?.status === 401) {
      // Unauthorized - redirect to login or clear auth
      localStorage.removeItem('auth_token')
      // Could redirect to login page here
    } else if (error.response?.status && error.response.status >= 500) {
      // Server error
      toast.error('Erreur serveur. Veuillez réessayer plus tard.')
    } else if (error.code === 'ECONNABORTED') {
      // Timeout
      toast.error('Délai d\'attente dépassé. Veuillez réessayer.')
    } else if (!error.response) {
      // Network error
      toast.error('Erreur de connexion. Vérifiez votre connexion internet.')
    }
    
    return Promise.reject(error)
  }
)

// Chat API
export const chatApi = {
  sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    const response = await api.post<ChatResponse>('/chat', request)
    return response.data
  },

  getHistory: async (sessionId: string, limit = 50): Promise<any[]> => {
    const response = await api.get(`/chat/sessions/${sessionId}/history`, {
      params: { limit }
    })
    return response.data
  },

  getSessionSummary: async (sessionId: string): Promise<ChatSession> => {
    const response = await api.get<ChatSession>(`/chat/sessions/${sessionId}/summary`)
    return response.data
  },

  clearSession: async (sessionId: string): Promise<void> => {
    await api.delete(`/chat/sessions/${sessionId}`)
  },
}

// Documents API
export const documentsApi = {
  upload: async (file: File, keywords?: string): Promise<Document> => {
    const formData = new FormData()
    formData.append('file', file)
    if (keywords) {
      formData.append('keywords', keywords)
    }

    const response = await api.post<Document>('/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000, // 1 minute for file uploads
    })
    return response.data
  },

  list: async (skip = 0, limit = 100): Promise<DocumentListResponse> => {
    const response = await api.get<DocumentListResponse>('/documents', {
      params: { skip, limit }
    })
    return response.data
  },

  get: async (documentId: string): Promise<Document> => {
    const response = await api.get<Document>(`/documents/${documentId}`)
    return response.data
  },

  delete: async (documentId: string): Promise<void> => {
    await api.delete(`/documents/${documentId}`)
  },
}

// Search API
export const searchApi = {
  semantic: async (request: SearchRequest): Promise<SearchResponse> => {
    const response = await api.post<SearchResponse>('/search/semantic', request)
    return response.data
  },

  keywords: async (keywords: string[], k = 10, chunkType?: string): Promise<SearchResponse> => {
    const response = await api.post<SearchResponse>('/search/keywords', null, {
      params: { keywords: keywords.join(','), k, chunk_type: chunkType }
    })
    return response.data
  },

  hybrid: async (request: {
    query: string
    keywords?: string[]
    k?: number
    semantic_weight?: number
    keyword_weight?: number
  }): Promise<SearchResponse> => {
    const response = await api.post<SearchResponse>('/search/hybrid', request)
    return response.data
  },

  getAnalytics: async (days = 30): Promise<AnalyticsData> => {
    const response = await api.get<AnalyticsData>('/search/analytics', {
      params: { days }
    })
    return response.data
  },

  getStats: async (): Promise<any> => {
    const response = await api.get('/search/stats')
    return response.data
  },
}

// Widget API
export const widgetApi = {
  chat: async (request: WidgetChatRequest): Promise<WidgetChatResponse> => {
    const response = await api.post<WidgetChatResponse>('/widget/chat', request)
    return response.data
  },

  getConfig: async (): Promise<WidgetConfig> => {
    const response = await api.get<WidgetConfig>('/widget/config')
    return response.data
  },

  getHealth: async (): Promise<any> => {
    const response = await api.get('/widget/health')
    return response.data
  },

  getStatus: async (): Promise<any> => {
    const response = await api.get('/widget/status')
    return response.data
  },
}

// Health API
export const healthApi = {
  basic: async (): Promise<HealthCheck> => {
    const response = await api.get<HealthCheck>('/health')
    return response.data
  },

  detailed: async (): Promise<HealthCheck> => {
    const response = await api.get<HealthCheck>('/health/detailed')
    return response.data
  },

  ready: async (): Promise<any> => {
    const response = await api.get('/health/ready')
    return response.data
  },

  live: async (): Promise<any> => {
    const response = await api.get('/health/live')
    return response.data
  },
}

// Utility functions
export const handleApiError = (error: AxiosError): string => {
  const data = error.response?.data as any
  if (data?.detail) {
    return data.detail
  } else if (data?.message) {
    return data.message
  } else if (error.message) {
    return error.message
  } else {
    return 'Une erreur inattendue s\'est produite'
  }
}

export const isApiError = (error: any): error is AxiosError => {
  return error.isAxiosError === true
}

export default api
