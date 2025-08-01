// API Response Types
export interface ApiResponse<T = any> {
  data?: T
  message?: string
  error?: string
  status?: number
}

// Chat Types
export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp?: string
}

export interface ChatRequest {
  message: string
  session_id?: string
  user_id?: string
  keywords?: string[]
  use_history?: boolean
  max_context_chunks?: number
}

export interface ChatResponse {
  message: string
  session_id: string
  message_id: string
  sources: DocumentSource[]
  timestamp: string
  response_time_ms: number
  tokens_used: number
  context_used: number
  has_history: boolean
}

export interface DocumentSource {
  document_id?: string
  filename?: string
  chunk_type?: string
  relevance_score?: number
}

export interface ChatSession {
  session_id: string
  status: 'active' | 'inactive' | 'archived'
  started_at: string
  last_message_at: string
  message_count: number
  total_tokens_used: number
  total_cost: number
  average_tokens_per_message: number
  duration_minutes?: number
  user_info: {
    preferred_language: string
    total_sessions: number
    total_messages: number
  }
}

// Document Types
export interface Document {
  id: string
  filename: string
  file_type: string
  file_size: number
  upload_date: string
  status: 'uploaded' | 'processing' | 'processed' | 'failed' | 'deleted'
  chunk_count?: number
}

export interface DocumentInfo {
  id: string
  filename: string
  original_filename?: string
  file_type: string
  file_size: number
  file_size_mb?: number
  mime_type?: string
  status: string
  processing_started_at?: string
  processing_completed_at?: string
  processing_duration?: number
  processing_error?: string
  total_pages?: number
  total_words?: number
  total_characters?: number
  language?: string
  chunk_count: number
  chunk_size?: number
  chunk_overlap?: number
  embedding_model?: string
  vector_count?: number
  title?: string
  author?: string
  subject?: string
  keywords?: string
  query_count?: number
  last_queried?: string
  is_active?: boolean
  is_public?: boolean
  created_at?: string
  updated_at?: string
}

export interface DocumentListResponse {
  documents: Document[]
  total_count: number
}

// Search Types
export interface SearchRequest {
  query: string
  k?: number
  keywords?: string[]
  document_ids?: string[]
  chunk_type?: string
  min_score?: number
}

export interface SearchResult {
  content: string
  score: number
  distance: number
  chunk_type: string
  has_questions: boolean
  has_answers: boolean
  keywords: string
  document_info?: {
    id?: string
    filename?: string
    title?: string
  }
  relevance_explanation: string
}

export interface SearchResponse {
  results: SearchResult[]
  total_found: number
  query: string
  search_time_ms?: number
}

// Analytics Types
export interface AnalyticsData {
  vector_database: {
    total_chunks: number
    total_documents: number
    chunk_types: Record<string, number>
    top_keywords: Record<string, number>
    average_chunk_length: number
    qa_coverage: number
    has_questions_count: number
    has_answers_count: number
  }
  retrieval_performance: {
    average_response_time_ms: number
    success_rate: number
    total_queries: number
  }
  content_analysis: {
    most_retrieved_keywords: Record<string, number>
    qa_pair_usage: number
    chunk_type_distribution: Record<string, number>
  }
}

// UI State Types
export interface LoadingState {
  isLoading: boolean
  message?: string
}

export interface ErrorState {
  hasError: boolean
  message?: string
  details?: string
}

// Form Types
export interface DocumentUploadForm {
  file: File | null
  keywords: string
}

export interface ChatForm {
  message: string
  keywords: string[]
}

// Widget Types
export interface WidgetConfig {
  position: 'left' | 'right'
  accent_color: string
  company_name: string
  assistant_name: string
  welcome_message: string
  enabled: boolean
}

export interface WidgetChatRequest {
  message: string
  session_id?: string
  widget_key?: string
}

export interface WidgetChatResponse {
  message: string
  session_id: string
  timestamp: string
}

// Health Check Types
export interface HealthCheck {
  status: 'healthy' | 'unhealthy' | 'degraded'
  timestamp: string
  version: string
  environment: string
  service: string
  checks?: Record<string, {
    status: string
    message: string
  }>
}

// Utility Types
export type Status = 'idle' | 'loading' | 'success' | 'error'

export interface PaginationParams {
  skip?: number
  limit?: number
}

export interface FilterParams {
  search?: string
  status?: string
  file_type?: string
  keywords?: string[]
}

// Component Props Types
export interface BaseComponentProps {
  className?: string
  children?: React.ReactNode
}

export interface ButtonProps extends BaseComponentProps {
  variant?: 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'ghost' | 'outline'
  size?: 'sm' | 'md' | 'lg'
  disabled?: boolean
  loading?: boolean
  onClick?: () => void
  type?: 'button' | 'submit' | 'reset'
}

export interface InputProps extends BaseComponentProps {
  type?: string
  placeholder?: string
  value?: string
  onChange?: (value: string) => void
  error?: string
  disabled?: boolean
  required?: boolean
}

export interface ModalProps extends BaseComponentProps {
  isOpen: boolean
  onClose: () => void
  title?: string
  size?: 'sm' | 'md' | 'lg' | 'xl'
}
