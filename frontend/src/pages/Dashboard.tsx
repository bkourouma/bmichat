import React from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  ChatBubbleLeftRightIcon,
  DocumentTextIcon,
  ChartBarIcon,
  ClockIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline'

import { searchApi, documentsApi } from '@/lib/api'
import { formatNumber, formatFileSize, formatDuration } from '@/lib/utils'
import LoadingSpinner from '@/components/common/LoadingSpinner'

const Dashboard: React.FC = () => {
  // Fetch analytics data
  const { data: analytics, isLoading: analyticsLoading } = useQuery({
    queryKey: ['analytics'],
    queryFn: () => searchApi.getAnalytics(30),
    refetchInterval: 60000, // Refresh every minute
  })

  // Fetch documents data
  const { data: documents, isLoading: documentsLoading } = useQuery({
    queryKey: ['documents'],
    queryFn: () => documentsApi.list(0, 10),
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  // Fetch vector stats
  const { data: vectorStats, isLoading: vectorLoading } = useQuery({
    queryKey: ['vector-stats'],
    queryFn: searchApi.getStats,
    refetchInterval: 60000,
  })

  const isLoading = analyticsLoading || documentsLoading || vectorLoading

  if (isLoading) {
    return <LoadingSpinner message="Chargement du tableau de bord..." />
  }

  const stats = [
    {
      name: 'Documents traités',
      value: vectorStats?.total_documents || 0,
      icon: DocumentTextIcon,
      color: 'text-primary-600',
      bgColor: 'bg-primary-50',
    },
    {
      name: 'Chunks vectorisés',
      value: formatNumber(vectorStats?.total_chunks || 0),
      icon: SparklesIcon,
      color: 'text-success-600',
      bgColor: 'bg-success-50',
    },
    {
      name: 'Requêtes traitées',
      value: formatNumber(analytics?.retrieval_performance?.total_queries || 0),
      icon: ChatBubbleLeftRightIcon,
      color: 'text-warning-600',
      bgColor: 'bg-warning-50',
    },
    {
      name: 'Taux de succès',
      value: `${Math.round((analytics?.retrieval_performance?.success_rate || 0) * 100)}%`,
      icon: ChartBarIcon,
      color: 'text-error-600',
      bgColor: 'bg-error-50',
    },
  ]

  const recentDocuments = documents?.documents?.slice(0, 5) || []

  return (
    <div className="space-y-8">
      {/* Welcome section */}
      <div className="bg-gradient-to-r from-primary-600 to-primary-700 rounded-xl p-8 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">
              Bienvenue sur BMI Chat
            </h1>
            <p className="text-primary-100 text-lg">
              Votre assistant IA intelligent pour répondre aux questions sur vos documents
            </p>
          </div>
          <div className="hidden md:block">
            <div className="w-24 h-24 bg-white/10 rounded-full flex items-center justify-center">
              <SparklesIcon className="w-12 h-12 text-white" />
            </div>
          </div>
        </div>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => (
          <div key={stat.name} className="card p-6">
            <div className="flex items-center">
              <div className={`p-3 rounded-lg ${stat.bgColor}`}>
                <stat.icon className={`w-6 h-6 ${stat.color}`} />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">{stat.name}</p>
                <p className="text-2xl font-semibold text-gray-900">{stat.value}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Recent documents */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold text-gray-900">
              Documents récents
            </h2>
            <a
              href="/documents"
              className="text-sm text-primary-600 hover:text-primary-700 font-medium"
            >
              Voir tout
            </a>
          </div>
          
          <div className="space-y-4">
            {recentDocuments.length > 0 ? (
              recentDocuments.map((doc) => (
                <div key={doc.id} className="flex items-center space-x-3 p-3 rounded-lg hover:bg-gray-50">
                  <div className="flex-shrink-0">
                    <DocumentTextIcon className="w-8 h-8 text-gray-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {doc.filename}
                    </p>
                    <div className="flex items-center space-x-2 text-xs text-gray-500">
                      <span>{doc.file_type.toUpperCase()}</span>
                      <span>•</span>
                      <span>{formatFileSize(doc.file_size)}</span>
                      <span>•</span>
                      <span className={`badge ${
                        doc.status === 'processed' ? 'badge-success' :
                        doc.status === 'processing' ? 'badge-warning' :
                        doc.status === 'failed' ? 'badge-error' : 'badge-secondary'
                      }`}>
                        {doc.status}
                      </span>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8">
                <DocumentTextIcon className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500">Aucun document trouvé</p>
              </div>
            )}
          </div>
        </div>

        {/* Performance metrics */}
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-6">
            Performances du système
          </h2>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <ClockIcon className="w-5 h-5 text-gray-400" />
                <span className="text-sm text-gray-600">Temps de réponse moyen</span>
              </div>
              <span className="text-sm font-medium text-gray-900">
                {formatDuration(analytics?.retrieval_performance?.average_response_time_ms || 0)}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <SparklesIcon className="w-5 h-5 text-gray-400" />
                <span className="text-sm text-gray-600">Couverture Q&A</span>
              </div>
              <span className="text-sm font-medium text-gray-900">
                {Math.round(vectorStats?.qa_coverage || 0)}%
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <ChartBarIcon className="w-5 h-5 text-gray-400" />
                <span className="text-sm text-gray-600">Longueur moyenne des chunks</span>
              </div>
              <span className="text-sm font-medium text-gray-900">
                {Math.round(vectorStats?.average_chunk_length || 0)} caractères
              </span>
            </div>
          </div>

          {/* Quick actions */}
          <div className="mt-6 pt-6 border-t border-gray-200">
            <h3 className="text-sm font-medium text-gray-900 mb-3">Actions rapides</h3>
            <div className="space-y-2">
              <a
                href="/chat"
                className="block w-full btn-primary text-center"
              >
                Démarrer une conversation
              </a>
              <a
                href="/documents"
                className="block w-full btn-outline text-center"
              >
                Ajouter un document
              </a>
            </div>
          </div>
        </div>
      </div>

      {/* Top keywords */}
      {vectorStats?.top_keywords && Object.keys(vectorStats.top_keywords).length > 0 && (
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-6">
            Mots-clés les plus fréquents
          </h2>
          <div className="flex flex-wrap gap-2">
            {Object.entries(vectorStats.top_keywords)
              .slice(0, 10)
              .map(([keyword, count]) => (
                <span
                  key={keyword}
                  className="badge badge-primary"
                  title={`${count} occurrences`}
                >
                  {keyword}
                </span>
              ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default Dashboard
