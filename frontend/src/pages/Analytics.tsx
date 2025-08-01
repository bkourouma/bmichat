import React from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  ChartBarIcon,
  DocumentTextIcon,
  ChatBubbleLeftRightIcon,
  ClockIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline'

import { searchApi } from '@/lib/api'
import { formatNumber, formatPercentage, formatDuration } from '@/lib/utils'
import LoadingSpinner from '@/components/common/LoadingSpinner'

const Analytics: React.FC = () => {
  // Fetch analytics data
  const { data: analytics, isLoading } = useQuery({
    queryKey: ['analytics', 30],
    queryFn: () => searchApi.getAnalytics(30),
    refetchInterval: 60000, // Refresh every minute
  })

  // Fetch vector stats
  const { data: vectorStats, isLoading: vectorLoading } = useQuery({
    queryKey: ['vector-stats'],
    queryFn: searchApi.getStats,
    refetchInterval: 60000,
  })

  if (isLoading || vectorLoading) {
    return <LoadingSpinner message="Chargement des analytiques..." />
  }

  const performanceMetrics = [
    {
      name: 'Temps de réponse moyen',
      value: formatDuration(analytics?.retrieval_performance?.average_response_time_ms || 0),
      icon: ClockIcon,
      color: 'text-primary-600',
      bgColor: 'bg-primary-50',
    },
    {
      name: 'Taux de succès',
      value: formatPercentage((analytics?.retrieval_performance?.success_rate || 0) * 100),
      icon: ChartBarIcon,
      color: 'text-success-600',
      bgColor: 'bg-success-50',
    },
    {
      name: 'Requêtes totales',
      value: formatNumber(analytics?.retrieval_performance?.total_queries || 0),
      icon: ChatBubbleLeftRightIcon,
      color: 'text-warning-600',
      bgColor: 'bg-warning-50',
    },
    {
      name: 'Documents actifs',
      value: formatNumber(vectorStats?.total_documents || 0),
      icon: DocumentTextIcon,
      color: 'text-error-600',
      bgColor: 'bg-error-50',
    },
  ]

  const contentMetrics = [
    {
      name: 'Total des chunks',
      value: formatNumber(vectorStats?.total_chunks || 0),
      description: 'Segments de texte indexés',
    },
    {
      name: 'Couverture Q&A',
      value: formatPercentage(vectorStats?.qa_coverage || 0),
      description: 'Pourcentage de contenu Q&A',
    },
    {
      name: 'Longueur moyenne',
      value: `${Math.round(vectorStats?.average_chunk_length || 0)} chars`,
      description: 'Taille moyenne des chunks',
    },
    {
      name: 'Chunks avec questions',
      value: formatNumber(vectorStats?.has_questions_count || 0),
      description: 'Segments contenant des questions',
    },
  ]

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-gradient-to-r from-secondary-600 to-secondary-700 rounded-xl p-8 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">
              Analytiques et Performances
            </h1>
            <p className="text-secondary-100 text-lg">
              Suivi des performances et utilisation du système BMI Chat
            </p>
          </div>
          <div className="hidden md:block">
            <div className="w-24 h-24 bg-white/10 rounded-full flex items-center justify-center">
              <ChartBarIcon className="w-12 h-12 text-white" />
            </div>
          </div>
        </div>
      </div>

      {/* Performance metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {performanceMetrics.map((metric) => (
          <div key={metric.name} className="card p-6">
            <div className="flex items-center">
              <div className={`p-3 rounded-lg ${metric.bgColor}`}>
                <metric.icon className={`w-6 h-6 ${metric.color}`} />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">{metric.name}</p>
                <p className="text-2xl font-semibold text-gray-900">{metric.value}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Content analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Content metrics */}
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-6">
            Analyse du contenu
          </h2>
          
          <div className="space-y-4">
            {contentMetrics.map((metric) => (
              <div key={metric.name} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                  <p className="text-sm font-medium text-gray-900">{metric.name}</p>
                  <p className="text-xs text-gray-500">{metric.description}</p>
                </div>
                <p className="text-lg font-semibold text-gray-900">{metric.value}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Chunk types distribution */}
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-6">
            Distribution des types de chunks
          </h2>
          
          {vectorStats?.chunk_types ? (
            <div className="space-y-4">
              {Object.entries(vectorStats.chunk_types).map(([type, count]) => {
                const total = Object.values(vectorStats.chunk_types).reduce((a: number, b: unknown) => a + (b as number), 0)
                const percentage = total > 0 ? ((count as number) / total) * 100 : 0
                
                return (
                  <div key={type} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-700 capitalize">
                        {type.replace('_', ' ')}
                      </span>
                      <span className="text-sm text-gray-500">
                        {count as number} ({formatPercentage(percentage, 1)})
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-primary-600 h-2 rounded-full"
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="text-center py-8">
              <ChartBarIcon className="w-12 h-12 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-500">Aucune donnée disponible</p>
            </div>
          )}
        </div>
      </div>

      {/* Top keywords */}
      <div className="card p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-6">
          Mots-clés les plus fréquents
        </h2>
        
        {vectorStats?.top_keywords && Object.keys(vectorStats.top_keywords).length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(vectorStats.top_keywords)
              .slice(0, 12)
              .map(([keyword, count], index) => (
                <div key={keyword} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
                      <span className="text-sm font-medium text-primary-700">
                        {index + 1}
                      </span>
                    </div>
                    <span className="text-sm font-medium text-gray-900">{keyword}</span>
                  </div>
                  <span className="text-sm text-gray-500">{count as number}</span>
                </div>
              ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <SparklesIcon className="w-12 h-12 text-gray-300 mx-auto mb-4" />
            <p className="text-gray-500">Aucun mot-clé trouvé</p>
          </div>
        )}
      </div>

      {/* System health */}
      <div className="card p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-6">
          État du système
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="w-16 h-16 bg-success-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <ChartBarIcon className="w-8 h-8 text-success-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900">Base vectorielle</h3>
            <p className="text-sm text-gray-500">Opérationnelle</p>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 bg-success-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <SparklesIcon className="w-8 h-8 text-success-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900">IA Conversationnelle</h3>
            <p className="text-sm text-gray-500">Opérationnelle</p>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 bg-success-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <DocumentTextIcon className="w-8 h-8 text-success-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900">Traitement docs</h3>
            <p className="text-sm text-gray-500">Opérationnel</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Analytics
