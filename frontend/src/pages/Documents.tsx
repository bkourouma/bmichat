import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  DocumentTextIcon,
  CloudArrowUpIcon,
  TrashIcon,
  EyeIcon,
  MagnifyingGlassIcon,
} from '@heroicons/react/24/outline'
import toast from 'react-hot-toast'

import { documentsApi } from '@/lib/api'
import { formatFileSize, formatDate } from '@/lib/utils'
import LoadingSpinner from '@/components/common/LoadingSpinner'
import type { Document } from '@/types'

const Documents: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [keywords, setKeywords] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const queryClient = useQueryClient()

  // Fetch documents
  const { data: documentsData, isLoading } = useQuery({
    queryKey: ['documents'],
    queryFn: () => documentsApi.list(0, 100),
    refetchInterval: 10000, // Refresh every 10 seconds
  })

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: ({ file, keywords }: { file: File; keywords?: string }) =>
      documentsApi.upload(file, keywords),
    onSuccess: () => {
      toast.success('Document téléchargé avec succès')
      setSelectedFile(null)
      setKeywords('')
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Erreur lors du téléchargement')
    },
  })

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: documentsApi.delete,
    onSuccess: () => {
      toast.success('Document supprimé')
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Erreur lors de la suppression')
    },
  })

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
    }
  }

  const handleUpload = () => {
    if (!selectedFile) return
    uploadMutation.mutate({ file: selectedFile, keywords: keywords || undefined })
  }

  const handleDelete = (documentId: string) => {
    if (confirm('Êtes-vous sûr de vouloir supprimer ce document ?')) {
      deleteMutation.mutate(documentId)
    }
  }

  const getStatusBadge = (status: string) => {
    const statusMap = {
      uploaded: { class: 'badge-secondary', text: 'Téléchargé' },
      processing: { class: 'badge-warning', text: 'En cours' },
      processed: { class: 'badge-success', text: 'Traité' },
      failed: { class: 'badge-error', text: 'Échec' },
      deleted: { class: 'badge-secondary', text: 'Supprimé' },
    }
    const statusInfo = statusMap[status as keyof typeof statusMap] || statusMap.uploaded
    return <span className={`badge ${statusInfo.class}`}>{statusInfo.text}</span>
  }

  const filteredDocuments = documentsData?.documents?.filter(doc =>
    doc.filename.toLowerCase().includes(searchQuery.toLowerCase())
  ) || []

  if (isLoading) {
    return <LoadingSpinner message="Chargement des documents..." />
  }

  return (
    <div className="space-y-8">
      {/* Upload section */}
      <div className="card p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Télécharger un nouveau document
        </h2>
        
        <div className="space-y-4">
          {/* File input */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Sélectionner un fichier
            </label>
            <div className="flex items-center space-x-4">
              <input
                type="file"
                accept=".pdf,.txt,.docx,.md"
                onChange={handleFileSelect}
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-primary-50 file:text-primary-700 hover:file:bg-primary-100"
              />
              {selectedFile && (
                <div className="text-sm text-gray-600">
                  {selectedFile.name} ({formatFileSize(selectedFile.size)})
                </div>
              )}
            </div>
          </div>

          {/* Keywords input */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Mots-clés (optionnel)
            </label>
            <input
              type="text"
              value={keywords}
              onChange={(e) => setKeywords(e.target.value)}
              placeholder="ex: assurance, réclamation, support"
              className="input w-full"
            />
            <p className="mt-1 text-xs text-gray-500">
              Séparez les mots-clés par des virgules pour faciliter la recherche
            </p>
          </div>

          {/* Upload button */}
          <button
            onClick={handleUpload}
            disabled={!selectedFile || uploadMutation.isPending}
            className="btn-primary"
          >
            <CloudArrowUpIcon className="w-4 h-4 mr-2" />
            {uploadMutation.isPending ? 'Téléchargement...' : 'Télécharger'}
          </button>
        </div>

        {/* Supported formats */}
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <p className="text-sm font-medium text-gray-700 mb-2">Formats supportés :</p>
          <div className="flex flex-wrap gap-2">
            {['PDF', 'TXT', 'DOCX', 'MD'].map((format) => (
              <span key={format} className="badge badge-secondary">
                {format}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Documents list */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-gray-900">
            Documents ({documentsData?.total_count || 0})
          </h2>
          
          {/* Search */}
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />
            </div>
            <input
              type="text"
              placeholder="Rechercher..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="block w-64 pl-10 pr-3 py-2 border border-gray-300 rounded-lg text-sm"
            />
          </div>
        </div>

        {filteredDocuments.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Document
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Type
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Taille
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Statut
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Date
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Chunks
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredDocuments.map((document: Document) => (
                  <tr key={document.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <DocumentTextIcon className="w-8 h-8 text-gray-400 mr-3" />
                        <div>
                          <div className="text-sm font-medium text-gray-900">
                            {document.filename}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="badge badge-secondary">
                        {document.file_type.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {formatFileSize(document.file_size)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {getStatusBadge(document.status)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatDate(document.upload_date, { relative: true })}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {document.chunk_count || '-'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <div className="flex items-center justify-end space-x-2">
                        <button
                          className="text-primary-600 hover:text-primary-900"
                          title="Voir les détails"
                        >
                          <EyeIcon className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleDelete(document.id)}
                          disabled={deleteMutation.isPending}
                          className="text-error-600 hover:text-error-900"
                          title="Supprimer"
                        >
                          <TrashIcon className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-12">
            <DocumentTextIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              {searchQuery ? 'Aucun document trouvé' : 'Aucun document'}
            </h3>
            <p className="text-gray-500">
              {searchQuery
                ? 'Essayez de modifier votre recherche'
                : 'Commencez par télécharger votre premier document'}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

export default Documents
