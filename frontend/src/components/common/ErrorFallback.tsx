import React from 'react'
import { ExclamationTriangleIcon, ArrowPathIcon } from '@heroicons/react/24/outline'

interface ErrorFallbackProps {
  error: Error
  resetErrorBoundary: () => void
}

const ErrorFallback: React.FC<ErrorFallbackProps> = ({
  error,
  resetErrorBoundary,
}) => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div className="text-center">
          <ExclamationTriangleIcon className="mx-auto h-12 w-12 text-error-500" />
          <h2 className="mt-6 text-3xl font-extrabold text-gray-900">
            Oups ! Une erreur s'est produite
          </h2>
          <p className="mt-2 text-sm text-gray-600">
            Nous nous excusons pour ce problème technique.
          </p>
        </div>
        
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Détails de l'erreur
          </h3>
          <div className="bg-gray-50 rounded-md p-3">
            <code className="text-sm text-gray-800 break-all">
              {error.message}
            </code>
          </div>
        </div>

        <div className="flex flex-col space-y-3">
          <button
            onClick={resetErrorBoundary}
            className="btn-primary flex items-center justify-center"
          >
            <ArrowPathIcon className="w-4 h-4 mr-2" />
            Réessayer
          </button>
          
          <button
            onClick={() => window.location.href = '/'}
            className="btn-outline"
          >
            Retour à l'accueil
          </button>
        </div>

        <div className="text-center">
          <p className="text-xs text-gray-500">
            Si le problème persiste, veuillez contacter le support technique.
          </p>
        </div>
      </div>
    </div>
  )
}

export default ErrorFallback
