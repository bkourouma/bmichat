import React from 'react'
import { Link } from 'react-router-dom'
import { HomeIcon, ArrowLeftIcon } from '@heroicons/react/24/outline'

const NotFound: React.FC = () => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8 text-center">
        <div>
          <h1 className="text-9xl font-bold text-primary-600">404</h1>
          <h2 className="mt-6 text-3xl font-extrabold text-gray-900">
            Page non trouvée
          </h2>
          <p className="mt-2 text-sm text-gray-600">
            Désolé, nous n'avons pas pu trouver la page que vous recherchez.
          </p>
        </div>
        
        <div className="mt-8 space-y-4">
          <Link
            to="/dashboard"
            className="btn-primary w-full flex items-center justify-center"
          >
            <HomeIcon className="w-4 h-4 mr-2" />
            Retour au tableau de bord
          </Link>
          
          <button
            onClick={() => window.history.back()}
            className="btn-outline w-full flex items-center justify-center"
          >
            <ArrowLeftIcon className="w-4 h-4 mr-2" />
            Page précédente
          </button>
        </div>

        <div className="mt-8">
          <p className="text-xs text-gray-500">
            Si vous pensez qu'il s'agit d'une erreur, veuillez contacter le support.
          </p>
        </div>
      </div>
    </div>
  )
}

export default NotFound
