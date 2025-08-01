import React from 'react'
import { useLocation } from 'react-router-dom'
import {
  BellIcon,
  MagnifyingGlassIcon,
  Cog6ToothIcon,
} from '@heroicons/react/24/outline'
import { useQuery } from '@tanstack/react-query'

import { healthApi } from '@/lib/api'
import { cn } from '@/lib/utils'

const pageNames: Record<string, string> = {
  '/dashboard': 'Tableau de bord',
  '/chat': 'Chat avec Akissi',
  '/documents': 'Gestion des documents',
  '/analytics': 'Analytiques et statistiques',
  '/settings': 'Paramètres',
}

const Header: React.FC = () => {
  const location = useLocation()
  const pageName = pageNames[location.pathname] || 'BMI Chat'

  // Health check query
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: healthApi.basic,
    refetchInterval: 30000, // Check every 30 seconds
    retry: false,
  })

  const isHealthy = health?.status === 'healthy'

  return (
    <header className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Page title */}
        <div className="flex items-center space-x-4">
          <h1 className="text-2xl font-semibold text-gray-900">
            {pageName}
          </h1>
          
          {/* Health indicator */}
          <div className="flex items-center space-x-2">
            <div
              className={cn(
                'w-2 h-2 rounded-full',
                isHealthy ? 'bg-success-500' : 'bg-error-500'
              )}
            />
            <span className="text-xs text-gray-500">
              {isHealthy ? 'Système opérationnel' : 'Problème détecté'}
            </span>
          </div>
        </div>

        {/* Right side */}
        <div className="flex items-center space-x-4">
          {/* Search */}
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />
            </div>
            <input
              type="text"
              placeholder="Rechercher..."
              className="block w-64 pl-10 pr-3 py-2 border border-gray-300 rounded-lg text-sm placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            />
          </div>

          {/* Notifications */}
          <button className="relative p-2 text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded-lg">
            <BellIcon className="h-6 w-6" />
            {/* Notification badge */}
            <span className="absolute top-1 right-1 block h-2 w-2 rounded-full bg-error-500"></span>
          </button>

          {/* Settings */}
          <button className="p-2 text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded-lg">
            <Cog6ToothIcon className="h-6 w-6" />
          </button>

          {/* User menu */}
          <div className="relative">
            <button className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-primary-500">
              <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-primary-600 rounded-full flex items-center justify-center">
                <span className="text-sm font-medium text-white">U</span>
              </div>
              <div className="hidden md:block text-left">
                <p className="text-sm font-medium text-gray-900">Utilisateur</p>
                <p className="text-xs text-gray-500">Administrateur</p>
              </div>
            </button>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header
