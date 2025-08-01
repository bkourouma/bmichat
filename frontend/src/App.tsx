import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { ErrorBoundary } from 'react-error-boundary'

import Layout from '@/components/layout/Layout'
import ErrorFallback from '@/components/common/ErrorFallback'
import LoadingSpinner from '@/components/common/LoadingSpinner'

// Lazy load pages for better performance
const Dashboard = React.lazy(() => import('@/pages/Dashboard'))
const Chat = React.lazy(() => import('@/pages/Chat'))
const Documents = React.lazy(() => import('@/pages/Documents'))
const Analytics = React.lazy(() => import('@/pages/Analytics'))
const Settings = React.lazy(() => import('@/pages/Settings'))
const NotFound = React.lazy(() => import('@/pages/NotFound'))

function App() {
  return (
    <ErrorBoundary
      FallbackComponent={ErrorFallback}
      onError={(error, errorInfo) => {
        console.error('Application error:', error, errorInfo)
        // Here you could send error to monitoring service
      }}
    >
      <div className="min-h-screen bg-gray-50">
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route
              path="dashboard"
              element={
                <React.Suspense fallback={<LoadingSpinner />}>
                  <Dashboard />
                </React.Suspense>
              }
            />
            <Route
              path="chat"
              element={
                <React.Suspense fallback={<LoadingSpinner />}>
                  <Chat />
                </React.Suspense>
              }
            />
            <Route
              path="documents"
              element={
                <React.Suspense fallback={<LoadingSpinner />}>
                  <Documents />
                </React.Suspense>
              }
            />
            <Route
              path="analytics"
              element={
                <React.Suspense fallback={<LoadingSpinner />}>
                  <Analytics />
                </React.Suspense>
              }
            />
            <Route
              path="settings"
              element={
                <React.Suspense fallback={<LoadingSpinner />}>
                  <Settings />
                </React.Suspense>
              }
            />
          </Route>
          <Route
            path="*"
            element={
              <React.Suspense fallback={<LoadingSpinner />}>
                <NotFound />
              </React.Suspense>
            }
          />
        </Routes>
      </div>
    </ErrorBoundary>
  )
}

export default App
