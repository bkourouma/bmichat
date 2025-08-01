import React, { useState } from 'react'
import {
  Cog6ToothIcon,
  SparklesIcon,
  DocumentTextIcon,
  ChatBubbleLeftRightIcon,
  BellIcon,
} from '@heroicons/react/24/outline'
import toast from 'react-hot-toast'

const Settings: React.FC = () => {
  const [settings, setSettings] = useState({
    // Chat settings
    defaultLanguage: 'fr',
    maxTokens: 1000,
    temperature: 0.7,
    maxContextChunks: 5,
    
    // Processing settings
    chunkSize: 1000,
    chunkOverlap: 200,
    enableQADetection: true,
    
    // Widget settings
    widgetEnabled: true,
    widgetPosition: 'right',
    widgetAccentColor: '#3b82f6',
    assistantName: 'Akissi',
    welcomeMessage: 'Bonjour ! Je suis Akissi, votre assistante BMI. Comment puis-je vous aider ?',
    
    // Notifications
    emailNotifications: true,
    systemAlerts: true,
    performanceReports: false,
  })

  const handleSave = () => {
    // TODO: Implement settings save
    toast.success('Paramètres sauvegardés')
  }

  const handleReset = () => {
    if (confirm('Êtes-vous sûr de vouloir réinitialiser tous les paramètres ?')) {
      // Reset to defaults
      toast.success('Paramètres réinitialisés')
    }
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-gradient-to-r from-gray-600 to-gray-700 rounded-xl p-8 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">
              Paramètres du système
            </h1>
            <p className="text-gray-100 text-lg">
              Configuration et personnalisation de BMI Chat
            </p>
          </div>
          <div className="hidden md:block">
            <div className="w-24 h-24 bg-white/10 rounded-full flex items-center justify-center">
              <Cog6ToothIcon className="w-12 h-12 text-white" />
            </div>
          </div>
        </div>
      </div>

      {/* Settings sections */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Chat settings */}
        <div className="card p-6">
          <div className="flex items-center space-x-3 mb-6">
            <div className="p-2 bg-primary-100 rounded-lg">
              <ChatBubbleLeftRightIcon className="w-6 h-6 text-primary-600" />
            </div>
            <h2 className="text-lg font-semibold text-gray-900">
              Paramètres de conversation
            </h2>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Langue par défaut
              </label>
              <select
                value={settings.defaultLanguage}
                onChange={(e) => setSettings({ ...settings, defaultLanguage: e.target.value })}
                className="input w-full"
              >
                <option value="fr">Français</option>
                <option value="en">English</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Tokens maximum par réponse
              </label>
              <input
                type="number"
                value={settings.maxTokens}
                onChange={(e) => setSettings({ ...settings, maxTokens: parseInt(e.target.value) })}
                className="input w-full"
                min="100"
                max="4000"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Température (créativité)
              </label>
              <input
                type="range"
                value={settings.temperature}
                onChange={(e) => setSettings({ ...settings, temperature: parseFloat(e.target.value) })}
                className="w-full"
                min="0"
                max="1"
                step="0.1"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Précis (0.0)</span>
                <span>{settings.temperature}</span>
                <span>Créatif (1.0)</span>
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Chunks de contexte maximum
              </label>
              <input
                type="number"
                value={settings.maxContextChunks}
                onChange={(e) => setSettings({ ...settings, maxContextChunks: parseInt(e.target.value) })}
                className="input w-full"
                min="1"
                max="10"
              />
            </div>
          </div>
        </div>

        {/* Document processing settings */}
        <div className="card p-6">
          <div className="flex items-center space-x-3 mb-6">
            <div className="p-2 bg-success-100 rounded-lg">
              <DocumentTextIcon className="w-6 h-6 text-success-600" />
            </div>
            <h2 className="text-lg font-semibold text-gray-900">
              Traitement des documents
            </h2>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Taille des chunks (caractères)
              </label>
              <input
                type="number"
                value={settings.chunkSize}
                onChange={(e) => setSettings({ ...settings, chunkSize: parseInt(e.target.value) })}
                className="input w-full"
                min="500"
                max="2000"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Chevauchement des chunks
              </label>
              <input
                type="number"
                value={settings.chunkOverlap}
                onChange={(e) => setSettings({ ...settings, chunkOverlap: parseInt(e.target.value) })}
                className="input w-full"
                min="0"
                max="500"
              />
            </div>
            
            <div className="flex items-center">
              <input
                type="checkbox"
                id="qaDetection"
                checked={settings.enableQADetection}
                onChange={(e) => setSettings({ ...settings, enableQADetection: e.target.checked })}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
              />
              <label htmlFor="qaDetection" className="ml-2 block text-sm text-gray-900">
                Activer la détection automatique Q&A
              </label>
            </div>
          </div>
        </div>

        {/* Widget settings */}
        <div className="card p-6">
          <div className="flex items-center space-x-3 mb-6">
            <div className="p-2 bg-warning-100 rounded-lg">
              <SparklesIcon className="w-6 h-6 text-warning-600" />
            </div>
            <h2 className="text-lg font-semibold text-gray-900">
              Widget de chat
            </h2>
          </div>
          
          <div className="space-y-4">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="widgetEnabled"
                checked={settings.widgetEnabled}
                onChange={(e) => setSettings({ ...settings, widgetEnabled: e.target.checked })}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
              />
              <label htmlFor="widgetEnabled" className="ml-2 block text-sm text-gray-900">
                Activer le widget de chat
              </label>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Position du widget
              </label>
              <select
                value={settings.widgetPosition}
                onChange={(e) => setSettings({ ...settings, widgetPosition: e.target.value })}
                className="input w-full"
              >
                <option value="left">Gauche</option>
                <option value="right">Droite</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Couleur d'accent
              </label>
              <input
                type="color"
                value={settings.widgetAccentColor}
                onChange={(e) => setSettings({ ...settings, widgetAccentColor: e.target.value })}
                className="h-10 w-20 border border-gray-300 rounded-lg"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Nom de l'assistant
              </label>
              <input
                type="text"
                value={settings.assistantName}
                onChange={(e) => setSettings({ ...settings, assistantName: e.target.value })}
                className="input w-full"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Message de bienvenue
              </label>
              <textarea
                value={settings.welcomeMessage}
                onChange={(e) => setSettings({ ...settings, welcomeMessage: e.target.value })}
                className="input w-full"
                rows={3}
              />
            </div>
          </div>
        </div>

        {/* Notifications */}
        <div className="card p-6">
          <div className="flex items-center space-x-3 mb-6">
            <div className="p-2 bg-error-100 rounded-lg">
              <BellIcon className="w-6 h-6 text-error-600" />
            </div>
            <h2 className="text-lg font-semibold text-gray-900">
              Notifications
            </h2>
          </div>
          
          <div className="space-y-4">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="emailNotifications"
                checked={settings.emailNotifications}
                onChange={(e) => setSettings({ ...settings, emailNotifications: e.target.checked })}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
              />
              <label htmlFor="emailNotifications" className="ml-2 block text-sm text-gray-900">
                Notifications par email
              </label>
            </div>
            
            <div className="flex items-center">
              <input
                type="checkbox"
                id="systemAlerts"
                checked={settings.systemAlerts}
                onChange={(e) => setSettings({ ...settings, systemAlerts: e.target.checked })}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
              />
              <label htmlFor="systemAlerts" className="ml-2 block text-sm text-gray-900">
                Alertes système
              </label>
            </div>
            
            <div className="flex items-center">
              <input
                type="checkbox"
                id="performanceReports"
                checked={settings.performanceReports}
                onChange={(e) => setSettings({ ...settings, performanceReports: e.target.checked })}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
              />
              <label htmlFor="performanceReports" className="ml-2 block text-sm text-gray-900">
                Rapports de performance hebdomadaires
              </label>
            </div>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex justify-end space-x-4">
        <button
          onClick={handleReset}
          className="btn-outline"
        >
          Réinitialiser
        </button>
        <button
          onClick={handleSave}
          className="btn-primary"
        >
          Sauvegarder les paramètres
        </button>
      </div>
    </div>
  )
}

export default Settings
