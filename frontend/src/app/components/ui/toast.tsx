import { useState, useEffect } from 'react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faCheckCircle, faTimes, faExclamationTriangle, faInfoCircle } from '@fortawesome/free-solid-svg-icons'

export type ToastType = 'success' | 'error' | 'warning' | 'info'

export interface ToastProps {
  id: string
  type: ToastType
  title: string
  message?: string
  duration?: number
  onClose: (id: string) => void
}

export function Toast({ id, type, title, message, duration = 5000, onClose }: ToastProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [isLeaving, setIsLeaving] = useState(false)

  useEffect(() => {
    // Animate in
    setTimeout(() => setIsVisible(true), 100)

    // Auto close
    const timer = setTimeout(() => {
      handleClose()
    }, duration)

    return () => clearTimeout(timer)
  }, [duration])

  const handleClose = () => {
    setIsLeaving(true)
    setTimeout(() => {
      onClose(id)
    }, 300)
  }

  const getIcon = () => {
    switch (type) {
      case 'success':
        return faCheckCircle
      case 'error':
        return faExclamationTriangle
      case 'warning':
        return faExclamationTriangle
      case 'info':
        return faInfoCircle
      default:
        return faInfoCircle
    }
  }

  const getStyles = () => {
    switch (type) {
      case 'success':
        return 'bg-green-50 border-green-200 text-green-800'
      case 'error':
        return 'bg-red-50 border-red-200 text-red-800'
      case 'warning':
        return 'bg-yellow-50 border-yellow-200 text-yellow-800'
      case 'info':
        return 'bg-blue-50 border-blue-200 text-blue-800'
      default:
        return 'bg-gray-50 border-gray-200 text-gray-800'
    }
  }

  const getIconStyles = () => {
    switch (type) {
      case 'success':
        return 'text-green-600'
      case 'error':
        return 'text-red-600'
      case 'warning':
        return 'text-yellow-600'
      case 'info':
        return 'text-blue-600'
      default:
        return 'text-gray-600'
    }
  }

  return (
    <div
      className={`
        fixed top-4 right-4 z-50 max-w-sm w-full transform transition-all duration-300 ease-out
        ${isVisible && !isLeaving ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'}
      `}
    >
      <div className={`rounded-xl border shadow-lg p-4 ${getStyles()}`}>
        <div className="flex items-start gap-3">
          <FontAwesomeIcon 
            icon={getIcon()} 
            className={`text-xl mt-0.5 ${getIconStyles()}`} 
          />
          <div className="flex-1">
            <h4 className="font-semibold text-sm mb-1">{title}</h4>
            {message && (
              <p className="text-sm opacity-90">{message}</p>
            )}
          </div>
          <button
            onClick={handleClose}
            className="text-current opacity-50 hover:opacity-100 transition-opacity"
          >
            <FontAwesomeIcon icon={faTimes} className="text-sm" />
          </button>
        </div>
      </div>
    </div>
  )
}