import { useState, useCallback } from 'react'
import { ToastType } from '../app/components/ui/toast'

export interface ToastData {
  id: string
  type: ToastType
  title: string
  message?: string
  duration?: number
}

export function useToast() {
  const [toasts, setToasts] = useState<ToastData[]>([])

  const addToast = useCallback((toast: Omit<ToastData, 'id'>) => {
    const id = Math.random().toString(36).substr(2, 9)
    const newToast: ToastData = {
      id,
      ...toast
    }
    
    setToasts(prev => [...prev, newToast])
    return id
  }, [])

  const removeToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(toast => toast.id !== id))
  }, [])

  const showSuccess = useCallback((title: string, message?: string, duration?: number) => {
    return addToast({ type: 'success', title, message, duration })
  }, [addToast])

  const showError = useCallback((title: string, message?: string, duration?: number) => {
    return addToast({ type: 'error', title, message, duration })
  }, [addToast])

  const showWarning = useCallback((title: string, message?: string, duration?: number) => {
    return addToast({ type: 'warning', title, message, duration })
  }, [addToast])

  const showInfo = useCallback((title: string, message?: string, duration?: number) => {
    return addToast({ type: 'info', title, message, duration })
  }, [addToast])

  return {
    toasts,
    addToast,
    removeToast,
    showSuccess,
    showError,
    showWarning,
    showInfo
  }
}