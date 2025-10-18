/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      boxShadow: {
        sombraPadrao: '0 0 30px 0 rgba(0, 0, 0, 0.2)',
      },
      colors: {
        primary: '#1e293b',
        secondary: '#334155',
        accent: '#6366f1',
        success: '#059669',
        warning: '#d97706',
        danger: '#dc2626',
        dark: {
          900: '#0f172a',
          800: '#1e293b',
          700: '#334155',
          600: '#475569',
          500: '#64748b',
          400: '#94a3b8',
          300: '#cbd5e1',
          200: '#e2e8f0',
          100: '#f1f5f9',
          50: '#f8fafc'
        },
        cinzaEscuro: '#1e293b',
        cinzaClaro: '#f8fafc',
        cinzaBordas: '#e2e8f0'
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        fredoka: ['Fredoka', 'cursive']
      },
      keyframes: {
        wave: {
          '0%': { transform: 'rotate(0deg)' },
          '15%': { transform: 'rotate(14deg)' },
          '30%': { transform: 'rotate(-8deg)' },
          '40%': { transform: 'rotate(14deg)' },
          '50%': { transform: 'rotate(-4deg)' },
          '60%': { transform: 'rotate(10deg)' },
          '70%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(0deg)' },
        },
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' }
        },
        aurora: {
          '0%': {
            backgroundPosition: '50% 50%, 50% 50%'
          },
          '50%': {
            backgroundPosition: '350% 50%, 350% 50%'
          },
          '100%': {
            backgroundPosition: '50% 50%, 50% 50%'
          }
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' }
        }
      },
      animation: {
        wave: 'wave 1.2s ease-in-out',
        fadeIn: 'fadeIn 0.5s ease-out',
        aurora: 'aurora 60s ease infinite',
        float: 'float 6s ease-in-out infinite'
      }
    }
  },
  plugins: [require('@tailwindcss/typography')]
}
