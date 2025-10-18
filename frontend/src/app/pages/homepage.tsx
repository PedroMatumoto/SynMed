import { useNavigate } from 'react-router-dom'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPills, faArrowRight, faShieldAlt, faBrain, faChartLine } from '@fortawesome/free-solid-svg-icons'
import AuroraBackground from '../components/ui/auroraBackground'

export default function HomePage() {
  const navigate = useNavigate()

  return (
    <AuroraBackground className="min-h-screen w-full">
      <div className="flex min-h-screen w-full flex-col items-center justify-center px-4 py-8 font-sans">
        {/* Hero Section */}
        <div className="animate-fadeIn flex w-full max-w-3xl flex-col items-center justify-center gap-6 text-center text-white mx-auto">
          <div className="animate-float flex items-center gap-4">
            <div className="rounded-xl bg-gradient-to-br from-accent/20 to-white/10 p-3 backdrop-blur-sm">
              <FontAwesomeIcon icon={faPills} className="text-4xl text-white" />
            </div>
            <h1 className="text-5xl font-bold tracking-tight">SynMed</h1>
          </div>

          <div className="space-y-3">
            <p className="text-xl font-light text-dark-200">
              Sistema Inteligente de Análise de Efeitos Colaterais
            </p>
            <p className="text-base text-dark-300 max-w-2xl">
              Utilize inteligência artificial para identificar possíveis relações entre seus sintomas 
              e medicamentos, baseado em dados científicos reais.
            </p>
          </div>

          {/* Features Grid */}
          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4 w-full max-w-3xl">
            <div className="rounded-lg bg-white/5 backdrop-blur-sm border border-white/10 p-4 text-center hover:bg-white/10 transition-all duration-300">
              <FontAwesomeIcon icon={faShieldAlt} className="text-2xl text-accent mb-2" />
              <h3 className="font-semibold text-base mb-2">Análise Segura</h3>
              <p className="text-xs text-dark-300">Dados baseados em estudos científicos e bases confiáveis</p>
            </div>
            
            <div className="rounded-lg bg-white/5 backdrop-blur-sm border border-white/10 p-4 text-center hover:bg-white/10 transition-all duration-300">
              <FontAwesomeIcon icon={faBrain} className="text-2xl text-accent mb-2" />
              <h3 className="font-semibold text-base mb-2">IA Avançada</h3>
              <p className="text-xs text-dark-300">Processamento inteligente com modelo Gemma</p>
            </div>
            
            <div className="rounded-lg bg-white/5 backdrop-blur-sm border border-white/10 p-4 text-center hover:bg-white/10 transition-all duration-300">
              <FontAwesomeIcon icon={faChartLine} className="text-2xl text-accent mb-2" />
              <h3 className="font-semibold text-base mb-2">Histórico</h3>
              <p className="text-xs text-dark-300">Acompanhe suas consultas e análises anteriores</p>
            </div>
          </div>

          {/* CTA Buttons */}
          <div className="mt-8 flex flex-col sm:flex-row gap-3">
            <button
              onClick={() => navigate('/login')}
              className="group flex items-center justify-center gap-2 rounded-lg bg-accent px-6 py-3 text-base font-semibold text-white transition-all duration-300 hover:bg-accent/90 hover:scale-105 hover:shadow-2xl shadow-accent/25"
            >
              Começar Análise
              <FontAwesomeIcon
                icon={faArrowRight}
                className="text-sm transition-transform duration-300 group-hover:translate-x-1"
              />
            </button>

            <button
              onClick={() => navigate('/register')}
              className="rounded-lg border-2 border-white/20 bg-white/5 backdrop-blur-sm px-6 py-3 text-base font-semibold text-white transition-all duration-300 hover:bg-white/10 hover:border-white/30"
            >
              Criar Conta
            </button>
          </div>

          {/* Bottom Info */}
          <div className="mt-12 text-center text-xs text-dark-400 space-y-2 max-w-2xl">
            <p className="border-t border-white/10 pt-4">
              ⚠️ <strong>Importante:</strong> Este sistema é uma ferramenta de apoio. 
              Sempre consulte um profissional de saúde para diagnósticos definitivos.
            </p>
            <p className="text-xs">
              Análises baseadas em dados do SIDER, MedDRA e literatura científica
            </p>
          </div>
        </div>
      </div>
    </AuroraBackground>
  )
}
