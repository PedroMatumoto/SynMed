import { useEffect, useRef } from 'react'

interface AuroraBackgroundProps {
  children?: React.ReactNode
  className?: string
}

export default function AuroraBackground({ children, className = '' }: AuroraBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }

    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)

    let animationId: number
    let time = 0

    const animate = () => {
    time += 0.005;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Background gradient (tons mais neutros e suaves)
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    gradient.addColorStop(0, `hsla(${210 + Math.sin(time) * 10}, 30%, 18%, 0.9)`);
    gradient.addColorStop(0.5, `hsla(${230 + Math.cos(time * 0.7) * 10}, 25%, 22%, 0.8)`);
    gradient.addColorStop(1, `hsla(${250 + Math.sin(time * 1.2) * 10}, 20%, 15%, 0.9)`);

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Aurora waves (tons discretos, translúcidos e com leve movimento)
    for (let i = 0; i < 3; i++) {
        const wave = ctx.createLinearGradient(
        0,
        canvas.height * (0.3 + i * 0.2),
        canvas.width,
        canvas.height * (0.7 + i * 0.1)
        );
        wave.addColorStop(0, `hsla(${200 + i * 15 + Math.sin(time + i) * 10}, 25%, 40%, 0.05)`);
        wave.addColorStop(0.5, `hsla(${210 + i * 10 + Math.cos(time * 0.8 + i) * 10}, 30%, 45%, 0.12)`);
        wave.addColorStop(1, `hsla(${220 + i * 10 + Math.sin(time * 1.1 + i) * 10}, 20%, 35%, 0.05)`);

        ctx.fillStyle = wave;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    animationId = requestAnimationFrame(animate);
    };


    animate()

    return () => {
      window.removeEventListener('resize', resizeCanvas)
      cancelAnimationFrame(animationId)
    }
  }, [])

  return (
    <div className={`relative overflow-hidden w-full h-full ${className}`}>
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full"
        style={{ background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%)' }}
      />
      <div className="relative z-10 w-full h-full flex items-center justify-center">{children}</div>
    </div>
  )
}