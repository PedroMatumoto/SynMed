import { useEffect, useState } from 'react'

export default function BackGroundAnimatedGrid({
  squareSize
}: {
  squareSize: number
}) {
  const [squares, setSquares] = useState<Array<{ id: number; top: number; left: number }>>([])

  useEffect(() => {
    const width = window.innerWidth
    const height = window.innerHeight

    const cols = Math.ceil(width / squareSize)
    const rows = Math.ceil(height / squareSize)

    const newSquares = []
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        newSquares.push({
          id: i * cols + j,
          top: i * squareSize,
          left: j * squareSize
        })
      }
    }

    setSquares(newSquares)
  }, [squareSize])

  return (
    <div className="absolute inset-0 overflow-hidden bg-gradient-to-br from-primary to-secondary">
      {squares.map((square) => (
        <div
          key={square.id}
          className="absolute border border-white/10 transition-all duration-500 hover:bg-white/5"
          style={{
            top: `${square.top}px`,
            left: `${square.left}px`,
            width: `${squareSize}px`,
            height: `${squareSize}px`
          }}
        />
      ))}
    </div>
  )
}
