/** GrowthSpine — the signature visual element of the app.
 *  Five horizontal dots (one per class) representing the growth timeline
 *  germination → young → pod → Ready, with the predicted stage filled in
 *  its class color and connector lines drawn between adjacent stages.
 *  Appears in every route. Don't redesign it per-view — that's the point. */
import { CLASSES, CLASS_META, type LettuceClass } from '../types'

interface Props {
  active: LettuceClass | null
  /** "lg" = hero card  •  "sm" = inline in a row  •  "axis" = chart label */
  size?: 'lg' | 'sm' | 'axis'
  /** When true, show the class token label under the active dot */
  showLabel?: boolean
}

const DIMENSIONS = {
  lg: { dot: 12, gap: 28, line: 2, labelPx: 11 },
  sm: { dot:  8, gap: 14, line: 1, labelPx:  9 },
  axis:{ dot:  6, gap: 10, line: 1, labelPx:  0 },
} as const

export function GrowthSpine({ active, size = 'lg', showLabel = false }: Props) {
  const dims = DIMENSIONS[size]
  const activeIdx = active ? CLASSES.indexOf(active) : -1
  const totalWidth = dims.dot * CLASSES.length + dims.gap * (CLASSES.length - 1)

  return (
    <div
      className="flex items-center"
      role="img"
      aria-label={
        active
          ? `Growth-stage spine, predicted stage: ${active} (${CLASS_META[active].verb})`
          : 'Growth-stage spine, no prediction yet'
      }
      style={{ width: totalWidth, height: dims.dot + (showLabel ? 18 : 0) }}
    >
      <svg
        width={totalWidth}
        height={dims.dot + (showLabel ? 18 : 0)}
        viewBox={`0 0 ${totalWidth} ${dims.dot + (showLabel ? 18 : 0)}`}
        fill="none"
      >
        {/* Connector lines first so dots sit on top */}
        {CLASSES.slice(0, -1).map((_, i) => {
          const x1 = dims.dot * (i + 1) + dims.gap * i
          const x2 = x1 + dims.gap
          const y = dims.dot / 2
          const reached = activeIdx >= 0 && i < activeIdx
          return (
            <line
              key={`l${i}`}
              x1={x1}
              x2={x2}
              y1={y}
              y2={y}
              stroke={reached ? CLASS_META[CLASSES[i + 1]].cssVar : 'var(--color-border)'}
              strokeWidth={dims.line}
            />
          )
        })}

        {CLASSES.map((cls, i) => {
          const cx = dims.dot * i + dims.gap * i + dims.dot / 2
          const cy = dims.dot / 2
          const isActive = i === activeIdx
          const isReached = activeIdx >= 0 && i <= activeIdx
          return (
            <circle
              key={cls}
              cx={cx}
              cy={cy}
              r={dims.dot / 2}
              fill={isReached ? CLASS_META[cls].cssVar : 'transparent'}
              stroke={isReached ? CLASS_META[cls].cssVar : 'var(--color-ink-mute)'}
              strokeWidth={dims.line}
              opacity={isActive ? 1 : isReached ? 0.85 : 0.55}
            />
          )
        })}

        {showLabel && activeIdx >= 0 && (
          <text
            x={dims.dot * activeIdx + dims.gap * activeIdx + dims.dot / 2}
            y={dims.dot + 14}
            fontSize={dims.labelPx}
            fontFamily="var(--font-mono)"
            fill="var(--color-ink-2)"
            textAnchor="middle"
            className="tabular"
          >
            {CLASS_META[active!].token}
          </text>
        )}
      </svg>
    </div>
  )
}
