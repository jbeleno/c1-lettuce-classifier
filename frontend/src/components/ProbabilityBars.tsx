import { CLASSES, CLASS_META, type LettuceClass } from '../types'

interface Props {
  probabilities: Record<LettuceClass, number>
  predicted: LettuceClass
}

/** Horizontal monospaced bars, aligned tabularly like a Lighthouse report.
 *  Only the predicted-class row gets its accent color; the rest stay muted
 *  so the eye lands on the answer first. */
export function ProbabilityBars({ probabilities, predicted }: Props) {
  return (
    <ul className="space-y-1.5" aria-label="Class probabilities">
      {CLASSES.map((cls) => {
        const value = probabilities[cls] ?? 0
        const pct = Math.max(value * 100, 0.1)
        const isPredicted = cls === predicted
        const fill = isPredicted
          ? CLASS_META[cls].cssVar
          : 'color-mix(in oklab, var(--color-ink-mute) 35%, transparent)'
        return (
          <li key={cls} className="flex items-center gap-3 font-mono text-[12px] tabular">
            <span
              className="w-24 truncate"
              style={{ color: isPredicted ? CLASS_META[cls].cssVar : 'var(--color-ink-3)' }}
            >
              {cls}
            </span>
            <div
              className="relative h-1.5 flex-1 rounded-full"
              style={{ background: 'var(--color-surface-inset)' }}
              role="meter"
              aria-valuemin={0}
              aria-valuemax={1}
              aria-valuenow={Number(value.toFixed(4))}
              aria-label={`${cls} confidence ${(value * 100).toFixed(1)} percent`}
            >
              <div
                className="absolute inset-y-0 left-0 rounded-full"
                style={{ width: `${pct}%`, background: fill, transition: 'width 240ms cubic-bezier(0.2, 0.8, 0.2, 1)' }}
              />
            </div>
            <span
              className="w-16 text-right"
              style={{ color: isPredicted ? 'var(--color-ink)' : 'var(--color-ink-3)' }}
            >
              {(value * 100).toFixed(2)}%
            </span>
          </li>
        )
      })}
    </ul>
  )
}
