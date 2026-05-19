import { useEffect, useState } from 'react'
import { listModels } from '../api'
import type { ModelInfo } from '../types'
import { Card } from '../components/Card'
import { GrowthSpine } from '../components/GrowthSpine'
import { CLASSES } from '../types'

export function Models() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    listModels()
      .then(setModels)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (error) {
    return (
      <Card className="p-6">
        <p role="alert" className="font-mono text-[12px]"
           style={{ color: 'var(--color-class-empty)' }}>
          {error}
        </p>
      </Card>
    )
  }
  if (loading) {
    return (
      <p className="font-mono text-[12px]" style={{ color: 'var(--color-ink-mute)' }}>
        loading models…
      </p>
    )
  }

  return (
    <div className="grid gap-4 sm:grid-cols-2">
      {models.map((m) => (
        <Card key={m.name} className="p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="font-mono text-[13px]" style={{ color: 'var(--color-ink)' }}>
                {m.name}
              </h3>
              <p className="font-mono text-[11px] uppercase tracking-wider"
                 style={{ color: 'var(--color-ink-3)' }}>
                {m.framework}
              </p>
            </div>
            <div className="text-right font-mono tabular">
              <p className="text-[20px]" style={{ color: 'var(--color-ink)' }}>
                {m.test_accuracy != null ? `${(m.test_accuracy * 100).toFixed(2)}%` : '—'}
              </p>
              <p className="text-[10px]" style={{ color: 'var(--color-ink-mute)' }}>
                test accuracy
              </p>
            </div>
          </div>

          <div className="mt-4 flex items-center justify-between gap-4"
               style={{ borderTop: '1px solid var(--color-border-soft)', paddingTop: '12px' }}>
            <span className="font-mono text-[11px]" style={{ color: 'var(--color-ink-3)' }}>
              macro&nbsp;f1
            </span>
            <span className="font-mono text-[12px] tabular" style={{ color: 'var(--color-ink-2)' }}>
              {m.macro_f1 != null ? (m.macro_f1 * 100).toFixed(2) + '%' : '—'}
            </span>
          </div>

          <div className="mt-4">
            <p className="mb-2 font-mono text-[10px] uppercase tracking-wider"
               style={{ color: 'var(--color-ink-3)' }}>
              classes the model can predict
            </p>
            <div className="flex flex-wrap items-center gap-x-3 gap-y-2 font-mono text-[10px] tabular"
                 style={{ color: 'var(--color-ink-3)' }}>
              <GrowthSpine active={null} size="sm" />
              <span>{(m.classes ?? CLASSES).join(' · ')}</span>
            </div>
          </div>
        </Card>
      ))}
    </div>
  )
}
