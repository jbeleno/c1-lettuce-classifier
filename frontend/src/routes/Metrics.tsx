import { useEffect, useState } from 'react'
import { getMetrics } from '../api'
import type { MetricsRow } from '../types'
import { Card } from '../components/Card'
import { GrowthSpine } from '../components/GrowthSpine'
import { CLASSES, type LettuceClass } from '../types'

export function Metrics() {
  const [rows, setRows] = useState<MetricsRow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getMetrics()
      .then((m) => setRows(m.rows))
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

  return (
    <div className="flex flex-col gap-6">
      <Card>
        <div className="px-4 py-3 flex items-center justify-between border-b"
             style={{ borderColor: 'var(--color-border-soft)' }}>
          <h2 className="font-mono text-[12px] uppercase tracking-wider"
              style={{ color: 'var(--color-ink-3)' }}>
            model comparison
          </h2>
          <span className="font-mono text-[11px] tabular"
                style={{ color: 'var(--color-ink-mute)' }}>
            held-out test set
          </span>
        </div>
        {loading ? (
          <p className="px-4 py-8 text-center font-mono text-[12px]"
             style={{ color: 'var(--color-ink-mute)' }}>
            loading…
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full font-mono text-[12px]">
              <thead className="text-left" style={{ color: 'var(--color-ink-3)' }}>
                <tr>
                  <Th>model</Th>
                  <Th className="text-right">accuracy</Th>
                  <Th className="text-right">macro f1</Th>
                  <Th className="text-right">macro p</Th>
                  <Th className="text-right">macro r</Th>
                  <Th className="text-right">best val</Th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const acc = r.test_accuracy ?? 0
                  return (
                    <tr key={r.model}
                        className="border-t hover:bg-[var(--color-surface-2)] transition-colors"
                        style={{ borderColor: 'var(--color-border-soft)' }}>
                      <Td className="text-[var(--color-ink-2)]">{r.model}</Td>
                      <Td className="tabular text-right">
                        <BarCell value={acc} />
                      </Td>
                      <Td className="tabular text-right text-[var(--color-ink-2)]">
                        {pct(r.macro_f1)}
                      </Td>
                      <Td className="tabular text-right text-[var(--color-ink-3)]">
                        {pct(r.macro_precision)}
                      </Td>
                      <Td className="tabular text-right text-[var(--color-ink-3)]">
                        {pct(r.macro_recall)}
                      </Td>
                      <Td className="tabular text-right text-[var(--color-ink-3)]">
                        {pct(r.best_val_accuracy)}
                      </Td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Card className="p-5">
        <h2 className="mb-4 font-mono text-[12px] uppercase tracking-wider"
            style={{ color: 'var(--color-ink-3)' }}>
          growth-stage axis (used as the row and column labels of the per-model
          confusion matrices saved to <span className="text-[var(--color-ink-2)]">models_saved/&lt;name&gt;/confusion_matrix.png</span>)
        </h2>
        <div className="grid grid-cols-[1fr_auto] gap-x-6 gap-y-2 items-center">
          {CLASSES.map((c) => (
            <SpineRow key={c} cls={c} />
          ))}
        </div>
      </Card>
    </div>
  )
}

function SpineRow({ cls }: { cls: LettuceClass }) {
  return (
    <>
      <GrowthSpine active={cls} size="axis" />
      <span className="font-mono text-[12px] tabular"
            style={{ color: 'var(--color-ink-2)' }}>{cls}</span>
    </>
  )
}

function BarCell({ value }: { value: number }) {
  return (
    <span className="inline-flex items-center justify-end gap-3">
      <span
        className="relative inline-block h-1 w-28 rounded-full overflow-hidden"
        style={{ background: 'var(--color-surface-inset)' }}
      >
        <span
          className="absolute inset-y-0 left-0"
          style={{
            width: `${Math.max(value * 100, 0.5)}%`,
            background: 'var(--color-class-ready)',
          }}
        />
      </span>
      <span style={{ color: 'var(--color-ink)' }}>{(value * 100).toFixed(2)}%</span>
    </span>
  )
}

function pct(v: number | null): string {
  return v == null ? '—' : `${(v * 100).toFixed(2)}%`
}

function Th({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return <th className={`px-4 py-2 font-medium text-[11px] uppercase tracking-wider ${className}`}>{children}</th>
}
function Td({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return <td className={`px-4 py-2.5 ${className}`}>{children}</td>
}
