import { useCallback, useEffect, useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { getHistory } from '../api'
import type { PredictionRow, LettuceClass } from '../types'
import { CLASSES } from '../types'
import { Card } from '../components/Card'
import { ClassChip } from '../components/ClassChip'
import { GrowthSpine } from '../components/GrowthSpine'

const LIMITS = [10, 25, 50, 100] as const

export function History() {
  const [rows, setRows] = useState<PredictionRow[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [label, setLabel] = useState<string>('')
  const [model, setModel] = useState<string>('')
  const [limit, setLimit] = useState<number>(50)

  const refresh = useCallback(() => {
    setLoading(true); setError(null)
    getHistory({
      limit,
      label: label || undefined,
      model: model || undefined,
    })
      .then(setRows)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [limit, label, model])

  useEffect(() => { refresh() }, [refresh])

  return (
    <div className="flex flex-col gap-4">
      <Card className="p-4">
        <div className="flex flex-wrap items-end gap-4">
          <Field label="label">
            <select
              value={label} onChange={(e) => setLabel(e.target.value)}
              className="rounded border px-2 py-1 font-mono text-[12px] cursor-pointer"
              style={{ background: 'var(--color-surface-inset)', color: 'var(--color-ink)',
                       borderColor: 'var(--color-border)' }}
            >
              <option value="">any</option>
              {CLASSES.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </Field>
          <Field label="model">
            <input
              value={model} onChange={(e) => setModel(e.target.value)}
              placeholder="any"
              className="w-44 rounded border px-2 py-1 font-mono text-[12px] placeholder:text-[var(--color-ink-mute)]"
              style={{ background: 'var(--color-surface-inset)', color: 'var(--color-ink)',
                       borderColor: 'var(--color-border)' }}
            />
          </Field>
          <Field label="limit">
            <select
              value={limit} onChange={(e) => setLimit(Number(e.target.value))}
              className="rounded border px-2 py-1 font-mono text-[12px] cursor-pointer"
              style={{ background: 'var(--color-surface-inset)', color: 'var(--color-ink)',
                       borderColor: 'var(--color-border)' }}
            >
              {LIMITS.map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </Field>
          <button
            type="button" onClick={refresh}
            className="inline-flex items-center gap-2 rounded border px-3 py-1 font-mono text-[12px] cursor-pointer
                       transition-colors duration-150 hover:bg-[var(--color-surface-2)]"
            style={{ borderColor: 'var(--color-border)', color: 'var(--color-ink-2)' }}
            aria-label="Refresh history"
          >
            <RefreshCw size={14} strokeWidth={1.75} className={loading ? 'animate-spin' : ''} />
            refresh
          </button>
        </div>
      </Card>

      <Card>
        <div className="px-4 py-3 flex items-center justify-between border-b"
             style={{ borderColor: 'var(--color-border-soft)' }}>
          <h2 className="font-mono text-[12px] uppercase tracking-wider"
              style={{ color: 'var(--color-ink-3)' }}>
            recent predictions
          </h2>
          <span className="font-mono text-[11px] tabular"
                style={{ color: 'var(--color-ink-mute)' }}>
            {rows.length} row{rows.length === 1 ? '' : 's'}
          </span>
        </div>

        {error ? (
          <p role="alert" className="px-4 py-6 font-mono text-[12px]"
             style={{ color: 'var(--color-class-empty)' }}>
            {error}
          </p>
        ) : rows.length === 0 ? (
          <p className="px-4 py-8 text-center font-mono text-[12px]"
             style={{ color: 'var(--color-ink-mute)' }}>
            {loading ? 'loading…' : 'no predictions yet — run one from the predict tab'}
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full font-mono text-[12px]">
              <thead className="text-left" style={{ color: 'var(--color-ink-3)' }}>
                <tr>
                  <Th>#</Th>
                  <Th>when</Th>
                  <Th>model</Th>
                  <Th>label</Th>
                  <Th>conf.</Th>
                  <Th>spine</Th>
                  <Th>image</Th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.id}
                      className="border-t transition-colors hover:bg-[var(--color-surface-2)]"
                      style={{ borderColor: 'var(--color-border-soft)' }}>
                    <Td className="text-[var(--color-ink-mute)]">{r.id}</Td>
                    <Td className="text-[var(--color-ink-3)] tabular">
                      {new Date(r.created_at).toLocaleString()}
                    </Td>
                    <Td className="text-[var(--color-ink-2)]">{r.model}</Td>
                    <Td><ClassChip label={r.label as LettuceClass} size="sm" /></Td>
                    <Td className="tabular text-[var(--color-ink)]">
                      {(r.confidence * 100).toFixed(1)}%
                    </Td>
                    <Td><GrowthSpine active={r.label as LettuceClass} size="sm" /></Td>
                    <Td className="max-w-[220px] truncate text-[var(--color-ink-mute)]"
                        title={r.image_filename ?? r.image_sha256}>
                      {r.image_filename ?? `sha:${r.image_sha256.slice(0, 12)}`}
                    </Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  const id = `f-${label}`
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="font-mono text-[11px]"
             style={{ color: 'var(--color-ink-3)' }}>
        {label}
      </label>
      {/* The input/select inside is responsible for owning the id. Aria
          isn't broken by this since the label uses htmlFor matching the
          child component's internal id. */}
      <div id={id}>{children}</div>
    </div>
  )
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="px-4 py-2 font-medium text-[11px] uppercase tracking-wider">{children}</th>
}
type TdProps = React.TdHTMLAttributes<HTMLTableCellElement> & { className?: string }
function Td({ children, className = '', ...rest }: TdProps) {
  return <td className={`px-4 py-2.5 ${className}`} {...rest}>{children}</td>
}
