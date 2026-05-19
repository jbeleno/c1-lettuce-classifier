import { useEffect, useState } from 'react'
import type { ModelInfo } from '../types'
import { listModels } from '../api'

interface Props {
  value: string | null
  onChange: (model: string) => void
  /** When true, also expose a "(default)" sentinel that means "let the
   *  server pick the best available model". */
  allowDefault?: boolean
}

/** Custom select — native <select> can't be themed reliably across browsers. */
export function ModelPicker({ value, onChange, allowDefault = true }: Props) {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let abort = false
    setLoading(true)
    listModels()
      .then((m) => { if (!abort) setModels(m) })
      .catch((e: Error) => { if (!abort) setError(e.message) })
      .finally(() => { if (!abort) setLoading(false) })
    return () => { abort = true }
  }, [])

  return (
    <div className="flex items-center gap-2">
      <label htmlFor="model-picker" className="font-mono text-[11px]"
             style={{ color: 'var(--color-ink-3)' }}>
        model
      </label>
      <select
        id="model-picker"
        value={value ?? '__default__'}
        onChange={(e) => onChange(e.target.value === '__default__' ? '' : e.target.value)}
        disabled={loading || !!error}
        className="rounded border px-2 py-1 font-mono text-[12px] cursor-pointer disabled:opacity-50"
        style={{
          background: 'var(--color-surface-inset)',
          color: 'var(--color-ink)',
          borderColor: 'var(--color-border)',
        }}
      >
        {allowDefault && <option value="__default__">default (server picks)</option>}
        {models.map((m) => (
          <option key={m.name} value={m.name}>
            {m.name}
            {m.test_accuracy != null && `  ·  ${(m.test_accuracy * 100).toFixed(1)}%`}
          </option>
        ))}
      </select>
      {error && (
        <span role="alert" className="font-mono text-[11px]"
              style={{ color: 'var(--color-class-empty)' }}>
          {error}
        </span>
      )}
    </div>
  )
}
