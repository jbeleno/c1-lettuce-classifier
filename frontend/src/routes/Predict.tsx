import { useEffect, useState } from 'react'
import { Loader2 } from 'lucide-react'
import { ApiError, predict } from '../api'
import type { PredictionResponse } from '../types'
import { CLASS_META } from '../types'
import { Card } from '../components/Card'
import { ClassChip } from '../components/ClassChip'
import { GrowthSpine } from '../components/GrowthSpine'
import { ProbabilityBars } from '../components/ProbabilityBars'
import { ViewfinderUpload } from '../components/ViewfinderUpload'
import { WebcamCapture } from '../components/WebcamCapture'
import { ModelPicker } from '../components/ModelPicker'
import { CropTool } from '../components/CropTool'

type Source = 'upload' | 'webcam'

/** Three-step flow:
 *    select  →  framed (crop)  →  predicted
 *  The classifier was trained on single-pod crops, so the user is asked to
 *  frame one pod before we send anything to the backend. */
type Stage =
  | { kind: 'select' }
  | { kind: 'framing'; blob: Blob; filename?: string }

export function Predict() {
  const [source, setSource] = useState<Source>('upload')
  const [model, setModel] = useState<string>('')
  const [stage, setStage] = useState<Stage>({ kind: 'select' })
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [resultPreview, setResultPreview] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [latencyMs, setLatencyMs] = useState<number | null>(null)

  useEffect(
    () => () => { if (resultPreview) URL.revokeObjectURL(resultPreview) },
    [resultPreview],
  )

  const startFraming = (blob: Blob, filename?: string) => {
    setError(null)
    setStage({ kind: 'framing', blob, filename })
  }

  const submit = async (cropped: Blob) => {
    if (resultPreview) URL.revokeObjectURL(resultPreview)
    setResultPreview(URL.createObjectURL(cropped))
    setError(null)
    setPending(true)
    const t0 = performance.now()
    try {
      const filename =
        stage.kind === 'framing' && stage.filename
          ? `crop-${stage.filename}`
          : 'crop.jpg'
      const res = await predict(cropped, model || undefined, filename)
      setResult(res)
      setLatencyMs(performance.now() - t0)
      setStage({ kind: 'select' })
    } catch (e) {
      setResult(null)
      setLatencyMs(null)
      setError(
        e instanceof ApiError
          ? `${e.status} · ${e.message}`
          : e instanceof Error ? e.message : 'unknown error',
      )
    } finally {
      setPending(false)
    }
  }

  return (
    <div className="grid gap-6 md:grid-cols-[5fr_7fr]">
      <Card className="p-4 md:p-5">
        <div className="mb-4 flex items-center justify-between gap-3">
          <SourceToggle
            value={source}
            onChange={(s) => { setSource(s); setStage({ kind: 'select' }) }}
            disabled={stage.kind === 'framing'}
          />
          <ModelPicker value={model || null} onChange={setModel} />
        </div>

        {stage.kind === 'framing' ? (
          <CropTool
            blob={stage.blob}
            onConfirm={submit}
            onCancel={() => setStage({ kind: 'select' })}
          />
        ) : source === 'upload' ? (
          <ViewfinderUpload
            onPick={(f) => startFraming(f, f.name)}
            busy={pending}
            preview={resultPreview}
          />
        ) : (
          <WebcamCapture onCapture={(b) => startFraming(b)} />
        )}

        {stage.kind === 'select' && (
          <p className="mt-3 font-mono text-[11px]"
             style={{ color: 'var(--color-ink-mute)' }}>
            jpg · png · webp · bmp · tiff · gif (no heic / avif).
            after picking, you frame ONE pod before predicting — the model
            classifies a single pod per request.
          </p>
        )}
      </Card>

      <ResultPane
        pending={pending}
        result={result}
        error={error}
        latencyMs={latencyMs}
      />
    </div>
  )
}

function SourceToggle({
  value,
  onChange,
  disabled = false,
}: { value: Source; onChange: (s: Source) => void; disabled?: boolean }) {
  return (
    <div role="tablist" aria-label="Image source"
         className="inline-flex rounded border overflow-hidden"
         style={{
           borderColor: 'var(--color-border)',
           opacity: disabled ? 0.5 : 1,
         }}>
      {(['upload', 'webcam'] as const).map((s) => {
        const active = value === s
        return (
          <button
            key={s}
            role="tab"
            aria-selected={active}
            type="button"
            disabled={disabled}
            onClick={() => onChange(s)}
            className="px-3 py-1 font-mono text-[12px] transition-colors duration-150 cursor-pointer disabled:cursor-not-allowed"
            style={{
              background: active ? 'var(--color-surface-2)' : 'transparent',
              color: active ? 'var(--color-ink)' : 'var(--color-ink-3)',
            }}
          >
            {s}
          </button>
        )
      })}
    </div>
  )
}

function ResultPane({
  pending,
  result,
  error,
  latencyMs,
}: {
  pending: boolean
  result: PredictionResponse | null
  error: string | null
  latencyMs: number | null
}) {
  const accent = result ? CLASS_META[result.label].cssVar : undefined

  return (
    <Card className="p-5 md:p-6" accent={accent}>
      <div className="flex items-center justify-between">
        <h2 className="font-mono text-[12px] uppercase tracking-wider"
            style={{ color: 'var(--color-ink-3)' }}>
          prediction
        </h2>
        {latencyMs !== null && (
          <span className="font-mono text-[11px] tabular"
                style={{ color: 'var(--color-ink-mute)' }}>
            {Math.round(latencyMs)} ms
          </span>
        )}
      </div>

      <div
        className="mt-4 min-h-[88px] flex items-start gap-5"
        aria-live="polite"
        aria-busy={pending}
      >
        {pending && !result ? (
          <SkeletonResult />
        ) : result ? (
          <>
            <div className="flex flex-col gap-3">
              <ClassChip label={result.label} size="lg" />
              <p className="font-mono text-[42px] leading-none tabular"
                 style={{ color: 'var(--color-ink)' }}>
                {(result.confidence * 100).toFixed(2)}
                <span className="text-[20px] text-[var(--color-ink-3)]">%</span>
              </p>
              <p className="font-mono text-[11px]" style={{ color: 'var(--color-ink-3)' }}>
                via <span style={{ color: 'var(--color-ink-2)' }}>{result.model}</span>
                {' · '}{CLASS_META[result.label].verb}
              </p>
            </div>
            <div className="flex-1 pt-2">
              <GrowthSpine active={result.label} size="lg" showLabel />
            </div>
          </>
        ) : error ? (
          <p role="alert" className="font-mono text-[12px]"
             style={{ color: 'var(--color-class-empty)' }}>
            {error}
          </p>
        ) : (
          <p className="font-mono text-[12px]" style={{ color: 'var(--color-ink-mute)' }}>
            waiting for an image…
          </p>
        )}
      </div>

      {result && (
        <div className="mt-6 border-t pt-5" style={{ borderColor: 'var(--color-border-soft)' }}>
          <h3 className="mb-3 font-mono text-[11px] uppercase tracking-wider"
              style={{ color: 'var(--color-ink-3)' }}>
            class probabilities
          </h3>
          <ProbabilityBars probabilities={result.probabilities} predicted={result.label} />
        </div>
      )}
    </Card>
  )
}

function SkeletonResult() {
  return (
    <div className="flex w-full items-center gap-4">
      <Loader2 size={28} className="animate-spin" color="var(--color-ink-3)" />
      <span className="font-mono text-[12px]" style={{ color: 'var(--color-ink-3)' }}>
        running model on the image…
      </span>
    </div>
  )
}
