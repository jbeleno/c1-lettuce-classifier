/** Server-side label set. Order MATTERS — it mirrors the model's softmax
 *  index ordering and the spine reads left → right as time progresses. */
export const CLASSES = [
  'empty_pod',
  'germination',
  'young',
  'pod',
  'Ready',
] as const

export type LettuceClass = (typeof CLASSES)[number]

/** Mirrors backend/schemas.py:PredictionResponse */
export interface PredictionResponse {
  label: LettuceClass
  confidence: number
  probabilities: Record<LettuceClass, number>
  model: string
}

/** Mirrors backend/schemas.py:PredictionRow */
export interface PredictionRow extends PredictionResponse {
  id: number
  created_at: string
  image_sha256: string
  image_filename: string | null
  image_width: number
  image_height: number
}

export interface ModelInfo {
  name: string
  framework: 'tf' | 'torch' | 'ensemble'
  test_accuracy: number | null
  macro_f1: number | null
  classes: LettuceClass[] | null
}

export interface MetricsRow {
  model: string
  test_accuracy: number | null
  macro_f1: number | null
  macro_precision: number | null
  macro_recall: number | null
  best_val_accuracy: number | null
}

export interface MetricsResponse {
  rows: MetricsRow[]
}

/** Class-to-token mapping used by ClassChip, ProbabilityBars and the spine.
 *  Kept here (not in CSS) because the React renderers need both the hex
 *  value AND a human-readable verb for screen readers. */
export const CLASS_META: Record<
  LettuceClass,
  { token: string; verb: string; cssVar: string }
> = {
  empty_pod:   { token: 'empty',   verb: 'nothing growing',         cssVar: 'var(--color-class-empty)' },
  germination: { token: 'germ',    verb: 'first cotyledons',        cssVar: 'var(--color-class-germ)'  },
  young:       { token: 'young',   verb: 'true leaves emerging',    cssVar: 'var(--color-class-young)' },
  pod:         { token: 'pod',     verb: 'full vegetative growth',  cssVar: 'var(--color-class-pod)'   },
  Ready:       { token: 'ready',   verb: 'ready for harvest',       cssVar: 'var(--color-class-ready)' },
}
