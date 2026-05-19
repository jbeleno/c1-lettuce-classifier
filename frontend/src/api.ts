/** Typed fetch wrappers for the four FastAPI endpoints. Vite proxies /api/*
 *  to http://127.0.0.1:8000/* in dev so the browser never sees CORS. */
import type {
  ModelInfo,
  PredictionResponse,
  PredictionRow,
  MetricsResponse,
} from './types'

const BASE = '/api'

class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body?.detail ?? detail
    } catch {
      /* ignore — non-JSON error body */
    }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

export async function healthz(): Promise<{ status: string }> {
  return json(await fetch(`${BASE}/healthz`))
}

export async function listModels(): Promise<ModelInfo[]> {
  return json(await fetch(`${BASE}/models`))
}

export async function getMetrics(): Promise<MetricsResponse> {
  return json(await fetch(`${BASE}/metrics`))
}

export interface HistoryQuery {
  limit?: number
  label?: string
  model?: string
}
export async function getHistory(q: HistoryQuery = {}): Promise<PredictionRow[]> {
  const params = new URLSearchParams()
  if (q.limit) params.set('limit', String(q.limit))
  if (q.label) params.set('label', q.label)
  if (q.model) params.set('model', q.model)
  const qs = params.toString()
  return json(await fetch(`${BASE}/history${qs ? `?${qs}` : ''}`))
}

export async function predict(
  file: Blob,
  model?: string,
  filename = 'capture.jpg',
): Promise<PredictionResponse> {
  const fd = new FormData()
  fd.append('file', file, filename)
  if (model) fd.append('model', model)
  return json(await fetch(`${BASE}/predict`, { method: 'POST', body: fd }))
}

export { ApiError }
