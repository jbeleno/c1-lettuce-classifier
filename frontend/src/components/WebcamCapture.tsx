/** WebcamCapture — uses the native getUserMedia API. No library.
 *  Streams the laptop camera into a <video>, takes a frame on click, and
 *  returns a JPEG Blob to the parent. Cleans up the stream on unmount. */
import { useCallback, useEffect, useRef, useState } from 'react'
import { Camera, CircleStop, RefreshCw } from 'lucide-react'

interface Props {
  onCapture: (blob: Blob) => void
}

export function WebcamCapture({ onCapture }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [active, setActive] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setActive(false)
  }, [])

  const start = useCallback(async () => {
    setError(null)
    setBusy(true)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setActive(true)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Could not access the camera. Check the browser permission.',
      )
    } finally {
      setBusy(false)
    }
  }, [])

  const capture = useCallback(() => {
    const video = videoRef.current
    if (!video) return
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    canvas.toBlob(
      (blob) => {
        if (blob) onCapture(blob)
      },
      'image/jpeg',
      0.92,
    )
  }, [onCapture])

  useEffect(() => stop, [stop])

  return (
    <div className="flex flex-col gap-3">
      <div
        className="relative aspect-[4/3] w-full overflow-hidden rounded"
        style={{ background: 'var(--color-surface-inset)' }}
      >
        <video
          ref={videoRef}
          playsInline
          muted
          aria-label="Live webcam preview"
          className={`size-full object-cover ${active ? '' : 'opacity-0'}`}
        />
        {!active && (
          <div className="absolute inset-0 grid place-items-center">
            <p className="font-mono text-[12px] text-[var(--color-ink-3)]">
              webcam idle — click <span className="text-[var(--color-ring-focus)]">Start</span>
            </p>
          </div>
        )}
        <Brackets />
      </div>

      <div className="flex flex-wrap gap-2">
        {!active ? (
          <button
            type="button"
            onClick={start}
            disabled={busy}
            className="inline-flex items-center gap-2 rounded border px-3 py-1.5 font-mono text-[12px]
                       transition-colors duration-150 hover:bg-[var(--color-surface-2)] disabled:opacity-50 cursor-pointer"
            style={{ borderColor: 'var(--color-border)', color: 'var(--color-ink)' }}
          >
            <Camera size={14} strokeWidth={1.75} />
            {busy ? 'requesting…' : 'start webcam'}
          </button>
        ) : (
          <>
            <button
              type="button"
              onClick={capture}
              className="inline-flex items-center gap-2 rounded px-3 py-1.5 font-mono text-[12px] cursor-pointer
                         transition-colors duration-150"
              style={{
                background: 'color-mix(in oklab, var(--color-class-ready) 22%, transparent)',
                color: 'var(--color-class-ready)',
                border: '1px solid color-mix(in oklab, var(--color-class-ready) 45%, transparent)',
              }}
            >
              <Camera size={14} strokeWidth={1.75} />
              capture frame
            </button>
            <button
              type="button"
              onClick={stop}
              className="inline-flex items-center gap-2 rounded border px-3 py-1.5 font-mono text-[12px]
                         transition-colors duration-150 hover:bg-[var(--color-surface-2)] cursor-pointer"
              style={{ borderColor: 'var(--color-border)', color: 'var(--color-ink-2)' }}
            >
              <CircleStop size={14} strokeWidth={1.75} />
              stop
            </button>
            <button
              type="button"
              onClick={() => { stop(); start() }}
              className="inline-flex items-center gap-2 rounded border px-3 py-1.5 font-mono text-[12px]
                         transition-colors duration-150 hover:bg-[var(--color-surface-2)] cursor-pointer"
              style={{ borderColor: 'var(--color-border)', color: 'var(--color-ink-2)' }}
              aria-label="Restart camera"
            >
              <RefreshCw size={14} strokeWidth={1.75} />
              restart
            </button>
          </>
        )}
      </div>

      {error && (
        <p
          role="alert"
          className="font-mono text-[11px]"
          style={{ color: 'var(--color-class-empty)' }}
        >
          {error}
        </p>
      )}
    </div>
  )
}

function Brackets() {
  const map = {
    tl: 'top-2 left-2 border-t-2 border-l-2',
    tr: 'top-2 right-2 border-t-2 border-r-2',
    bl: 'bottom-2 left-2 border-b-2 border-l-2',
    br: 'bottom-2 right-2 border-b-2 border-r-2',
  } as const
  return (
    <>
      {(['tl', 'tr', 'bl', 'br'] as const).map((p) => (
        <span
          key={p}
          aria-hidden
          className={`pointer-events-none absolute size-5 ${map[p]}`}
          style={{ borderColor: 'var(--color-ring-focus)' }}
        />
      ))}
    </>
  )
}
