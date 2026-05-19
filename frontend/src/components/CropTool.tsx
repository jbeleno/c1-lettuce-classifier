/** CropTool — between picking an image and sending it. The user frames a
 *  single pod by dragging the crop rectangle or its four corner handles.
 *  The dim overlay outside the crop reinforces "this is what the classifier
 *  will see", and the four corner brackets keep the viewfinder signature.
 *
 *  All coordinates are stored in the IMAGE's own pixel space (not display
 *  pixels) so resizing the browser doesn't lose framing. */
import { useCallback, useEffect, useRef, useState } from 'react'
import { Check, X } from 'lucide-react'

interface Props {
  blob: Blob
  onConfirm: (croppedJpeg: Blob) => void
  onCancel: () => void
}

interface Crop {
  x: number
  y: number
  w: number
  h: number
}

type Mode =
  | { kind: 'move' }
  | { kind: 'resize'; corner: 'nw' | 'ne' | 'sw' | 'se' }

interface DragState {
  mode: Mode
  pointerId: number
  startClientX: number
  startClientY: number
  startCrop: Crop
}

const MIN_SIZE_PX = 32

export function CropTool({ blob, onConfirm, onCancel }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const [src, setSrc] = useState<string>('')
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null)
  const [crop, setCrop] = useState<Crop | null>(null)
  const [drag, setDrag] = useState<DragState | null>(null)
  const [working, setWorking] = useState(false)

  // Hold the URL while we use it, revoke on unmount
  useEffect(() => {
    const url = URL.createObjectURL(blob)
    setSrc(url)
    return () => URL.revokeObjectURL(url)
  }, [blob])

  const onImgLoad = () => {
    const img = imgRef.current
    if (!img) return
    const w = img.naturalWidth
    const h = img.naturalHeight
    setNatural({ w, h })
    // Default: centered square at 70% of the shorter side — sized so the
    // classifier sees something close to its 224 x 224 training crops.
    const side = Math.floor(Math.min(w, h) * 0.7)
    setCrop({
      x: Math.floor((w - side) / 2),
      y: Math.floor((h - side) / 2),
      w: side,
      h: side,
    })
  }

  const pxPerImg = useCallback((): number => {
    const img = imgRef.current
    if (!img || !natural) return 1
    return img.clientWidth / natural.w
  }, [natural])

  // ── Pointer handlers ────────────────────────────────────────────────────
  const onPointerDown = (
    e: React.PointerEvent<HTMLDivElement>,
    mode: Mode,
  ) => {
    if (!crop) return
    e.preventDefault()
    e.stopPropagation()
    containerRef.current?.setPointerCapture(e.pointerId)
    setDrag({
      mode,
      pointerId: e.pointerId,
      startClientX: e.clientX,
      startClientY: e.clientY,
      startCrop: { ...crop },
    })
  }

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!drag || !natural || !crop) return
    const scale = pxPerImg()
    const dx = (e.clientX - drag.startClientX) / scale
    const dy = (e.clientY - drag.startClientY) / scale
    const next = applyDelta(drag.startCrop, drag.mode, dx, dy, natural)
    setCrop(next)
  }

  const onPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (drag) {
      containerRef.current?.releasePointerCapture(drag.pointerId)
      setDrag(null)
    } else {
      e.stopPropagation()
    }
  }

  // ── Confirm: rasterize the crop to a JPEG blob ──────────────────────────
  const confirm = useCallback(async () => {
    if (!crop || !imgRef.current) return
    setWorking(true)
    try {
      const canvas = document.createElement('canvas')
      canvas.width = crop.w
      canvas.height = crop.h
      const ctx = canvas.getContext('2d')
      if (!ctx) throw new Error('canvas 2d context unavailable')
      ctx.drawImage(
        imgRef.current,
        crop.x, crop.y, crop.w, crop.h,
        0, 0, crop.w, crop.h,
      )
      const out = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, 'image/jpeg', 0.92),
      )
      if (!out) throw new Error('canvas.toBlob returned null')
      onConfirm(out)
    } finally {
      setWorking(false)
    }
  }, [crop, onConfirm])

  // ── Render ──────────────────────────────────────────────────────────────
  const ready = !!(natural && crop)

  return (
    <div className="flex flex-col gap-3">
      <div
        ref={containerRef}
        className="relative w-full select-none"
        style={{ background: 'var(--color-surface-inset)', touchAction: 'none' }}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <img
          ref={imgRef}
          src={src}
          alt="Image being framed for prediction"
          onLoad={onImgLoad}
          className="block w-full"
          draggable={false}
        />

        {ready && crop && (
          <CropOverlay
            crop={crop}
            scale={pxPerImg()}
            onMoveStart={(e) => onPointerDown(e, { kind: 'move' })}
            onResizeStart={(corner, e) =>
              onPointerDown(e, { kind: 'resize', corner })
            }
          />
        )}
      </div>

      <p className="font-mono text-[11px]" style={{ color: 'var(--color-ink-3)' }}>
        frame ONE pod — the classifier was trained on single-pod crops, so
        cropping tight gives the best confidence
      </p>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={confirm}
          disabled={!ready || working}
          className="inline-flex items-center gap-2 rounded px-3 py-1.5 font-mono text-[12px] cursor-pointer
                     transition-colors duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
          style={{
            background: 'color-mix(in oklab, var(--color-class-ready) 22%, transparent)',
            color: 'var(--color-class-ready)',
            border: '1px solid color-mix(in oklab, var(--color-class-ready) 45%, transparent)',
          }}
        >
          <Check size={14} strokeWidth={1.75} />
          {working ? 'cropping…' : 'predict this crop'}
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="inline-flex items-center gap-2 rounded border px-3 py-1.5 font-mono text-[12px]
                     transition-colors duration-150 hover:bg-[var(--color-surface-2)] cursor-pointer"
          style={{ borderColor: 'var(--color-border)', color: 'var(--color-ink-2)' }}
        >
          <X size={14} strokeWidth={1.75} />
          pick another image
        </button>
        {ready && crop && (
          <span
            className="ml-auto self-center font-mono text-[11px] tabular"
            style={{ color: 'var(--color-ink-mute)' }}
          >
            crop: {crop.w} × {crop.h} px
          </span>
        )}
      </div>
    </div>
  )
}

/** Visual layer over the image: 4 dim panels outside the crop, the crop
 *  rectangle itself (transparent so the image shows through clearly), four
 *  corner brackets and four resize handles. */
function CropOverlay({
  crop,
  scale,
  onMoveStart,
  onResizeStart,
}: {
  crop: Crop
  scale: number
  onMoveStart: (e: React.PointerEvent<HTMLDivElement>) => void
  onResizeStart: (
    corner: 'nw' | 'ne' | 'sw' | 'se',
    e: React.PointerEvent<HTMLDivElement>,
  ) => void
}) {
  const px = (n: number) => `${n * scale}px`
  const dim = {
    position: 'absolute' as const,
    background: 'rgb(15 23 42 / 0.55)', // --color-bg at 55% alpha
    pointerEvents: 'none' as const,
  }
  return (
    <>
      {/* Four dim panels around the crop */}
      <div style={{ ...dim, top: 0, left: 0, right: 0, height: px(crop.y) }} />
      <div
        style={{
          ...dim,
          top: px(crop.y),
          left: 0,
          width: px(crop.x),
          height: px(crop.h),
        }}
      />
      <div
        style={{
          ...dim,
          top: px(crop.y),
          left: px(crop.x + crop.w),
          right: 0,
          height: px(crop.h),
        }}
      />
      <div
        style={{
          ...dim,
          top: px(crop.y + crop.h),
          left: 0,
          right: 0,
          bottom: 0,
        }}
      />

      {/* The crop body — draggable, no fill */}
      <div
        role="button"
        aria-label="Drag to move the crop region"
        tabIndex={0}
        onPointerDown={onMoveStart}
        style={{
          position: 'absolute',
          left: px(crop.x),
          top: px(crop.y),
          width: px(crop.w),
          height: px(crop.h),
          cursor: 'move',
          outline: '1px solid color-mix(in oklab, var(--color-ring-focus) 60%, transparent)',
        }}
      >
        {/* Four corner brackets — signature continued from ViewfinderUpload */}
        <Bracket position="tl" />
        <Bracket position="tr" />
        <Bracket position="bl" />
        <Bracket position="br" />

        {/* Four resize handles — slightly bigger hit area than they look */}
        {(['nw', 'ne', 'sw', 'se'] as const).map((c) => (
          <Handle
            key={c}
            corner={c}
            onPointerDown={(e) => onResizeStart(c, e)}
          />
        ))}
      </div>
    </>
  )
}

function Bracket({ position }: { position: 'tl' | 'tr' | 'bl' | 'br' }) {
  const map = {
    tl: { top: -1, left: -1, borderTop: '2px solid', borderLeft: '2px solid' },
    tr: { top: -1, right: -1, borderTop: '2px solid', borderRight: '2px solid' },
    bl: { bottom: -1, left: -1, borderBottom: '2px solid', borderLeft: '2px solid' },
    br: { bottom: -1, right: -1, borderBottom: '2px solid', borderRight: '2px solid' },
  } as const
  return (
    <span
      aria-hidden
      className="pointer-events-none absolute"
      style={{
        ...map[position],
        width: 18,
        height: 18,
        borderColor: 'var(--color-ring-focus)',
      }}
    />
  )
}

function Handle({
  corner,
  onPointerDown,
}: {
  corner: 'nw' | 'ne' | 'sw' | 'se'
  onPointerDown: (e: React.PointerEvent<HTMLDivElement>) => void
}) {
  const placement = {
    nw: { top: -6, left: -6, cursor: 'nwse-resize' },
    ne: { top: -6, right: -6, cursor: 'nesw-resize' },
    sw: { bottom: -6, left: -6, cursor: 'nesw-resize' },
    se: { bottom: -6, right: -6, cursor: 'nwse-resize' },
  } as const
  return (
    <div
      role="button"
      aria-label={`Drag ${corner} corner to resize the crop`}
      tabIndex={0}
      onPointerDown={onPointerDown}
      style={{
        position: 'absolute',
        ...placement[corner],
        width: 14,
        height: 14,
        background: 'var(--color-ring-focus)',
        borderRadius: 3,
      }}
    />
  )
}

// ── Pure logic ─────────────────────────────────────────────────────────────

function applyDelta(
  start: Crop,
  mode: Mode,
  dx: number,
  dy: number,
  bounds: { w: number; h: number },
): Crop {
  if (mode.kind === 'move') {
    return {
      x: clamp(Math.round(start.x + dx), 0, bounds.w - start.w),
      y: clamp(Math.round(start.y + dy), 0, bounds.h - start.h),
      w: start.w,
      h: start.h,
    }
  }
  const { corner } = mode
  let { x, y, w, h } = start
  switch (corner) {
    case 'nw':
      x = clamp(Math.round(start.x + dx), 0, start.x + start.w - MIN_SIZE_PX)
      y = clamp(Math.round(start.y + dy), 0, start.y + start.h - MIN_SIZE_PX)
      w = start.w + (start.x - x)
      h = start.h + (start.y - y)
      break
    case 'ne':
      y = clamp(Math.round(start.y + dy), 0, start.y + start.h - MIN_SIZE_PX)
      w = clamp(Math.round(start.w + dx), MIN_SIZE_PX, bounds.w - start.x)
      h = start.h + (start.y - y)
      break
    case 'sw':
      x = clamp(Math.round(start.x + dx), 0, start.x + start.w - MIN_SIZE_PX)
      w = start.w + (start.x - x)
      h = clamp(Math.round(start.h + dy), MIN_SIZE_PX, bounds.h - start.y)
      break
    case 'se':
      w = clamp(Math.round(start.w + dx), MIN_SIZE_PX, bounds.w - start.x)
      h = clamp(Math.round(start.h + dy), MIN_SIZE_PX, bounds.h - start.y)
      break
  }
  return { x, y, w, h }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}
