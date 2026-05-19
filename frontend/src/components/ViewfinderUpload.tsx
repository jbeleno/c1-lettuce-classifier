/** ViewfinderUpload — replaces the generic "drop file here" box. Borders
 *  look like camera-viewfinder corner brackets (┏┓┗┛). Drag, click and
 *  paste-from-clipboard all work. Returns a Blob to the parent.  */
import { useCallback, useRef, useState } from 'react'
import { Upload } from 'lucide-react'

interface Props {
  onPick: (file: File) => void
  busy?: boolean
  preview?: string | null
}

export function ViewfinderUpload({ onPick, busy, preview }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [drag, setDrag] = useState(false)

  const handle = useCallback(
    (file: File | null | undefined) => {
      if (!file) return
      if (!file.type.startsWith('image/')) {
        // Surface error via the input's validity instead of an alert
        inputRef.current?.setCustomValidity('Image files only')
        inputRef.current?.reportValidity()
        return
      }
      onPick(file)
    },
    [onPick],
  )

  return (
    <div
      className="relative w-full"
      onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDrag(false)
        handle(e.dataTransfer.files?.[0])
      }}
      onPaste={(e) => handle(e.clipboardData.files?.[0])}
    >
      <input
        ref={inputRef}
        id="viewfinder-input"
        type="file"
        accept="image/*"
        className="sr-only"
        onChange={(e) => handle(e.target.files?.[0])}
      />
      <label
        htmlFor="viewfinder-input"
        className={`group relative grid aspect-[4/3] w-full place-items-center cursor-pointer
                   transition-colors duration-150
                   ${drag ? 'bg-[var(--color-surface-2)]' : 'bg-[var(--color-surface-inset)]'}
                   ${busy ? 'pointer-events-none opacity-60' : ''}`}
      >
        {preview ? (
          <img
            src={preview}
            alt="Selected pod (preview)"
            className="absolute inset-2 size-[calc(100%-1rem)] object-contain"
          />
        ) : (
          <div className="flex flex-col items-center gap-3 text-center" aria-hidden>
            <Upload size={28} strokeWidth={1.5} color="var(--color-ink-3)" />
            <p className="font-mono text-[12px] text-[var(--color-ink-3)]">
              drag a JPG, paste from clipboard, or
              <span className="text-[var(--color-ring-focus)]"> click to browse</span>
            </p>
          </div>
        )}

        {/* Four corner brackets — the signature viewfinder framing */}
        <Bracket position="tl" />
        <Bracket position="tr" />
        <Bracket position="bl" />
        <Bracket position="br" />
      </label>
    </div>
  )
}

function Bracket({ position }: { position: 'tl' | 'tr' | 'bl' | 'br' }) {
  const map = {
    tl: 'top-2 left-2 border-t-2 border-l-2',
    tr: 'top-2 right-2 border-t-2 border-r-2',
    bl: 'bottom-2 left-2 border-b-2 border-l-2',
    br: 'bottom-2 right-2 border-b-2 border-r-2',
  } as const
  return (
    <span
      aria-hidden
      className={`pointer-events-none absolute size-6 ${map[position]}`}
      style={{ borderColor: 'var(--color-ring-focus)' }}
    />
  )
}
