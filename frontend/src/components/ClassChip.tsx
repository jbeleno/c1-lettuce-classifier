import { CLASS_META, type LettuceClass } from '../types'

interface Props {
  label: LettuceClass
  size?: 'sm' | 'md' | 'lg'
}

/** Single accent per card. Used as the only piece of color in result cards,
 *  table rows and confusion-matrix axes. */
export function ClassChip({ label, size = 'md' }: Props) {
  const meta = CLASS_META[label]
  const px =
    size === 'sm' ? 'px-1.5 py-0.5 text-[10px]'
    : size === 'md' ? 'px-2 py-0.5 text-[11px]'
    : 'px-2.5 py-1 text-xs'

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded font-mono tabular ${px}`}
      style={{
        background: `color-mix(in oklab, ${meta.cssVar} 18%, transparent)`,
        color: meta.cssVar,
        border: `1px solid color-mix(in oklab, ${meta.cssVar} 35%, transparent)`,
      }}
    >
      <span
        aria-hidden
        className="size-1.5 rounded-full"
        style={{ background: meta.cssVar }}
      />
      {label}
    </span>
  )
}
