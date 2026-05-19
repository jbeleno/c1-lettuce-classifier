import type { ReactNode } from 'react'

interface Props {
  children: ReactNode
  /** Optional accent color (used by ResultCard for the predicted class). */
  accent?: string
  className?: string
}

/** Subtle elevation via color shift (not shadow). One accent per card max,
 *  expressed as a 1px top border in the class color. */
export function Card({ children, accent, className = '' }: Props) {
  return (
    <section
      className={`relative rounded overflow-hidden ${className}`}
      style={{
        background: 'var(--color-surface-1)',
        border: '1px solid var(--color-border-soft)',
      }}
    >
      {accent && (
        <span
          aria-hidden
          className="absolute inset-x-0 top-0 h-px"
          style={{ background: accent }}
        />
      )}
      {children}
    </section>
  )
}
