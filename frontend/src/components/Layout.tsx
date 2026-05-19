import { NavLink, Outlet } from 'react-router-dom'

const TABS = [
  { to: '/',         label: 'predict'  },
  { to: '/history',  label: 'history'  },
  { to: '/metrics',  label: 'metrics'  },
  { to: '/models',   label: 'models'   },
] as const

export function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      <header
        className="sticky top-0 z-10 flex items-center justify-between px-6 py-3 backdrop-blur"
        style={{
          background: 'color-mix(in oklab, var(--color-bg) 92%, transparent)',
          borderBottom: '1px solid var(--color-border-soft)',
        }}
      >
        <a
          href="/"
          className="flex items-center gap-2 font-mono text-[13px] tracking-tight"
          style={{ color: 'var(--color-ink-2)' }}
        >
          <span style={{ color: 'var(--color-class-ready)' }}>●</span>
          <span style={{ color: 'var(--color-ink)' }}>c1</span>
          <span style={{ color: 'var(--color-ink-3)' }}>/lettuce</span>
        </a>
        <nav aria-label="Primary">
          <ul className="flex items-center gap-1">
            {TABS.map((t) => (
              <li key={t.to}>
                <NavLink
                  to={t.to}
                  end={t.to === '/'}
                  className={({ isActive }) =>
                    `relative inline-block px-3 py-1.5 font-mono text-[12px] transition-colors duration-150
                     ${isActive ? 'text-[var(--color-ink)]' : 'text-[var(--color-ink-3)] hover:text-[var(--color-ink-2)]'}`
                  }
                >
                  {({ isActive }) => (
                    <>
                      {t.label}
                      <span
                        aria-hidden
                        className="pointer-events-none absolute inset-x-3 bottom-0 h-px transition-opacity duration-150"
                        style={{
                          background: 'var(--color-ring-focus)',
                          opacity: isActive ? 1 : 0,
                        }}
                      />
                    </>
                  )}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-8">
        <Outlet />
      </main>

      <footer className="mx-auto w-full max-w-6xl px-6 pb-6 pt-4 font-mono text-[11px]"
              style={{ color: 'var(--color-ink-mute)' }}>
        c1 · hydroponic lettuce growth-stage classifier · USCO BEINSOF52
      </footer>
    </div>
  )
}
