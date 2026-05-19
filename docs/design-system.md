# Design System — Hydroponic Lettuce Classifier

> **Direction.** A precision-instrument dashboard for hydroponic monitoring.
> Cards look like sensor readouts; class labels behave like vegetation
> indicators; probabilities read like a control panel. The interface stays
> out of the way — what holds the eye is the verdict.

The visual language is inherited from the C2 project (commit classifier)
and adapted to the agriculture domain: same slate-900 IDE backdrop, but
the accent palette is re-mapped to lettuce growth stages.

---

## 1. Intent

| Question | Answer |
|---|---|
| **Who** | A greenhouse operator or agronomist deciding *"is this pod ready to harvest?"* on a row of hydroponic lettuce. Also the panel of evaluators looking at the system's outputs. |
| **What must they do** | Read one prediction in under one second, scan the last 50 predictions, compare 6 models. |
| **Feel** | Lab-precise. Dark surface, single bright accent per row. Hierarchy through alignment, not boxes-within-boxes. |

---

## 2. Tokens

### 2.1 Surfaces (whisper-quiet elevation; ~5 % lightness per step)

| Token | Hex | Role |
|---|---|---|
| `--bg` | `#0F172A` | Page canvas (slate-900). |
| `--surface-1` | `#1E293B` | Default card surface. |
| `--surface-2` | `#334155` | Elevated: dropdowns, modals, hover. |
| `--surface-inset` | `#0B1220` | Inputs, code blocks — slightly DARKER than `--bg` so they read as "type here". |

### 2.2 Borders (low-opacity, structural)

| Token | Hex / rgba | Role |
|---|---|---|
| `--border-soft` | `rgba(148,163,184,0.10)` | Default card separation. |
| `--border` | `rgba(148,163,184,0.18)` | Standard divider. |
| `--border-strong` | `rgba(148,163,184,0.28)` | Emphasis (active tab, focus). |
| `--ring-focus` | `#38BDF8` (40 % alpha overlay) | Keyboard focus only. |

### 2.3 Text hierarchy

| Token | Hex | Role |
|---|---|---|
| `--ink` | `#F8FAFC` | Headlines, predicted class label. |
| `--ink-2` | `#CBD5E1` | Body and primary data. |
| `--ink-3` | `#94A3B8` | Metadata (timestamps, model name). |
| `--ink-mute` | `#64748B` | Placeholder, disabled. |

### 2.4 Class colors — the signature

Each growth stage maps to a botanical-meaningful hue. **These five colors
are the only accents in the app.** No others.

| Class | Hex | Interpretation |
|---|---|---|
| `empty_pod` | `#64748B` | slate / muted — nothing growing |
| `germination` | `#A78BFA` | violet — early biological activity |
| `young` | `#FBBF24` | amber — nascent foliage |
| `pod` | `#38BDF8` | sky — developing canopy |
| `Ready` | `#22C55E` | green — harvestable |

The palette doubles as the diff between confidence rows: a single bright
chip per prediction, the rest stays in `--ink-3`.

---

## 3. Typography

| Token | Family | Use |
|---|---|---|
| `--font-sans` | Inter, system-ui | Body, labels, metadata |
| `--font-mono` | Fira Code, JetBrains Mono, ui-monospace | Probabilities, IDs, image SHA |
| `--font-display` | Inter Tight | Section headings only |

---

## 4. Component grammar

- **One accent per card.** No nested bright colors. The card's only color is
  the predicted class chip.
- **Numbers in monospace.** Probabilities and SHA hashes are always rendered
  in `--font-mono` so columns align across rows.
- **No skeumorphism.** Buttons are flat surfaces with a 1 px `--border-soft`
  edge; hover bumps elevation to `--surface-2`.
- **Density before decoration.** Whitespace is set per *information unit*,
  not per *visual element*. Two adjacent confidences are 4 px apart, two
  adjacent cards are 16 px apart.

---

## 5. Source of truth

This file. The Streamlit/React variants of the GUI (none built for C1, but
mocked in `docs/mockups/`) and the matplotlib chart styles all read these
same tokens to stay coherent.
