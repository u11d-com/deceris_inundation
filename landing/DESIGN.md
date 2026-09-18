---
name: deceris-inundation landing
description: Traceable GPU shallow-water modelling on polygonal meshes.
colors:
  paper: "#f5f7f3"
  paper-deep: "#e8ece6"
  ink: "#111613"
  muted: "#667069"
  line: "#c9d0c8"
  line-strong: "#98a39b"
  acid: "#c8f05a"
  acid-deep: "#4f7000"
  water: "#b9d9d1"
  water-deep: "#80b8ad"
  proof-text: "#aebbb0"
  proof-line: "#8ca89d"
  proof-label: "#8e9b91"
typography:
  display:
    fontFamily: "IBM Plex Sans, Avenir Next, Helvetica Neue, Arial, sans-serif"
    fontSize: "clamp(3.7rem, 5.6vw, 5.5rem)"
    fontWeight: 600
    lineHeight: 0.97
    letterSpacing: "-0.065em"
  headline:
    fontFamily: "IBM Plex Sans, Avenir Next, Helvetica Neue, Arial, sans-serif"
    fontSize: "clamp(2.8rem, 5.2vw, 5rem)"
    fontWeight: 600
    lineHeight: 0.97
    letterSpacing: "-0.065em"
  body:
    fontFamily: "IBM Plex Sans, Avenir Next, Helvetica Neue, Arial, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "IBM Plex Mono, SFMono-Regular, monospace"
    fontSize: "11px"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "0.1em"
  nav:
    fontSize: "13px"
  button:
    fontSize: "13px"
  mobile-wordmark:
    fontSize: "20px"
  hero-lede:
    fontSize: "18px"
  proof-copy:
    fontSize: "15px"
  card-title:
    fontSize: "23px"
  body-small:
    fontSize: "14px"
  code:
    fontSize: "12px"
  micro:
    fontSize: "10px"
  figure-micro:
    fontSize: "8px"
  figure-detail:
    fontSize: "9px"
  mobile-display:
    fontSize: "clamp(3.55rem, 15vw, 6rem)"
rounded:
  none: "0"
  dot: "50%"
spacing:
  shell: "min(1220px, calc(100vw - 64px))"
  section: "128px"
  compact: "24px"
components:
  button-primary:
    backgroundColor: "{colors.acid}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "0 17px"
    height: "47px"
  button-primary-hover:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.acid}"
    rounded: "{rounded.none}"
    padding: "0 17px"
    height: "47px"
---

# Design System: deceris-inundation landing

## Overview

**Creative North Star: "The Traceable Field"**

The landing page treats the solver as a field instrument: a quiet paper ground, precise ink, a single signal color, and geometry that shows its work. The page is conventional by intent—documentation-led, evidence-first, and legible beside serious open-source infrastructure—while the mesh diagram and benchmark section give it a product-specific visual voice.

The visual system is flat, measured, and information-dense without feeling compressed. Hairline rules make the page read like a technical record; chartreuse marks the current path, verified gate, or primary action. Water tones are reserved for mesh states and diagrams, never used as decorative decoration.

**Key Characteristics:**
- Light paper surface with ink-first text and hairline dividers.
- One acid accent for actions, selected emphasis, and status.
- Diagrams use crisp geometry, not illustrative chrome.
- Open-source navigation and proof appear before platform context.

## Colors

The palette pairs cool off-white surfaces with near-black text, a restrained green signal, and pale water tones for solver diagrams.

### Primary
- **Field acid** (`#c8f05a`): Primary action background, active signal, and key emphasis.
- **Field acid deep** (`#4f7000`): Accent text and status marks where the lighter acid needs contrast.

### Secondary
- **Water** (`#b9d9d1`): Filled mesh cells and calm simulation state.
- **Water deep** (`#80b8ad`): Water diagram strokes and secondary geometry.

### Neutral
- **Paper** (`#f5f7f3`): Page background.
- **Paper deep** (`#e8ece6`): Diagram field and quiet tonal surface.
- **Ink** (`#111613`): Primary text and dark proof panel.
- **Muted** (`#667069`): Supporting text and labels.
- **Line** (`#c9d0c8`): Standard divider.
- **Line strong** (`#98a39b`): Structural rule and table head.

### Named Rules
**The One Signal Rule.** Chartreuse marks what can be acted on or verified now; do not scatter additional accent colors across the surface.

## Typography

**Display Font:** IBM Plex Sans (with Avenir Next, Helvetica Neue, Arial fallbacks)

**Body Font:** IBM Plex Sans (with Avenir Next, Helvetica Neue, Arial fallbacks)

**Label/Mono Font:** IBM Plex Mono (with SFMono-Regular fallback)

**Character:** The sans is workhorse and compact with a slightly technical rhythm. Mono is reserved for measurements, backend names, formats, and execution labels so it carries meaning rather than costume.

### Hierarchy
- **Display** (600, `clamp(3.7rem, 5.6vw, 5.5rem)`, 0.97): Hero proposition.
- **Headline** (600, `clamp(2.8rem, 5.2vw, 5rem)`, 0.97): Section thesis statements.
- **Title** (600, 18–23px, 1.2): Mechanism and proof subheadings.
- **Body** (400, 16px, 1.5): Explanatory copy, constrained to roughly 39–48ch in landing compositions.
- **Label** (600, 11px, 1.2, 0.1em): Instrument labels, code metadata, and status vocabulary.

### Named Rules
**The Weight Rule.** Hierarchy comes from size and weight; emphasis uses weight or scale, never gradient text or decorative lettering.

## Layout

The page uses a centered shell capped at 1220px with a 32px viewport gutter on wide screens. The opening composition is a two-column hero: proposition and action beside a mesh field. A four-step pipeline follows as a ruled strip. Later sections alternate split editorial layouts, a dark proof panel, ruled benchmark rows, a three-column mechanism explanation, and an execution-path close.

Section spacing is generous above headings and tighter below explanatory copy. At 850px, grids collapse to one column and the pipeline becomes a 2×2 matrix. At 540px, the shell uses a 16px gutter, navigation condenses to wordmark plus repository action, benchmark rows stack their state beneath the contract, and the page retains the same reading order.

## Elevation & Depth

The system is flat by default. It uses tonal layering (`paper` / `paper-deep` / `ink`) and 1px rules instead of shadows or floating cards. Motion provides the only lift: the hero copy rises into place and the mesh field reveals with a clipped entrance. Reduced-motion users receive the fully visible final state.

### Named Rules
**The Flat Field Rule.** Surfaces are separated by tone or rule, never by ornamental shadow.

## Shapes

Primary controls and content containers use square corners (`0px`), reinforcing the instrument-panel language. Status marks and diagram nodes are circular. The hero mesh is a rectangular field with crisp SVG geometry. Dividers are 1px and remain visible at every breakpoint.

## Components

### Buttons
- **Shape:** Square, 47px minimum height for primary action.
- **Primary:** Field acid background, ink text, 17px horizontal padding, compact arrow affordance.
- **Hover / Focus:** Primary inverts to ink with acid text; all controls use a 3px deep-acid focus ring with 4px offset. Hover lifts by 2px.
- **Secondary:** Text-only link with ink underline on interaction; no button shell.

### Cards / Containers
- **Corner Style:** Square; no radius on panels or rows.
- **Background:** Paper, paper-deep, or ink by semantic mode.
- **Shadow Strategy:** No shadows; tonal layering and rules carry depth.
- **Border:** 1px line or line-strong; dark proof panel uses a darker internal rule.
- **Internal Padding:** 17–46px, scaled down at the compact breakpoint.

### Navigation
The masthead is a single ruled row. The wordmark is left-aligned; section links use muted 13px sans text; repository is ink with an underline. On compact screens, section links collapse and the repository arrow remains available.

### Mesh Field
The signature component is an inline SVG polygon mesh: linework for topology, pale water-filled cells for state, black nodes for inspectable points, and mono annotations for `z_mean`, `h + hu`, CSR, and Hilbert order. It is a diagram with semantic alternative text, not a decorative illustration.

### Benchmark Ledger
The proof surface combines a dark lake-at-rest diagram with a ruled six-row harness ledger. Results that have not been selected for ingest are explicitly marked `pending`; no placeholder metric is disguised as evidence.

## Do's and Don'ts

### Do:
- **Do** use the acid signal only for an action, selected proof state, or key emphasis.
- **Do** preserve the pipeline vocabulary: mesh, adjacency, shaders, snapshots.
- **Do** use mono for backend names, file formats, equations, and execution labels.
- **Do** keep proof states explicit when benchmark output is not yet ingested.
- **Do** preserve the flat field and hairline-rule grammar across future pages.

### Don't:
- **Don't** add performance numbers, customer proof, licensing, or benchmark outcomes without a supplied source.
- **Don't** replace the polygon mesh with a raster-only visual metaphor.
- **Don't** introduce gradients, glass, rounded card stacks, or decorative shadows.
- **Don't** use mono as display typography or as a generic marker for every paragraph.
- **Don't** hide the repository action behind platform marketing.
