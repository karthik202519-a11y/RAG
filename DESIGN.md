---
name: RAG
description: A guided engineering workbench for inspecting RAG stages and evidence.
colors:
  accent: "#2856c7"
  accent-ink: "#ffffff"
  accent-soft: "#e7efff"
  bg: "#f3f6fc"
  surface: "#ffffff"
  surface-2: "#edf2fa"
  border: "#dce4f1"
  text: "#17243b"
  text-dim: "#576780"
  sidebar: "#13213a"
  sidebar-text: "#e7edfa"
  sidebar-selected: "#dce8ff"
  sidebar-selected-text: "#1745a4"
  good: "#166a2e"
  good-soft: "#dcf3e1"
  warn: "#9a4608"
  warn-soft: "#fdf0d9"
  bad: "#b42318"
  bad-soft: "#fde7e4"
typography:
  headline:
    fontFamily: "Geist Sans, system-ui, sans-serif"
    fontSize: "28px"
    fontWeight: 600
    letterSpacing: "-0.025em"
  title:
    fontFamily: "Geist Sans, system-ui, sans-serif"
    fontSize: "20px"
    fontWeight: 600
    lineHeight: 1.4
  body:
    fontFamily: "Geist Sans, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Geist Sans, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 500
    lineHeight: 1.333333
  mono:
    fontFamily: "Geist Mono, monospace"
rounded:
  badge: "4px"
  control: "6px"
  disclosure: "8px"
  panel: "12px"
  canvas: "16px"
spacing:
  tight: "8px"
  control: "10px"
  compact: "12px"
  panel: "16px"
  generous: "20px"
  section: "24px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.accent-ink}"
    rounded: "{rounded.control}"
    height: "36px"
    padding: "0 14px"
  button-secondary:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text}"
    rounded: "{rounded.control}"
    height: "36px"
    padding: "0 14px"
  button-ghost:
    rounded: "{rounded.control}"
    height: "36px"
    padding: "0 14px"
  button-danger:
    backgroundColor: "{colors.bad-soft}"
    textColor: "{colors.bad}"
    rounded: "{rounded.control}"
    height: "36px"
    padding: "0 14px"
  input:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text}"
    rounded: "{rounded.control}"
    height: "36px"
    padding: "0 10px"
  badge:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.text-dim}"
    rounded: "{rounded.badge}"
    padding: "2px 6px"
  card:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.panel}"
    padding: "16px"
---

# Design System: RAG

## Overview

**Creative North Star: "The Guided Engineering Workbench"**

Cool, light working surfaces and a distinct navy desktop navigation rail support long sessions of reading, configuring and inspecting evidence. Geist Sans keeps explanations and dense controls in one practical voice; Geist Mono identifies vectors, configuration and numerical artifacts. The interface prioritizes task order, readable state and source provenance.

Compact controls sit within generously separated panels. Explanations accompany the choices they explain, in compact disclosures the learner can expand. Advanced controls remain available. This is a code-led implementation: 12ui's sign-in service was unreachable, and no generated comp was approved. This document records the implementation rather than a generated visual target.

**Key Characteristics:**

- Cool canvas, white bordered surfaces and restrained blue action emphasis.
- Grouped navigation, visible stage order and local configuration.
- Expandable teaching and source-construction evidence.
- Semantic status labels, tabular numbers and explicit uncertainty.

## Colors

The frontmatter records the default light palette; CSS custom properties in `frontend/src/app/globals.css` supply the live theme.

### Primary

- **Workbench Blue** (`accent`): primary actions, selected pipeline nodes, links and explanatory step numbers. Its pale companion (`accent-soft`) marks selection without overwhelming text.

### Secondary

- **Navigation Navy** (`sidebar`): a desktop wayfinding region with pale labels and a light selected row. Explicit sidebar colors remain independent of the working-surface theme.
- **Evidence Green**, **Caution Amber** and **Failure Red** (`good`, `warn`, `bad`): paired with their soft backgrounds for badges and messages. Meaning is also expressed in text.

### Neutral

- **Cool Canvas** (`bg`): page ground.
- **Working White** (`surface`): controls, cards and construction evidence.
- **Inset Mist** (`surface-2`): secondary information, hover surfaces and method explanations.
- **Quiet Divider**, **Navy Ink** and **Slate Annotation** (`border`, `text`, `text-dim`): boundaries, primary reading and supporting information.

Dark mode follows the OS unless a persisted manual choice overrides it. The dark theme uses an olive-black canvas (`#121311`), charcoal surfaces (`#1a1b18`, `#22231f`), pale text (`#ecece6`), muted annotation (`#a2a398`) and teal actions (`#3fb7aa`, with `#06201d` ink and `#123532` soft fill). Border is `#34362f`. Semantic text/soft pairs are green (`#66c47c` / `#15301b`), amber (`#e0a24a` / `#3a2a10`) and red (`#f08a80` / `#3d1a17`). Eight categorical chunk fills defined in CSS distinguish adjacent chunks rather than status or rank.

**The Evidence Color Rule.** Keep categorical chunk colors separate from semantic success, caution and failure labels.

## Typography

**Display and Body Font:** self-hosted Geist Sans, falling back to system UI and sans-serif.

**Label/Mono Font:** labels use Geist Sans; vectors, configuration and selected statistics use self-hosted Geist Mono.

**Character:** a single UI sans ties instructional text to workbench controls. Mono conveys inspected data rather than decorative technical styling.

### Hierarchy

- **Headline:** semibold route titles with tight tracking, as recorded in the headline token.
- **Title:** semibold section headings; smaller sections also use 18px/28px, and card headers use 14px/20px.
- **Body:** the global body token; subtitles and method prose sometimes use a more relaxed line height (1.625). Teaching introductions commonly stop at a 48rem measure.
- **Label:** field labels, hints, badges and provenance metadata. Local hint utilities may adjust line height.
- **Mono:** stored-coordinate previews, recipe JSON and statistic values. Tables and mono text use tabular numerals. Heading text balances where supported.

## Layout

At 768px and above, a sticky, vertically scrollable rail occupies 240px; the remaining width holds the project toolbar and work surface. Main padding is 16px, increasing to 20px at 640px. New workbench routes center within a 1500px maximum width. Panels use 16px or 20px internal padding, with 24px gaps between major sections.

Below 768px, a menu button opens a modal drawer, 288px wide and capped at 85vw. The drawer uses the normal surface theme rather than the desktop rail's navy override. It traps keyboard focus, closes with Escape, locks body scrolling and restores focus on close. The toolbar truncates project selection; secondary identity labels hide on narrow screens.

Teaching steps stack and become three columns at 768px. Pipeline nodes use two columns by default, three at 768px and six at 1280px; connecting arrows appear only in the six-column layout. Configuration and execution panels stack below 1024px. Wider Pipeline Builder screens pair flexible configuration with an execution sidebar; Experiment Studio pairs narrower setup with wider results. Bounded fields and min-width constraints let panels shrink. Wide experiment tables scroll inside their panel rather than widening the page.

Screenshots in `.impeccable/review/redesign-*.png` record reviewed desktop and 390px mobile states. The supplied finish-review disposition is **ship**, with the reported mobile overflow finding resolved. This records visual review rather than claiming every route or backend path was tested.

## Elevation & Depth

Working panels are flat: fine borders and tonal differences establish grouping without card shadows. The sticky toolbar uses translucent canvas and backdrop blur. Mobile navigation adds a dark scrim and an elevated drawer; the keyboard skip link also receives a shadow. Depth communicates overlay or persistent chrome rather than ordinary content importance.

Main content enters with a fade and 4px upward movement over 250ms, easing out. Hover transitions remain local; chunk segments use a 120ms filter transition. Reduced-motion preferences suppress animations and transitions.

## Shapes

Controls have compact rounded corners; badges are tighter, disclosures softer and panels more generous, following the frontmatter radius scale. The pipeline canvas has the largest recurring radius and an 18px dotted grid. Simple outlined Lucide icons accompany labels. Selected stage nodes change border and fill while preserving their silhouette.

## Components

### Buttons

Compact, medium-weight controls pair text with optional 16px icons. Standard dimensions are in the frontmatter; small buttons are 32px high with 10px horizontal padding and 12px text. Primary buttons brighten on hover. Secondary and ghost variants gain the inset surface; danger buttons use a red border and soft fill. Loading adds a spinner and disables the action. Disabled buttons reduce opacity and use a blocked cursor; disabled primary actions also adopt neutral surfaces and text. Keyboard focus uses a 2px accent outline with 2px offset.

### Chips

Badges are compact, nonwrapping descriptive labels with semantic text/soft-fill pairs. They are not filter controls. Status wording stays visible alongside color.

### Cards / Containers

Theme-adjusted panels use fine borders and no resting shadow. Shared Card offers an optional divided header and action group, with 16px body padding. New workbench sections often use 20px padding. Empty states use dashed borders; errors use a red soft fill, icon and alert role. Loading skeletons preserve structure with a live status message.

### Inputs / Fields

Inputs and selects share full-width bounded controls with surface fill, fine borders and 10px horizontal inset. Labels sit above, hints below. Native number controls, checkboxes and selects retain familiar behavior. The global keyboard outline supplies focus. Mutation errors appear near their action through ErrorBox; there is no universal field-error visual variant to infer.

### Navigation

Desktop navigation can collapse from a 224px labelled sidebar to a 64px icon rail; icon links retain accessible names and titles. Working pages use fluid width, with smaller companion panels and compact workflow links. Settings separates runtime, provider availability and connection tests. Chunking uses a method selector beside the preview instead of a tall catalog. Long index metadata is expandable. Desktop links group tools into Overview, Build, Explore, Measure, Learn and System. Outlined 16px icons accompany 14px labels. Active links use a pale row and stronger blue text with `aria-current`. Mobile links close the drawer after navigation. A keyboard skip link targets the main region.

### Teaching and construction disclosures

LearningGuide starts collapsed below the title, with purpose, three steps and an amber inspection note. MethodGuide explains the selected method, advantages, tradeoffs and prerequisites in a tonal inset. Construction evidence is collapsed above graph/embedding controls and expands actual source samples, stored-coordinate previews or grounded relation examples. Disclosure is the recurring signature: understanding stays beside the tool while advanced configuration remains accessible.

### Pipeline stage canvas and experiment comparison

Six stage buttons expose order, configuration summaries and typed input/output descriptions. Selecting a stage changes the adjacent form; this is a staged configuration interface, not a drag-and-drop editor. Saved recipes, execution selection and recent runs occupy a companion panel. Experiment results use a scrollable table and per-question inspection. Highest observed hit rate is accompanied by uncertainty language rather than asserting a universally better method.

### Storage inspection

Storage Explorer inherits the workbench panels, semantic badges and local table scrolling. A border-separated evidence path leads from document to version, chunk and native vector record. Two pressed-state buttons switch Application tables and Vector storage; an adjacent sample-size selector offers one, three or five examples. The inspected build exposes 23 application tables, with schema revision, purpose, benefit, column types and explicit Foreign key versus Logical link badges. The count is data-driven rather than a fixed navigation promise.

A full-width filter and table selector replace the tall table directory at every breakpoint. Sample rows, Columns & keys and Relationships are separate pressed-state views. Schema and sample tables scroll locally. All-field row inspection expands on request; short fields form two columns above 640px, while text and nested JSON span both columns. Monospace identifies table names, columns, IDs, types, JSON and vector coordinates. Long identifiers and nested payloads wrap inside their containers. Follow-link icons and explicit chunk-follow buttons change the inspected record; no editing controls are present.

A collapsed reference in the vector view compares five physical layouts (Qdrant, FAISS, Chroma, LanceDB and pgvector) without suggesting all engines have SQL tables. Store and ready-index selectors precede native storage objects, fields, available index definitions and sampled records. Metadata, the first 12 stored coordinates, vector norm and raw float32 byte count appear beside the linked application chunk on wide screens and stack on narrow screens. The byte count explicitly excludes engine and metadata overhead. Empty selections explain how to build an index; orphaned native records disclose the missing linked chunk. Read-only, project-scoped sample limits, truncation and redaction are stated beside examples.

The supplied Storage Explorer finish-review verdict is **ship**: both views passed at an actual 390 CSS-pixel viewport, with the RAG header correct and no open material overflow defect. Screenshots are `.impeccable/review/storage-tables-*.png` and `storage-vectors-*.png`. This is visual-review evidence, not certification of every storage engine's runtime behavior.

## Do's and Don'ts

### Do:

- **Do** use shared theme variables and primitives for new working surfaces.
- **Do** keep explanations, prerequisites and source evidence near the controls they qualify.
- **Do** preserve local scrolling for wide measurements and wrapping for source text and coordinates.
- **Do** show state in words as well as semantic color and retain visible keyboard focus.
- **Do** distinguish stored vectors from their lossy visualization and model proposals from grounded source evidence.

### Don't:

- **Don't** present illustrative or synthetic results as measured project evidence.
- **Don't** describe the stage selector as drag-and-drop or imply an approved generated comp exists.
- **Don't** infer production quality from a small benchmark or turn an observed winner into a universal claim.
- **Don't** replace explanations with decorative metrics or unlabeled color alone.
