/** Screens in the product spec that are not built yet. Shown honestly in the nav and on /planned/<slug>. */
export type Planned = {
  slug: string; title: string; stage: string; summary: string;
  willDo: string[]; backendToday: string;
};

export const PLANNED: Planned[] = [
  { slug: "pipelines", title: "Pipeline Builder", stage: "Stage 6",
    summary: "Compose a RAG pipeline visually and run it with artifact reuse.",
    willDo: ["Node-based builder with type checking", "Templates and versioning", "Live execution DAG"],
    backendToday: "Not started." },
  { slug: "experiments", title: "Experiment Studio", stage: "Stage 6",
    summary: "Run configurations against the same questions and compare quality, latency and cost.",
    willDo: ["Immutable configurations and isolated indexes", "Side-by-side metrics with example drill-down", "Exportable reports"],
    backendToday: "Partly built: the Evaluation Center runs a grid of indexes, modes, k values and rerankers on the same questions and compares runs with confidence intervals. Exportable reports and isolated experiment indexes are not built." },
];
