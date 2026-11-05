"use client";

import dynamic from "next/dynamic";

function ChartSkeleton() {
  return <div className="h-full min-h-32 w-full animate-pulse rounded-lg bg-surface-2" role="status" aria-label="Loading chart" />;
}

// ssr: false keeps the charting library out of the server render and out of every page's first-load JavaScript.
export const Histogram = dynamic(() => import("./charts").then((m) => m.Histogram), { ssr: false, loading: ChartSkeleton });
export const RunsBarChart = dynamic(() => import("./charts").then((m) => m.RunsBarChart), { ssr: false, loading: ChartSkeleton });
export const ProjectionPlot = dynamic(() => import("./charts").then((m) => m.ProjectionPlot), { ssr: false, loading: ChartSkeleton });
export { SERIES_COLORS } from "./charts-colors";
