/** Screens in the product spec that are not built yet. Shown honestly in the nav and on /planned/<slug>. */
export type Planned = {
  slug: string; title: string; stage: string; summary: string;
  willDo: string[]; backendToday: string;
};

export const PLANNED: Planned[] = [];
