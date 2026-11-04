import type { NextConfig } from "next";

// The studio is an authenticated, client-driven workbench, so the experimental
// cacheComponents / partialPrefetching defaults from the scaffold are switched off:
// they add Suspense requirements and state-preservation behaviour this app does not want.
const nextConfig: NextConfig = {
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
