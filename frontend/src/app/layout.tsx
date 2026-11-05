import type { Metadata } from "next";
import { GeistMono } from "geist/font/mono";
import { GeistSans } from "geist/font/sans"; // self-hosted: no network access is needed at build time
import { Providers } from "@/components/providers";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Learning Studio",
  description: "A laboratory for building, comparing and understanding retrieval-augmented generation.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable} h-full antialiased`}>
      <body className="min-h-full">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
