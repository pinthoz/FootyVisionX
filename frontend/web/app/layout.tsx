import type { Metadata } from "next";
import { Archivo, Inter } from "next/font/google";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { Analytics } from "@vercel/analytics/next";
import "./globals.css";

// Signage face for the wordmark, panel eyebrows and scores. A grotesque rather than a
// condensed one: condensed reads well on a scoreboard but cramps small interface
// labels like "Natural Language Query", which is where most of this font actually lands.
const display = Archivo({
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  variable: "--font-display",
  display: "swap",
});

// Body face stays quiet and highly legible: this is a reading and comparison tool.
const body = Inter({
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

export const metadata: Metadata = {
  title: "FootyVision Scouting",
  description: "Football scouting: player similarity, radars, performance scores and written reports.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    // suppressHydrationWarning: browser extensions inject attributes (e.g. theme/dark-mode)
    // onto <html>/<body> before React hydrates; ignore those benign attribute mismatches.
    <html lang="en" className={`${display.variable} ${body.variable}`} suppressHydrationWarning>
      <body suppressHydrationWarning>
        {/* Decorative pitch markings behind the dashboard. */}
        <div className="pitch-markings" aria-hidden="true" />
        {children}
        {/* Vercel's analytics load from /_vercel/insights, which exists only on Vercel.
            The static export served by CloudFront would request it and get a 404. */}
        {process.env.STATIC_EXPORT ? null : (
          <>
            <Analytics />
            <SpeedInsights />
          </>
        )}
      </body>
    </html>
  );
}
