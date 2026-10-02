"use client";

import { useState } from "react";

export type RadarPoint = {
  /** Position on the axis, 0–100. Percentile vs. same-position peers. */
  percentile: number;
  /** The raw per-90 figure the percentile was computed from. */
  value: number;
};

export type RadarSeries = { name: string; color: string; points: RadarPoint[] };

const SIZE = 440;
const CX = SIZE / 2;
const CY = SIZE / 2;
const RINGS = [20, 40, 60, 80, 100];

function point(i: number, n: number, valuePct: number, radius: number): [number, number] {
  const angle = -Math.PI / 2 + (i * 2 * Math.PI) / n;
  const rr = (radius * Math.max(2, Math.min(100, valuePct))) / 100;
  return [CX + rr * Math.cos(angle), CY + rr * Math.sin(angle)];
}

function ordinal(n: number): string {
  const r = Math.round(n);
  if (r % 100 >= 10 && r % 100 <= 20) return `${r}th`;
  return `${r}${({ 1: "st", 2: "nd", 3: "rd" }[r % 10] as string) ?? "th"}`;
}

export default function Radar({
  axes,
  series,
}: {
  axes: string[];
  series: RadarSeries[];
}) {
  const n = axes.length;
  // Radius dynamic based on number of axes
  const R = n > 12 ? SIZE / 2 - 76 : SIZE / 2 - 64;
  const [hoverAxis, setHoverAxis] = useState<number | null>(null);
  const [hoverSeries, setHoverSeries] = useState<number | null>(null);

  const showInlineValues = series.length === 1 && n <= 10;

  return (
    <div className="radarwrap">
      <svg
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        width="100%"
        className="radar-svg"
        style={{ maxWidth: SIZE, filter: "drop-shadow(0 4px 20px rgba(0,0,0,0.4))" }}
      >
        <defs>
          {series.map((s, idx) => (
            <linearGradient
              key={`grad-${idx}`}
              id={`radar-grad-${idx}`}
              x1="0%"
              y1="0%"
              x2="100%"
              y2="100%"
            >
              <stop offset="0%" stopColor={s.color} stopOpacity={0.35} />
              <stop offset="100%" stopColor={s.color} stopOpacity={0.08} />
            </linearGradient>
          ))}
          <radialGradient id="radar-bg-radial" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="rgba(255,255,255,0.03)" />
            <stop offset="100%" stopColor="rgba(0,0,0,0.3)" />
          </radialGradient>
        </defs>

        {/* Background web polygon */}
        <polygon
          points={axes.map((_, i) => point(i, n, 100, R).join(",")).join(" ")}
          fill="url(#radar-bg-radial)"
        />

        {/* Rings */}
        {RINGS.map((ring) => (
          <polygon
            key={ring}
            points={axes.map((_, i) => point(i, n, ring, R).join(",")).join(" ")}
            fill="none"
            stroke={ring === 100 ? "rgba(255,255,255,0.18)" : "rgba(255,255,255,0.07)"}
            strokeWidth={ring === 100 ? 1.5 : 1}
            strokeDasharray={ring % 40 === 0 ? undefined : "3,3"}
          />
        ))}

        {/* Outer label annotation */}
        <text
          x={CX}
          y={CY - R - 8}
          fontSize={9}
          fontWeight={600}
          fill="var(--muted)"
          textAnchor="middle"
          letterSpacing="0.05em"
        >
          100th %
        </text>

        {/* Axes and Labels */}
        {axes.map((label, i) => {
          const [x, y] = point(i, n, 100, R);
          const labelDist = n > 12 ? 116 : 118;
          const [lx, ly] = point(i, n, labelDist, R);
          const anchor = lx > CX + 8 ? "start" : lx < CX - 8 ? "end" : "middle";
          const isHighlighted = hoverAxis === i;

          return (
            <g key={label} className="radar-axis-group" onMouseEnter={() => setHoverAxis(i)} onMouseLeave={() => setHoverAxis(null)}>
              <line
                x1={CX}
                y1={CY}
                x2={x}
                y2={y}
                stroke={isHighlighted ? "rgba(255,255,255,0.35)" : "rgba(255,255,255,0.09)"}
                strokeWidth={isHighlighted ? 1.5 : 1}
              />
              <text
                x={lx}
                y={showInlineValues ? ly - 5 : ly}
                fontSize={n > 12 ? 9.5 : 11}
                fontWeight={isHighlighted ? 700 : 500}
                fill={isHighlighted ? "var(--text)" : "var(--muted)"}
                textAnchor={anchor}
                dominantBaseline="middle"
                style={{ cursor: "pointer", transition: "all 0.15s ease" }}
              >
                {label}
              </text>
              {showInlineValues && (
                <text
                  x={lx}
                  y={ly + 8}
                  fontSize={9.5}
                  fontWeight={600}
                  fill="var(--text)"
                  textAnchor={anchor}
                  dominantBaseline="middle"
                >
                  {series[0]?.points[i]?.value.toFixed(2)}
                </text>
              )}
            </g>
          );
        })}

        {/* Series Polygons */}
        {series.map((s, si) => {
          const pts = s.points.map((p, i) => point(i, n, p.percentile, R).join(",")).join(" ");
          const isSeriesHovered = hoverSeries === si || hoverSeries === null;

          return (
            <g key={`series-${si}`} opacity={isSeriesHovered ? 1 : 0.4} style={{ transition: "opacity 0.2s ease" }}>
              <polygon
                points={pts}
                fill={`url(#radar-grad-${si})`}
                stroke={s.color}
                strokeWidth={2.5}
                strokeLinejoin="round"
              />
              {s.points.map((p, i) => {
                const [x, y] = point(i, n, p.percentile, R);
                const isHovered = hoverAxis === i && (hoverSeries === si || hoverSeries === null);

                return (
                  <g key={i}>
                    {/* Glowing outer marker on hover */}
                    {isHovered && (
                      <circle
                        cx={x}
                        cy={y}
                        r={8}
                        fill="none"
                        stroke={s.color}
                        strokeWidth={1.5}
                        strokeOpacity={0.8}
                      />
                    )}
                    <circle
                      cx={x}
                      cy={y}
                      r={isHovered ? 5.5 : 4}
                      fill={s.color}
                      stroke="var(--panel)"
                      strokeWidth={2}
                      style={{ transition: "r 0.15s ease" }}
                    />
                    {/* Larger hit target */}
                    <circle
                      cx={x}
                      cy={y}
                      r={14}
                      fill="transparent"
                      onMouseEnter={() => {
                        setHoverAxis(i);
                        setHoverSeries(si);
                      }}
                      onMouseLeave={() => {
                        setHoverAxis(null);
                        setHoverSeries(null);
                      }}
                      style={{ cursor: "pointer" }}
                    />
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>

      {/* Dual Comparative Tooltip */}
      {hoverAxis !== null && (
        <div className="radartip dual-tip">
          <div className="tip-metric-title">{axes[hoverAxis]}</div>
          <div className="tip-entries">
            {series.map((s) => {
              const pt = s.points[hoverAxis];
              if (!pt) return null;
              return (
                <div key={s.name} className="tip-row">
                  <span className="dot" style={{ background: s.color }} />
                  <span className="tip-name">{s.name}:</span>
                  <span className="tip-val">{pt.value.toFixed(2)}/90</span>
                  <span className="tip-pct" style={{ color: s.color }}>
                    ({ordinal(pt.percentile)} pct)
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
