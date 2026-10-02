"use client";

import { SlotData } from "./PlayerCard";
import Bars, { Bar } from "./Bars";
import { getMetricLabel } from "../lib/api";
/** Why the Performance Score is what it is: weight x percentile, per metric. */
export default function ScoreBreakdown({
  slot,
  color,
}: {
  slot: NonNullable<SlotData>;
  color: string;
}) {
  const bars: Bar[] = slot.score.breakdown.map((b) => ({
    label: getMetricLabel(b.metric),
    value: b.contribution,
    display: b.contribution.toFixed(1),
    detail: `${ordinalPct(b.percentile)} percentile x weight ${b.weight} = ${b.contribution.toFixed(1)} points`,
  }));

  return (
    <div className="score-breakdown-view">
      <div className="score-header-row">
        <h4 className="player-subheading" style={{ color }}>{slot.player.name}</h4>
        <span className="score-pill">
          <strong>{slot.score.performance_score.toFixed(1)}</strong> / 100
        </span>
      </div>
      <Bars bars={bars} color={color} />
      <div className="chartnote">
        Each metric contributes: (peer percentile ÷ 100) × role weight.
      </div>
    </div>
  );
}

function ordinalPct(n: number): string {
  const r = Math.round(n);
  if (r % 100 >= 10 && r % 100 <= 20) return `${r}th`;
  return `${r}${({ 1: "st", 2: "nd", 3: "rd" }[r % 10] as string) ?? "th"}`;
}
