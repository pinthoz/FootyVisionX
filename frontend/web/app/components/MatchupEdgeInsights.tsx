"use client";

import { SlotData } from "./PlayerCard";
import { getCleanPlayerName } from "../lib/photos";
/** Quick comparative takeaways when both A and B are active. */
export default function MatchupEdgeInsights({
  slotA,
  slotB,
}: {
  slotA: NonNullable<SlotData>;
  slotB: NonNullable<SlotData>;
}) {
  const xgA = slotA.radar.metrics["xg_per90"]?.value ?? 0;
  const xgB = slotB.radar.metrics["xg_per90"]?.value ?? 0;
  const progA = slotA.radar.metrics["progressive_passes_per90"]?.value ?? 0;
  const progB = slotB.radar.metrics["progressive_passes_per90"]?.value ?? 0;
  const tklA = (slotA.radar.metrics["tackles_per90"]?.value ?? 0) + (slotA.radar.metrics["interceptions_per90"]?.value ?? 0);
  const tklB = (slotB.radar.metrics["tackles_per90"]?.value ?? 0) + (slotB.radar.metrics["interceptions_per90"]?.value ?? 0);

  const cleanNameA = getCleanPlayerName(slotA.player.name);
  const cleanNameB = getCleanPlayerName(slotB.player.name);

  return (
    <div className="matchup-edge-grid">
      <div className="edge-item">
        <div className="edge-label">Goal Threat (xG/90)</div>
        <div className="edge-comparison">
          <span className={xgA >= xgB ? "edge-leader a-color" : "edge-val"}>
            {cleanNameA}: {xgA.toFixed(2)}
          </span>
          <span className="edge-sep">vs</span>
          <span className={xgB > xgA ? "edge-leader b-color" : "edge-val"}>
            {cleanNameB}: {xgB.toFixed(2)}
          </span>
        </div>
      </div>

      <div className="edge-item">
        <div className="edge-label">Ball Progression (Prog/90)</div>
        <div className="edge-comparison">
          <span className={progA >= progB ? "edge-leader a-color" : "edge-val"}>
            {cleanNameA}: {progA.toFixed(1)}
          </span>
          <span className="edge-sep">vs</span>
          <span className={progB > progA ? "edge-leader b-color" : "edge-val"}>
            {cleanNameB}: {progB.toFixed(1)}
          </span>
        </div>
      </div>

      <div className="edge-item">
        <div className="edge-label">Defensive Actions (Tkl+Int)</div>
        <div className="edge-comparison">
          <span className={tklA >= tklB ? "edge-leader a-color" : "edge-val"}>
            {cleanNameA}: {tklA.toFixed(1)}
          </span>
          <span className="edge-sep">vs</span>
          <span className={tklB > tklA ? "edge-leader b-color" : "edge-val"}>
            {cleanNameB}: {tklB.toFixed(1)}
          </span>
        </div>
      </div>
    </div>
  );
}
