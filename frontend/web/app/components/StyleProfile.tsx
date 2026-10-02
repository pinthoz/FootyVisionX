"use client";

import { SlotData } from "./PlayerCard";
import Bars, { Bar } from "./Bars";
/** What the XGBoost classifier thinks the player's style looks like. */
export default function StyleProfile({
  slot,
  color,
}: {
  slot: NonNullable<SlotData>;
  color: string;
}) {
  // The ten-role read when it is available, falling back to the four broad groups.
  const roles = slot.score.role_profile ?? {};
  const useRoles = Object.keys(roles).length > 0;
  const source = useRoles ? roles : slot.score.style_profile;

  const entries = Object.entries(source).sort((x, y) => y[1] - x[1]);
  const top = entries[0]?.[0];

  // Ten roles is a long tail of near-zeros; only the ones worth reading are charted.
  const shown = useRoles ? entries.filter(([, p], i) => i < 5 && p >= 0.01) : entries;

  const bars: Bar[] = shown.map(([role, prob]) => ({
    label: role,
    value: prob * 100,
    display: `${(prob * 100).toFixed(1)}%`,
    highlight: role === top,
    detail: `Model classification confidence: ${(prob * 100).toFixed(1)}% for ${role}`,
  }));

  return (
    <div className="style-profile-view">
      <div className="score-header-row">
        <h4 className="player-subheading" style={{ color }}>{slot.player.name}</h4>
        <span className="role-tag">Plays like: {top}</span>
      </div>
      <Bars bars={bars} max={100} color={color} />
      <div className="chartnote">
        {useRoles
          ? "Ten side-agnostic roles, predicted from per-90 metrics alone. Left and right are not predicted: the metrics are counts and carry no side."
          : "Probabilities predicted strictly from spatial event frequencies and per-90 metrics."}
      </div>
    </div>
  );
}
