"use client";

export type Bar = {
  label: string;
  /** Drives the bar length, on the same scale as `max`. */
  value: number;
  /** Printed at the end of the bar. Defaults to `value` rounded to one decimal. */
  display?: string;
  /** Extra context shown on hover, e.g. the percentile behind a contribution. */
  detail?: string;
  /** Set on the row that represents the selected player, so it reads as "you are here". */
  highlight?: boolean;
  /** Custom highlight color override */
  customColor?: string;
  /** Rank number if in a leaderboard */
  rank?: number;
  /** Gender indicator if applicable */
  gender?: string | null;
};

export default function Bars({
  bars,
  max,
  color,
  onPick,
  onPickA,
  onPickB,
}: {
  bars: Bar[];
  max?: number;
  color: string;
  onPick?: (index: number) => void;
  onPickA?: (index: number) => void;
  onPickB?: (index: number) => void;
}) {
  const ceiling = max ?? Math.max(...bars.map((b) => b.value), 1);

  return (
    <div className="bars">
      {bars.map((b, i) => {
        const barColor = b.customColor || color;
        return (
          <div
            key={`${b.label}-${i}`}
            className={`barrow${b.highlight ? " on" : ""}${onPick ? " clickable" : ""}`}
            onClick={onPick ? () => onPick(i) : undefined}
            title={b.detail}
          >
            {b.rank !== undefined && (
              <span className={`rank-badge ${b.rank <= 3 ? `top-${b.rank}` : ""}`}>
                #{b.rank}
              </span>
            )}
            <span className="barlabel" title={b.label}>{b.label}</span>
            <span className="bartrack">
              <span
                className="barfill"
                style={{
                  width: `${Math.max(0, Math.min(100, (b.value / ceiling) * 100))}%`,
                  background: barColor,
                  opacity: b.highlight === false ? 0.45 : 1,
                  boxShadow: b.highlight ? `0 0 10px ${barColor}` : undefined,
                }}
              />
            </span>
            <span className="barvalue">{b.display ?? b.value.toFixed(1)}</span>

            {(onPickA || onPickB) && (
              <div className="bar-actions-hover" onClick={(e) => e.stopPropagation()}>
                {onPickA && (
                  <button
                    type="button"
                    className="micro-btn a-btn"
                    title="Set to Player A"
                    onClick={() => onPickA(i)}
                  >
                    + A
                  </button>
                )}
                {onPickB && (
                  <button
                    type="button"
                    className="micro-btn b-btn"
                    title="Set to Player B"
                    onClick={() => onPickB(i)}
                  >
                    + B
                  </button>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
