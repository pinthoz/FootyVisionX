// The corner arc of a pitch, with the flag as a dot. Drawn rather than imported: the
// mark is four geometric primitives, so an SVG stays sharp at any size, follows the
// theme colours, and costs no asset request.

export default function Logo({ size = 30 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="2 2 21 21"
      fill="none"
      role="img"
      aria-label="FootyVision"
      className="logo-mark"
    >
      <g
        stroke="var(--chalk)"
        strokeWidth={1.3}
        strokeLinecap="round"
      >
        {/* Touchlines meeting at the corner. */}
        <path d="M5 19V4" />
        <path d="M5 19h16" />
        {/* The corner arc, centred on the flag. */}
        <path d="M5 11a8 8 0 0 1 8 8" />
      </g>
      {/* The flag itself: the one place the mark carries colour. */}
      <circle cx={5} cy={19} r={1.8} fill="var(--a-light)" />
    </svg>
  );
}
