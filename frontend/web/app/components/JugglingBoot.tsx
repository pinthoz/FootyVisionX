import React from "react";
import SoccerBall from "./SoccerBall";

interface JugglingBootProps {
  scale?: number;
  className?: string;
}

export default function JugglingBoot({
  scale = 1,
  className = "",
}: JugglingBootProps) {
  // Base coordinate system: 80px width, 96px height.
  // The outer viewport establishes the scaled footprint in the DOM layout,
  // while the inner scene scales cleanly via CSS transform so keyframe physics,
  // trajectory, impact flash, and cleat geometry remain 100% locked and collision-free.
  const viewportWidth = Math.round(80 * scale);
  const viewportHeight = Math.round(96 * scale);

  return (
    <div
      className={`juggling-viewport ${className}`}
      style={{ width: viewportWidth, height: viewportHeight }}
      aria-label="Soccer ball juggling animation"
    >
      <div
        className="juggling-scene"
        style={{
          transform: `scale(${scale})`,
          transformOrigin: "bottom center",
        }}
      >
        {/* 1. Ball Juggling Flight */}
        <div className="juggling-ball-anchor">
          <SoccerBall size={22} mode="static" glow={true} />
        </div>

        {/* 2. Impact flash ring at contact point on instep */}
        <div className="juggling-impact-ring" />

        {/* 3. The Cleat / Chuteira */}
        <div className="juggling-boot-anchor">
          <svg
            width={54}
            height={34}
            viewBox="0 0 54 34"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            className="cleat-svg"
          >
            {/* Main Upper: Aerodynamic black / navy boot */}
            <path
              d="M 6 18 C 5 12, 8 7, 14 7 C 18 7, 19 10, 23 13 C 27 15, 34 18, 42 19.5 C 46 20, 50 22, 50 24.5 C 50 27.5, 45 28.5, 39 28.5 C 28 28.5, 22 27, 16 27 C 11 27, 7 28, 5 26 C 4 24, 5 20, 6 18 Z"
              fill="#0f172a"
              stroke="#334155"
              strokeWidth="1.1"
            />

            {/* Golden Soleplate / Outsole */}
            <path
              d="M 5 26 C 10 27.5, 20 28, 30 28.5 C 39 29, 47 28, 49.5 25.5 L 50 26.5 C 47 29.5, 38 30.5, 30 30 C 19 29.5, 10 29, 4.5 27.5 Z"
              fill="var(--a, #eab308)"
            />

            {/* Outsole Studs / Travas */}
            {/* Heel studs */}
            <rect x="7" y="28" width="3" height="4" rx="1" fill="#e2e8f0" />
            <rect x="13" y="28" width="3" height="4" rx="1" fill="#e2e8f0" />
            {/* Forefoot studs */}
            <rect x="32" y="29.5" width="2.8" height="3.8" rx="1" fill="#e2e8f0" />
            <rect x="38" y="29" width="2.8" height="3.8" rx="1" fill="#e2e8f0" />
            <rect x="44" y="27" width="2.8" height="3.5" rx="1" fill="#e2e8f0" />

            {/* Dynamic Speed Wave / Swoosh (Gold & Silver accent) */}
            <path
              d="M 12 19 C 20 18, 28 20, 42 24 C 35 25, 24 24, 14 22 Z"
              fill="var(--a, #eab308)"
              opacity="0.95"
            />
            <path
              d="M 15 16 C 22 16, 29 18, 40 21.5 C 34 22.5, 25 21.5, 17 19 Z"
              fill="#ffffff"
              opacity="0.8"
            />

            {/* Lacing details on instep */}
            <line x1="22" y1="13" x2="25" y2="15" stroke="#ffffff" strokeWidth="1.2" strokeLinecap="round" />
            <line x1="25" y1="15" x2="28" y2="17" stroke="#ffffff" strokeWidth="1.2" strokeLinecap="round" />
            <line x1="28" y1="17" x2="31" y2="19" stroke="#ffffff" strokeWidth="1.2" strokeLinecap="round" />

            {/* Ankle sock collar */}
            <path
              d="M 12 7 C 13 4.5, 16 4, 18 6 L 19 9 Z"
              fill="#1e293b"
            />
          </svg>
        </div>

        {/* 4. Contact shadow below cleat on pitch */}
        <div className="juggling-ground-shadow" />
      </div>
    </div>
  );
}
