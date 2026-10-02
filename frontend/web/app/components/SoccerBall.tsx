import React from "react";

interface SoccerBallProps {
  size?: number;
  mode?: "spin" | "bounce" | "roll" | "static";
  className?: string;
  glow?: boolean;
}

export default function SoccerBall({
  size = 18,
  mode = "spin",
  className = "",
  glow = false,
}: SoccerBallProps) {
  return (
    <span
      className={`soccer-ball-container mode-${mode} ${glow ? "has-glow" : ""} ${className}`}
      style={{ width: size, height: size }}
    >
      <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="soccer-ball-svg"
      >
        {/* Outer sphere with soft drop shadow definition */}
        <circle
          cx="12"
          cy="12"
          r="9.8"
          fill="#f8fafc"
          stroke="#0f172a"
          strokeWidth="1.1"
        />

        {/* Central classic pentagon */}
        <polygon
          points="12.0,8.2 15.6,10.8 14.2,15.1 9.8,15.1 8.4,10.8"
          fill="#0f172a"
          stroke="#0f172a"
          strokeWidth="0.6"
        />

        {/* Tactical seam lines from pentagon vertices outward */}
        <line x1="12.0" y1="8.2" x2="12.0" y2="2.2" stroke="#0f172a" strokeWidth="1.1" />
        <line x1="15.6" y1="10.8" x2="21.4" y2="8.9" stroke="#0f172a" strokeWidth="1.1" />
        <line x1="14.2" y1="15.1" x2="17.8" y2="20.0" stroke="#0f172a" strokeWidth="1.1" />
        <line x1="9.8" y1="15.1" x2="6.2" y2="20.0" stroke="#0f172a" strokeWidth="1.1" />
        <line x1="8.4" y1="10.8" x2="2.6" y2="8.9" stroke="#0f172a" strokeWidth="1.1" />

        {/* Peripheral pentagon patches around the curvature */}
        {/* Top patch */}
        <path d="M10.2 2.3 C11 2.2, 13 2.2, 13.8 2.3 L13.1 4.5 L10.9 4.5 Z" fill="#0f172a" />
        {/* Top-right patch */}
        <path d="M21.3 8.3 C21.7 9.2, 21.8 10.3, 21.6 11.2 L19.8 11.6 L19.2 9.5 Z" fill="#0f172a" />
        {/* Bottom-right patch */}
        <path d="M17.4 20.3 C16.5 21.1, 15.2 21.6, 14.2 21.8 L14.7 19.8 L16.8 18.8 Z" fill="#0f172a" />
        {/* Bottom-left patch */}
        <path d="M6.6 20.3 C7.5 21.1, 8.8 21.6, 9.8 21.8 L9.3 19.8 L7.2 18.8 Z" fill="#0f172a" />
        {/* Top-left patch */}
        <path d="M2.7 8.3 C2.3 9.2, 2.2 10.3, 2.4 11.2 L4.2 11.6 L4.8 9.5 Z" fill="#0f172a" />

        {/* 3D Sphere Lighting specular curve */}
        <circle
          cx="8.5"
          cy="8.5"
          r="6.5"
          fill="url(#ballSpecular)"
          opacity="0.55"
        />
        <defs>
          <radialGradient id="ballSpecular" cx="35%" cy="35%" r="65%">
            <stop offset="0%" stopColor="#ffffff" stopOpacity="0.9" />
            <stop offset="50%" stopColor="#ffffff" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
          </radialGradient>
        </defs>
      </svg>
      {mode === "bounce" && <span className="soccer-ball-shadow" />}
    </span>
  );
}
