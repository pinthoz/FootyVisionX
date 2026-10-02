"use client";

import { useEffect, useState } from "react";
import { resolvePlayerPhoto } from "../lib/photos";

type PlayerAvatarProps = {
  name: string;
  size?: "sm" | "md" | "lg" | "xl" | "hero";
  themeColor?: string;
  className?: string;
};

export default function PlayerAvatar({
  name,
  size = "md",
  themeColor = "var(--a)",
  className = "",
}: PlayerAvatarProps) {
  // Each lookup is stored with the name it was for, and "loading" is simply the absence of
  // one for the current name. Resetting state at the top of the effect did the same job
  // with an extra render per name change.
  const [photo, setPhoto] = useState<{ name: string; url: string | null } | null>(null);
  const [brokenFor, setBrokenFor] = useState<string | null>(null);
  const loading = photo?.name !== name;
  const photoUrl = loading ? null : photo.url;
  const error = brokenFor === name;

  useEffect(() => {
    let active = true;
    resolvePlayerPhoto(name)
      .then((url) => active && setPhoto({ name, url }))
      .catch(() => active && setPhoto({ name, url: null }));
    return () => {
      active = false;
    };
  }, [name]);

  const initials = getInitials(name);
  const sizeClass = `avatar-${size}`;

  if (photoUrl && !error) {
    return (
      <div className={`player-avatar-wrapper ${sizeClass} ${className}`}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={photoUrl}
          alt={name}
          className="player-avatar-img"
          onError={() => setBrokenFor(name)}
          loading="lazy"
        />
        <div className="avatar-ring" style={{ borderColor: themeColor }} />
      </div>
    );
  }

  return (
    <div
      className={`player-avatar-wrapper ${sizeClass} monogram-fallback ${className}`}
      style={{
        background: `linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%)`,
        borderColor: themeColor,
      }}
      title={name}
    >
      <span className="avatar-initials" style={{ color: themeColor }}>
        {initials}
      </span>
      {loading && <div className="avatar-pulse" />}
    </div>
  );
}

function getInitials(name: string): string {
  if (!name) return "⚽";
  const parts = name.trim().split(/\s+/);
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}
