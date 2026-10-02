"use client";

import { useEffect, useRef, useState } from "react";
import { Player, api } from "../lib/api";
import PlayerAvatar from "./PlayerAvatar";

export type FeaturedPlayer = {
  id: number;
  name: string;
  position: string;
  desc: string;
  gender: "male" | "female";
};

export const FEATURED_PLAYERS: FeaturedPlayer[] = [
  // Male Stars
  { id: 5503, name: "Lionel Messi", position: "FWD", desc: "Elite Playmaking & Finishing", gender: "male" },
  { id: 5207, name: "Cristiano Ronaldo", position: "FWD", desc: "Supreme Goalscorer & Aerial Threat", gender: "male" },
  { id: 5487, name: "Antoine Griezmann", position: "FWD", desc: "Dynamic Second Striker & Workrate", gender: "male" },
  { id: 5216, name: "Andrés Iniesta", position: "MID", desc: "Magical Progression & Vision", gender: "male" },
  { id: 5539, name: "Casemiro", position: "MID", desc: "Ball Winning & Defensive Anchor", gender: "male" },
  { id: 6394, name: "Aritz Aduriz", position: "FWD", desc: "Classic Box Striker & Heading", gender: "male" },
  { id: 4353, name: "Aymeric Laporte", position: "DEF", desc: "Ball-Playing Center Back", gender: "male" },
  { id: 4324, name: "Dani Alves", position: "DEF", desc: "Attacking Full-Back & Crossing", gender: "male" },
  // Female Stars
  { id: 10143, name: "Alexia Putellas", position: "FWD", desc: "Ballon d'Or Winner & Playmaker", gender: "female" },
  { id: 15284, name: "Aitana Bonmatí", position: "MID", desc: "Supreme Vision & Ball Progression", gender: "female" },
  { id: 10386, name: "Caroline Hansen", position: "FWD", desc: "Elite 1v1 Dribbling & Creator", gender: "female" },
  { id: 4961, name: "Sam Kerr", position: "FWD", desc: "Lethal Finisher & Movement", gender: "female" },
  { id: 15619, name: "Beth Mead", position: "FWD", desc: "Dynamic Winger & Delivery", gender: "female" },
];

type PlayerPickerModalProps = {
  isOpen: boolean;
  targetSlot: "A" | "B";
  initialGender?: "all" | "female" | "male";
  onClose: () => void;
  onSelectPlayer: (p: Player, slot: "A" | "B") => void;
};

export default function PlayerPickerModal({
  isOpen,
  targetSlot,
  initialGender = "all",
  onClose,
  onSelectPlayer,
}: PlayerPickerModalProps) {
  const [query, setQuery] = useState("");
  const [genderFilter, setGenderFilter] = useState<"all" | "female" | "male">(initialGender);
  const inputRef = useRef<HTMLInputElement>(null);

  // Opening the picker adopts the caller's gender filter. Done while rendering, the way
  // React documents for adjusting state to a prop change, rather than in an effect that
  // rendered once with the old filter and then again with the new one.
  const opening = `${isOpen}|${initialGender}`;
  const [lastOpening, setLastOpening] = useState(opening);
  if (opening !== lastOpening) {
    setLastOpening(opening);
    if (isOpen) setGenderFilter(initialGender);
  }

  // Results are kept with the search that produced them. The previous version had no guard
  // against responses arriving out of order, so a slow answer to an earlier keystroke could
  // replace the answer to the current one.
  const search = `${query.trim()}|${genderFilter}`;
  const [found, setFound] = useState<{ search: string; roster: Player[] } | null>(null);
  const roster = found?.roster ?? [];
  const isLoading = isOpen && found?.search !== search;

  useEffect(() => {
    if (!isOpen) return;
    let stale = false;
    api
      .searchPlayers(query.trim(), genderFilter)
      .then((data) => !stale && setFound({ search, roster: data }))
      .catch(() => !stale && setFound({ search, roster: [] }));
    const focus = setTimeout(() => inputRef.current?.focus(), 80);
    return () => {
      stale = true;
      clearTimeout(focus);
    };
  }, [isOpen, search, query, genderFilter]);

  if (!isOpen) return null;

  const isA = targetSlot === "A";
  const themeColor = isA ? "var(--a)" : "var(--b)";
  const displayedStars = FEATURED_PLAYERS.filter(
    (fp) => genderFilter === "all" || fp.gender === genderFilter
  );

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="picker-modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="picker-modal-header">
          <div>
            <div className="picker-target-pill" style={{ background: isA ? "rgba(184, 142, 45, 0.15)" : "rgba(224, 96, 126, 0.15)", color: themeColor }}>
              Targeting: Player {targetSlot}
            </div>
            <h3 className="picker-title">Select Player for Slot {targetSlot}</h3>
          </div>
          <button className="picker-close-btn" onClick={onClose}>✕</button>
        </div>

        {/* Gender Filter Segmented Control */}
        <div className="modal-filter-row">
          <label className="filter-label">Filter Gender:</label>
          <div className="gender-segmented-control">
            <button
              type="button"
              className={`gender-segment-btn ${genderFilter === "all" ? "active" : ""}`}
              onClick={() => setGenderFilter("all")}
            >
              All Players
            </button>
            <button
              type="button"
              className={`gender-segment-btn female ${genderFilter === "female" ? "active" : ""}`}
              onClick={() => setGenderFilter("female")}
            >
              ♀ Female
            </button>
            <button
              type="button"
              className={`gender-segment-btn male ${genderFilter === "male" ? "active" : ""}`}
              onClick={() => setGenderFilter("male")}
            >
              ♂ Male
            </button>
          </div>
        </div>

        {/* Search Input */}
        <div className="picker-search-bar">
          <input
            ref={inputRef}
            type="text"
            value={query}
            placeholder={
              genderFilter === "female"
                ? "Type female player name (e.g. Putellas, Bonmatí, Kerr)..."
                : genderFilter === "male"
                ? "Type male player name (e.g. Messi, Ronaldo, Iniesta)..."
                : "Type player name (e.g. Messi, Putellas, Ronaldo)..."
            }
            onChange={(e) => setQuery(e.target.value)}
          />
          {query && (
            <button className="clear-search-btn" onClick={() => setQuery("")}>✕</button>
          )}
        </div>

        {/* Featured Quick Stars */}
        <div className="featured-stars-section">
          <span className="featured-stars-title">
            Featured Stars {genderFilter !== "all" ? `(${genderFilter})` : ""}:
          </span>
          <div className="featured-chips-grid">
            {displayedStars.map((fp) => (
              <button
                key={fp.id}
                className="featured-chip"
                onClick={() => {
                  onSelectPlayer({ id: fp.id, name: fp.name, country: null, gender: fp.gender }, targetSlot);
                  onClose();
                }}
              >
                <PlayerAvatar name={fp.name} size="sm" themeColor={themeColor} />
                <span className="featured-chip-pos">{fp.position}</span>
                <span className="featured-chip-name">{fp.name}</span>
                <span className={`star-gender-dot ${fp.gender}`}>
                  {fp.gender === "female" ? "♀" : "♂"}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Available Roster List */}
        <div className="picker-roster-section">
          <div className="roster-header">
            <span>
              Available Roster {roster.length > 0 ? `(${roster.length} players)` : ""}
              {genderFilter !== "all" && <span className="roster-gender-filter-badge"> · {genderFilter}</span>}
            </span>
            {isLoading && <span className="loading-badge">Loading…</span>}
          </div>

          <div className="picker-roster-list">
            {roster.length === 0 && !isLoading ? (
              <div className="empty-roster-state">
                <p>No players found matching &ldquo;{query}&rdquo;{genderFilter !== "all" ? ` in ${genderFilter}` : ""}.</p>
                <p className="empty-sub">Try clicking one of the featured stars above or clearing the search/filter.</p>
              </div>
            ) : (
              roster.map((p) => (
                <div
                  key={p.id}
                  className="picker-player-row"
                  onClick={() => {
                    onSelectPlayer(p, targetSlot);
                    onClose();
                  }}
                >
                  <div className="picker-player-left">
                    <PlayerAvatar name={p.name} size="sm" themeColor={themeColor} />
                    <div className="picker-player-info">
                      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <span className="picker-name">{p.name}</span>
                        {genderFilter === "all" && p.gender && (
                          <span className={`player-gender-tag ${p.gender}`}>
                            {p.gender === "female" ? "♀ Female" : "♂ Male"}
                          </span>
                        )}
                      </div>
                      {p.country && <span className="picker-country">{p.country}</span>}
                    </div>
                  </div>
                  <button
                    className="action-btn primary-btn picker-select-btn"
                    style={{ background: themeColor, color: isA ? "#000" : "#fff" }}
                  >
                    Select for {targetSlot}
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
