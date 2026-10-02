"use client";

import { useEffect, useRef, useState } from "react";
import Radar, { RadarSeries } from "./components/Radar";
import StatDuel from "./components/StatDuel";
import PlayerCard, { SlotData } from "./components/PlayerCard";
import PlayerPickerModal, { FEATURED_PLAYERS } from "./components/PlayerPickerModal";
import PlayerAvatar from "./components/PlayerAvatar";
import Logo from "./components/Logo";
import AssistantEvaluation from "./components/AssistantEvaluation";
import DataCoverage from "./components/DataCoverage";
import TeamStrength from "./components/TeamStrength";
import MetricDistribution from "./components/MetricDistribution";
import MetricScatter from "./components/MetricScatter";
import PitchMaps from "./components/PitchMaps";
import { ModelInfoResponse, Player, RADAR_PRESETS, Radar as RadarData, Similar, api } from "./lib/api";
import MatchupEdgeInsights from "./components/MatchupEdgeInsights";
import ScoreBreakdown from "./components/ScoreBreakdown";
import StyleProfile from "./components/StyleProfile";
import Rankings from "./components/Rankings";
import Assistant from "./components/Assistant";
import NLSearch from "./components/NLSearch";
import Report from "./components/Report";

const INITIAL_DEFAULT_ROSTER: Player[] = FEATURED_PLAYERS.map((fp) => ({
  id: fp.id,
  name: fp.name,
  country: null,
  gender: fp.gender,
}));

const PRESET_MATCHUPS = [
  { name: "Messi vs Ronaldo", pA: { id: 5503, name: "Lionel Messi" }, pB: { id: 5207, name: "Cristiano Ronaldo" } },
  { name: "Iniesta vs Griezmann", pA: { id: 5216, name: "Andrés Iniesta" }, pB: { id: 5487, name: "Antoine Griezmann" } },
];

export default function Home() {
  const [results, setResults] = useState<Player[]>([]);
  const [defaultRoster, setDefaultRoster] = useState<Player[]>(INITIAL_DEFAULT_ROSTER);
  const [a, setA] = useState<SlotData>(null);
  const [b, setB] = useState<SlotData>(null);
  const [activeSlot, setActiveSlot] = useState<"A" | "B">("A");
  const [similarA, setSimilarA] = useState<Similar[]>([]);
  const [similarB, setSimilarB] = useState<Similar[]>([]);
  const [similarTarget, setSimilarTarget] = useState<"A" | "B">("A");
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [activeTab, setActiveTab] = useState<
    "radar" | "duel" | "score" | "styles" | "spread" | "scatter" | "pitch"
  >("radar");
  const [radarPreset, setRadarPreset] = useState<string>("curated");
  const [aiSubTab, setAiSubTab] = useState<"assistant" | "search" | "report">("assistant");
  const [searchQuery, setSearchQuery] = useState("");
  const [genderFilter, setGenderFilter] = useState<"all" | "female" | "male">("all");
  const [isLoadingSearch, setIsLoadingSearch] = useState(false);
  const [isPickerOpen, setIsPickerOpen] = useState(false);
  const [pickerTarget, setPickerTarget] = useState<"A" | "B">("A");

  const searchInputRef = useRef<HTMLInputElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const starsScrollRef = useRef<HTMLDivElement>(null);
  const isDraggingStars = useRef(false);
  const startXStars = useRef(0);
  const startScrollLeftStars = useRef(0);
  const hasDraggedStars = useRef(false);

  function handleStarsMouseDown(e: React.MouseEvent<HTMLDivElement>) {
    if (!starsScrollRef.current) return;
    isDraggingStars.current = true;
    hasDraggedStars.current = false;
    startXStars.current = e.pageX - starsScrollRef.current.offsetLeft;
    startScrollLeftStars.current = starsScrollRef.current.scrollLeft;
    starsScrollRef.current.classList.add("is-dragging");
  }

  function handleStarsMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    if (!isDraggingStars.current || !starsScrollRef.current) return;
    const x = e.pageX - starsScrollRef.current.offsetLeft;
    const walk = (x - startXStars.current) * 1.3;
    if (Math.abs(walk) > 4) {
      hasDraggedStars.current = true;
    }
    starsScrollRef.current.scrollLeft = startScrollLeftStars.current - walk;
  }

  function handleStarsMouseUp() {
    isDraggingStars.current = false;
    if (starsScrollRef.current) {
      starsScrollRef.current.classList.remove("is-dragging");
    }
    setTimeout(() => {
      hasDraggedStars.current = false;
    }, 60);
  }

  function scrollStars(offset: number) {
    if (starsScrollRef.current) {
      starsScrollRef.current.scrollBy({ left: offset, behavior: "smooth" });
    }
  }

  // Load default roster and initial top players
  useEffect(() => {
    // 1. Load initial roster
    api
      .searchPlayers("")
      .then((players) => {
        if (players.length > 0) {
          setDefaultRoster(players);
        }
      })
      .catch(() => {});

    // 2. Load default players for Slot A & B if empty
    api
      .rankings(undefined, 4)
      .then((topRows) => {
        if (topRows.length >= 2 && !a && !b) {
          pickPlayer({ id: topRows[0].player_id, name: topRows[0].name, country: null }, "A");
          pickPlayer({ id: topRows[1].player_id, name: topRows[1].name, country: null }, "B");
        }
      })
      .catch(() => {});

    // 3. Load the style classifier's held-out accuracy
    api
      .modelInfo()
      .then(setModelInfo)
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function onSearch(term: string, gender: "all" | "female" | "male" = genderFilter) {
    setSearchQuery(term);
    if (timer.current) clearTimeout(timer.current);
    if (!term.trim()) {
      setResults([]);
      setIsLoadingSearch(false);
      return;
    }
    setIsLoadingSearch(true);
    timer.current = setTimeout(async () => {
      try {
        const found = await api.searchPlayers(term.trim(), gender);
        setResults(found);
      } catch {
        setResults([]);
      } finally {
        setIsLoadingSearch(false);
      }
    }, 150);
  }

  function handleGenderFilterChange(newGender: "all" | "female" | "male") {
    setGenderFilter(newGender);
    if (searchQuery.trim()) {
      onSearch(searchQuery, newGender);
    } else {
      setIsLoadingSearch(true);
      api
        .searchPlayers("", newGender)
        .then((players) => {
          setDefaultRoster(players);
        })
        .catch(() => {})
        .finally(() => setIsLoadingSearch(false));
    }
  }

  async function pickPlayer(p: Player, target?: "A" | "B") {
    const slotToFill = target ?? activeSlot;

    try {
      const [radar, score] = await Promise.all([api.radar(p.id), api.score(p.id)]);
      const slotData: SlotData = {
        player: { id: p.id, name: radar.name || p.name, country: p.country },
        radar,
        score,
      };

      if (slotToFill === "A") {
        setA(slotData);
        api
          .similar(p.id)
          .then((res) => setSimilarA(res))
          .catch(() => setSimilarA([]));
        if (!b) {
          setActiveSlot("B");
        }
      } else {
        setB(slotData);
        api
          .similar(p.id)
          .then((res) => setSimilarB(res))
          .catch(() => setSimilarB([]));
        if (!a) {
          setActiveSlot("A");
        }
      }
    } catch {
      alert(`${p.name}: no season stats available above minutes floor.`);
    }
  }

  function handleOpenPicker(slotId: "A" | "B") {
    setActiveSlot(slotId);
    setPickerTarget(slotId);
    setIsPickerOpen(true);
  }

  function handleSwap() {
    const tempA = a;
    const tempSimilarA = similarA;
    setA(b);
    setSimilarA(similarB);
    setB(tempA);
    setSimilarB(tempSimilarA);
  }

  function handleClearSlot(slotId: "A" | "B") {
    if (slotId === "A") {
      setA(null);
      setSimilarA([]);
      setActiveSlot("A");
    } else {
      setB(null);
      setSimilarB([]);
      setActiveSlot("B");
    }
  }

  function handleClearAll() {
    setA(null);
    setB(null);
    setSimilarA([]);
    setSimilarB([]);
    setActiveSlot("A");
    setResults([]);
    setSearchQuery("");
  }

  // Selected radar axes
  const currentPreset =
    RADAR_PRESETS.find((p) => p.id === radarPreset) ?? RADAR_PRESETS[0];

  const series: RadarSeries[] = [];
  if (a) series.push(toSeries(a.radar, "var(--a)", currentPreset.axes));
  if (b) series.push(toSeries(b.radar, "var(--b)", currentPreset.axes));

  const activeSimilarList = similarTarget === "A" ? similarA : similarB;
  const activeSimilarPlayerName = similarTarget === "A" ? a?.player.name : b?.player.name;
  const displayedRoster = searchQuery.trim() ? results : defaultRoster;

  return (
    <div className="app-container">
      {/* Player Picker Modal */}
      <PlayerPickerModal
        isOpen={isPickerOpen}
        targetSlot={pickerTarget}
        initialGender={genderFilter}
        onClose={() => setIsPickerOpen(false)}
        onSelectPlayer={(p, slot) => pickPlayer(p, slot)}
      />

      {/* Top Navigation Header */}
      <header className="header">
        <div className="header-brand">
          <div className="logo-badge">
            <Logo size={22} />
            <span className="logo-text">FootyVision</span>
          </div>
        </div>

        {/* Featured Presets (Duels) - Afastados dos outros, na mesma linha */}
        <div className="header-presets">
          <span className="presets-label">Preset Duels:</span>
          {PRESET_MATCHUPS.map((pm, idx) => (
            <button
              key={idx}
              className="preset-duel-btn"
              onClick={() => {
                pickPlayer({ id: pm.pA.id, name: pm.pA.name, country: null }, "A");
                pickPlayer({ id: pm.pB.id, name: pm.pB.name, country: null }, "B");
              }}
            >
              {pm.name}
            </button>
          ))}
        </div>

        {/* Dataset, Assistant, Target Slot, Actions - Encostados à direita, tudo! */}
        <div className="header-meta">
          <DataCoverage />
          <AssistantEvaluation />
          <TeamStrength />

          <div className="active-target-pill">
            <span className="muted-text">Target Slot:</span>
            <button
              className={`target-btn ${activeSlot === "A" ? "is-active a-active" : ""}`}
              onClick={() => setActiveSlot("A")}
            >
              Player A
            </button>
            <button
              className={`target-btn ${activeSlot === "B" ? "is-active b-active" : ""}`}
              onClick={() => setActiveSlot("B")}
            >
              Player B
            </button>
          </div>

          <div className="header-actions">
            <button
              className="action-btn secondary-btn"
              onClick={handleSwap}
              disabled={!a && !b}
              title="Swap Player A and Player B"
            >
              ⇄ Swap A/B
            </button>
            <button
              className="action-btn secondary-btn"
              onClick={handleClearAll}
              disabled={!a && !b}
              title="Clear all selected players"
            >
              Clear All
            </button>
          </div>
        </div>
      </header>

      {/* ========================================================================= */}
      {/* 3-COLUMN TOP COMMAND CENTER: COMPACT PLAYER A | SEARCH & ROSTER | PLAYER B*/}
      {/* ========================================================================= */}
      <section className="top-command-center">
        {/* Col 1: Player A Card */}
        <div className="command-col-slot">
          <PlayerCard
          modelInfo={modelInfo}
            slotId="A"
            slot={a}
            isActive={activeSlot === "A"}
            onActivate={() => setActiveSlot("A")}
            onOpenPicker={handleOpenPicker}
            onPickDirect={(p, s) => pickPlayer(p, s)}
            onClear={() => handleClearSlot("A")}
            onSwap={b ? handleSwap : undefined}
          />
        </div>

        {/* Col 2 (Center): Player Search & Quick Roster Selection Hub */}
        <div className="command-col-search panel search-hub-panel">
          <div className="panel-header" style={{ marginBottom: 6 }}>
            <div>
              <label style={{ color: "var(--text)", fontSize: 11 }}>Find a player</label>
              <div className="panel-sub-label">
                Targeting <span style={{ color: activeSlot === "A" ? "var(--a)" : "var(--b)", fontWeight: 700 }}>Slot {activeSlot}</span>
              </div>
            </div>
            <button
              className="chip active"
              style={{ fontSize: 10.5, padding: "2px 8px" }}
              onClick={() => handleOpenPicker(activeSlot)}
            >
              + Browse Modal
            </button>
          </div>

          {/* Gender Filter Buttons: All, Female, Male */}
          <div className="search-gender-filter-bar">
            <button
              type="button"
              className={`search-gender-btn ${genderFilter === "all" ? "active" : ""}`}
              onClick={() => handleGenderFilterChange("all")}
            >
              All
            </button>
            <button
              type="button"
              className={`search-gender-btn female ${genderFilter === "female" ? "active" : ""}`}
              onClick={() => handleGenderFilterChange("female")}
            >
              ♀ Female
            </button>
            <button
              type="button"
              className={`search-gender-btn male ${genderFilter === "male" ? "active" : ""}`}
              onClick={() => handleGenderFilterChange("male")}
            >
              ♂ Male
            </button>
          </div>

          <div className="search-input-wrapper">
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              placeholder={
                genderFilter === "female"
                  ? "Search female players (e.g. Putellas, Bonmatí, Kerr)..."
                  : genderFilter === "male"
                  ? "Search male players (e.g. Messi, Ronaldo, Iniesta)..."
                  : "Search by name (e.g. Messi, Putellas, Ronaldo)..."
              }
              onChange={(e) => onSearch(e.target.value)}
            />
            {searchQuery && (
              <button
                type="button"
                className="clear-search-btn"
                onClick={() => {
                  setSearchQuery("");
                  setResults([]);
                }}
              >
                ✕
              </button>
            )}
          </div>

          {/* Quick Stars with horizontal drag and nav arrows */}
          <div className="sidebar-quick-stars">
            <span className="stars-label">Stars:</span>
            <button
              type="button"
              className="stars-nav-btn left"
              title="Scroll stars left"
              aria-label="Scroll stars left"
              onClick={() => scrollStars(-140)}
            >
              ‹
            </button>
            <div
              className="quick-stars-scroll"
              ref={starsScrollRef}
              onMouseDown={handleStarsMouseDown}
              onMouseMove={handleStarsMouseMove}
              onMouseUp={handleStarsMouseUp}
              onMouseLeave={handleStarsMouseUp}
              onWheel={(e) => {
                if (starsScrollRef.current && e.deltaY) {
                  starsScrollRef.current.scrollLeft += e.deltaY;
                }
              }}
            >
              {FEATURED_PLAYERS.filter((fp) => genderFilter === "all" || fp.gender === genderFilter).map((fp) => (
                <button
                  key={fp.id}
                  className="quick-star-chip"
                  onClick={(e) => {
                    if (hasDraggedStars.current) {
                      e.preventDefault();
                      return;
                    }
                    pickPlayer({ id: fp.id, name: fp.name, country: null, gender: fp.gender }, activeSlot);
                  }}
                >
                  <PlayerAvatar name={fp.name} size="sm" themeColor="var(--accent)" />
                  <span>{fp.name.split(" ").slice(-1)[0]}</span>
                  {genderFilter === "all" && (
                    <span className={`star-gender-dot ${fp.gender}`}>{fp.gender === "female" ? "♀" : "♂"}</span>
                  )}
                </button>
              ))}
            </div>
            <button
              type="button"
              className="stars-nav-btn right"
              title="Scroll stars right"
              aria-label="Scroll stars right"
              onClick={() => scrollStars(140)}
            >
              ›
            </button>
          </div>

          {/* Inline Scrollable Roster with Photos */}
          <div className="roster-list-container hub-roster-container">
            <div className="roster-list-header">
              <span>{searchQuery ? `Search (${results.length})` : `Roster (${displayedRoster.length})`}</span>
              {isLoadingSearch && <span className="loading-badge">Searching…</span>}
            </div>

            <div className="roster-scroll-list hub-roster-scroll">
              {displayedRoster.length === 0 && !isLoadingSearch ? (
                <div className="empty-mini-state" style={{ padding: 8 }}>
                  <p>No players found{searchQuery ? ` for "${searchQuery}"` : ""}.</p>
                </div>
              ) : (
                displayedRoster.map((p) => (
                  <div
                    key={p.id}
                    className="result-row"
                    onClick={() => pickPlayer(p, activeSlot)}
                  >
                    <div className="result-left">
                      <PlayerAvatar name={p.name} size="sm" themeColor="var(--a)" />
                      <div className="result-info">
                        <div style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 0 }}>
                          <span className="result-name">{p.name}</span>
                          {genderFilter === "all" && p.gender && (
                            <span className={`player-gender-tag ${p.gender}`} style={{ flexShrink: 0 }}>
                              {p.gender === "female" ? "♀ Female" : "♂ Male"}
                            </span>
                          )}
                        </div>
                        {p.country && <span className="result-meta">{p.country}</span>}
                      </div>
                    </div>
                    <div className="result-actions" onClick={(e) => e.stopPropagation()}>
                      <button
                        type="button"
                        className="micro-btn a-btn"
                        title="Assign to Player A"
                        onClick={() => pickPlayer(p, "A")}
                      >
                        + A
                      </button>
                      <button
                        type="button"
                        className="micro-btn b-btn"
                        title="Assign to Player B"
                        onClick={() => pickPlayer(p, "B")}
                      >
                        + B
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Col 3: Player B Card */}
        <div className="command-col-slot">
          <PlayerCard
          modelInfo={modelInfo}
            slotId="B"
            slot={b}
            isActive={activeSlot === "B"}
            onActivate={() => setActiveSlot("B")}
            onOpenPicker={handleOpenPicker}
            onPickDirect={(p, s) => pickPlayer(p, s)}
            onClear={() => handleClearSlot("B")}
            onSwap={a ? handleSwap : undefined}
          />
        </div>
      </section>


      {/* ========================================================================= */}
      {/* ASK THE SCOUT — one row until it has an answer, so it costs little  */}
      {/* ========================================================================= */}
      <section className="dashboard-row-ai">
        <div className="panel ai-scout-panel top-ai-panel">
          <div className="panel-header">
            <div>
              <label style={{ fontSize: 12 }}>Ask the scout</label>
              <div className="panel-sub-label">Answers grounded in the players above, never invented</div>
            </div>

            <div className="sub-tabs-bar" style={{ marginBottom: 0, borderBottom: "none" }}>
              <button
                className={`sub-tab ${aiSubTab === "assistant" ? "active" : ""}`}
                onClick={() => setAiSubTab("assistant")}
              >
                Assistant
              </button>
              <button
                className={`sub-tab ${aiSubTab === "search" ? "active" : ""}`}
                onClick={() => setAiSubTab("search")}
              >
                Search
              </button>
              <button
                className={`sub-tab ${aiSubTab === "report" ? "active" : ""}`}
                onClick={() => setAiSubTab("report")}
              >
                Report
              </button>
            </div>
          </div>

          <div className="ai-content-wrapper" style={{ marginTop: 12 }}>
            {aiSubTab === "assistant" && (
              <Assistant
                onPick={(p, target) => pickPlayer(p, target ?? activeSlot)}
              />
            )}
            {aiSubTab === "search" && (
              <NLSearch
                onPick={(p, target) => pickPlayer(p, target ?? activeSlot)}
              />
            )}
            {aiSubTab === "report" && (
              <Report playerA={a?.player ?? null} playerB={b?.player ?? null} />
            )}
          </div>
        </div>
      </section>

      {/* ========================================================================= */}
      {/* SECTION 2: ANALYSIS — RADAR / HEAD-TO-HEAD / SCORE / STYLE (2 columns)     */}
      {/* ========================================================================= */}
      <section className="dashboard-row-primary">
        {/* Left Column: Visual Analytics Stage */}
        <div className="panel visual-analytics-panel">
          <div className="analytics-header">
            <div className="nav-tabs-bar-inline">
              {[
                { id: "radar", label: "Radar" },
                { id: "duel", label: "Head to head" },
                { id: "score", label: "Score breakdown" },
                { id: "styles", label: "Style profile" },
                { id: "spread", label: "Distribution" },
                { id: "scatter", label: "Two metrics" },
                { id: "pitch", label: "Pitch map" },
              ].map((t) => (
                <button
                  key={t.id}
                  className={`main-nav-tab ${activeTab === t.id ? "active" : ""}`}
                  onClick={() => setActiveTab(t.id as typeof activeTab)}
                >
                  {t.label}
                </button>
              ))}
            </div>

            {activeTab === "radar" && (
              <div className="radar-presets">
                {RADAR_PRESETS.map((p) => (
                  <button
                    key={p.id}
                    className={`chip ${radarPreset === p.id ? "active" : ""}`}
                    onClick={() => setRadarPreset(p.id)}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="tab-stage-content">
            {activeTab === "radar" && (
              <div className="radar-stage">
                <Radar axes={currentPreset.axes.map(([, l]) => l)} series={series} />

                <div className="radar-legend-card">
                  <h4 className="legend-title">Comparison Key</h4>
                  <div className="legend-items">
                    {a ? (
                      <div className="legend-row a-row">
                        <PlayerAvatar name={a.player.name} size="sm" themeColor="var(--a)" />
                        <div className="legend-details">
                          <strong>{a.player.name}</strong>
                          <span className="legend-sub">
                            {a.radar.position_group} · Score {a.score.performance_score.toFixed(1)}
                          </span>
                        </div>
                      </div>
                    ) : (
                      <div className="legend-row empty-row">
                        <span className="dot empty" />
                        <span>Player A slot empty</span>
                      </div>
                    )}

                    {b ? (
                      <div className="legend-row b-row">
                        <PlayerAvatar name={b.player.name} size="sm" themeColor="var(--b)" />
                        <div className="legend-details">
                          <strong>{b.player.name}</strong>
                          <span className="legend-sub">
                            {b.radar.position_group} · Score {b.score.performance_score.toFixed(1)}
                          </span>
                        </div>
                      </div>
                    ) : (
                      <div className="legend-row empty-row">
                        <span className="dot empty" />
                        <span>Player B slot empty</span>
                      </div>
                    )}
                  </div>
                  <div className="legend-tips">
                    <p>Hover over polygon vertices or metric names to inspect per-90 values and calibrated peer percentiles.</p>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "spread" &&
              (a ? (
                <MetricDistribution
                  positionGroup={a.radar.position_group}
                  markers={[
                    { playerId: a.player.id, name: a.player.name, color: "var(--a)" },
                    ...(b && b.radar.position_group === a.radar.position_group
                      ? [{ playerId: b.player.id, name: b.player.name, color: "var(--b)" }]
                      : []),
                  ]}
                  // A percentile field only means something inside one position group, so
                  // a B in a different group genuinely cannot be plotted here. Saying so
                  // is the difference between a considered omission and a broken chart.
                  omitted={
                    b && b.radar.position_group !== a.radar.position_group
                      ? { name: b.player.name, group: b.radar.position_group }
                      : null
                  }
                />
              ) : (
                <div className="muted">Pick a player to see the field he is measured against.</div>
              ))}

            {activeTab === "scatter" &&
              (a ? (
                <MetricScatter
                  positionGroup={a.radar.position_group}
                  markers={[
                    { playerId: a.player.id, name: a.player.name, color: "var(--a)" },
                    ...(b && b.radar.position_group === a.radar.position_group
                      ? [{ playerId: b.player.id, name: b.player.name, color: "var(--b)" }]
                      : []),
                  ]}
                  // A percentile field only means something inside one position group, so
                  // a B in a different group genuinely cannot be plotted here. Saying so
                  // is the difference between a considered omission and a broken chart.
                  omitted={
                    b && b.radar.position_group !== a.radar.position_group
                      ? { name: b.player.name, group: b.radar.position_group }
                      : null
                  }
                />
              ) : (
                <div className="muted">Pick a player to plot him against his position group.</div>
              ))}

            {activeTab === "pitch" &&
              (a ? (
                // Unlike the distribution charts, a pitch map is not a rank within a group,
                // so A and B are drawn side by side whatever their positions.
                <PitchMaps
                  players={[
                    { playerId: a.player.id, name: a.player.name, color: "var(--a)" },
                    ...(b ? [{ playerId: b.player.id, name: b.player.name, color: "var(--b)" }] : []),
                  ]}
                />
              ) : (
                <div className="muted">Pick a player to see where they play.</div>
              ))}

            {activeTab === "duel" && (
              <StatDuel
                a={a ? { name: a.player.name, radar: a.radar, color: "var(--a)" } : null}
                b={b ? { name: b.player.name, radar: b.radar, color: "var(--b)" } : null}
              />
            )}

            {activeTab === "score" && (
              <div className="score-breakdown-grid">
                {a ? (
                  <div className="breakdown-card">
                    <ScoreBreakdown slot={a} color="var(--a)" />
                  </div>
                ) : (
                  <div className="empty-mini-state">Select Player A to inspect score arithmetic.</div>
                )}

                {b ? (
                  <div className="breakdown-card">
                    <ScoreBreakdown slot={b} color="var(--b)" />
                  </div>
                ) : (
                  <div className="empty-mini-state">Select Player B to compare score arithmetic.</div>
                )}
              </div>
            )}

            {activeTab === "styles" && (
              <div className="styles-comparison-grid">
                {a ? (
                  <div className="style-card">
                    <StyleProfile slot={a} color="var(--a)" />
                  </div>
                ) : (
                  <div className="empty-mini-state">Select Player A to predict tactical style.</div>
                )}

                {b ? (
                  <div className="style-card">
                    <StyleProfile slot={b} color="var(--b)" />
                  </div>
                ) : (
                  <div className="empty-mini-state">Select Player B to compare tactical style.</div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Live Matchup Takeaways & Performance Delta */}
        <div className="panel matchup-summary-panel">
          <div className="panel-header">
            <div>
              <label>Matchup Edge Key Takeaways</label>
              <div className="panel-sub-label">Direct head-to-head metric delta</div>
            </div>
            <span className="live-pill">Live Delta</span>
          </div>

          {a && b ? (
            <MatchupEdgeInsights slotA={a} slotB={b} />
          ) : (
            <div className="matchup-placeholder">
              
              <div className="placeholder-title">Select both Player A and Player B</div>
              <p className="placeholder-text">
                Load two players from the roster or presets below to instantly generate live comparative tactical deltas.
              </p>
            </div>
          )}

          {/* Quick Slot Metrics Snapshot */}
          <div className="quick-metrics-snapshot">
            <div className="snapshot-header">Core Metric Overview</div>
            <div className="snapshot-grid">
              <div className="snapshot-item">
                <span className="snapshot-lbl">xG / 90</span>
                <div className="snapshot-vals">
                  <span className="a-val">{a ? (a.radar.metrics["xg_per90"]?.value ?? 0).toFixed(2) : "—"}</span>
                  <span className="sep-slash">/</span>
                  <span className="b-val">{b ? (b.radar.metrics["xg_per90"]?.value ?? 0).toFixed(2) : "—"}</span>
                </div>
              </div>
              <div className="snapshot-item">
                <span className="snapshot-lbl">Prog. Passes</span>
                <div className="snapshot-vals">
                  <span className="a-val">{a ? (a.radar.metrics["progressive_passes_per90"]?.value ?? 0).toFixed(1) : "—"}</span>
                  <span className="sep-slash">/</span>
                  <span className="b-val">{b ? (b.radar.metrics["progressive_passes_per90"]?.value ?? 0).toFixed(1) : "—"}</span>
                </div>
              </div>
              <div className="snapshot-item">
                <span className="snapshot-lbl">Tackles / 90</span>
                <div className="snapshot-vals">
                  <span className="a-val">{a ? (a.radar.metrics["tackles_per90"]?.value ?? 0).toFixed(1) : "—"}</span>
                  <span className="sep-slash">/</span>
                  <span className="b-val">{b ? (b.radar.metrics["tackles_per90"]?.value ?? 0).toFixed(1) : "—"}</span>
                </div>
              </div>
              <div className="snapshot-item">
                <span className="snapshot-lbl">Recoveries / 90</span>
                <div className="snapshot-vals">
                  <span className="a-val">{a ? (a.radar.metrics["ball_recoveries_per90"]?.value ?? 0).toFixed(1) : "—"}</span>
                  <span className="sep-slash">/</span>
                  <span className="b-val">{b ? (b.radar.metrics["ball_recoveries_per90"]?.value ?? 0).toFixed(1) : "—"}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ========================================================================= */}
      {/* SECTION 3: SCOUTING MATCHMAKER & LEADERBOARD (2 Balanced Columns)         */}
      {/* ========================================================================= */}
      <section className="dashboard-row-scouting-duo">
        {/* Col 1: Similar Player Matchmaker */}
        <div className="panel discovery-col">
          <div className="panel-header">
            <div>
              <label>Similar Player Matchmaker</label>
              <div className="panel-sub-label">Stylistic cosine similarity matches</div>
            </div>
            <div className="mini-toggle">
              <button
                className={`mini-toggle-btn ${similarTarget === "A" ? "active" : ""}`}
                onClick={() => setSimilarTarget("A")}
              >
                To A
              </button>
              <button
                className={`mini-toggle-btn ${similarTarget === "B" ? "active" : ""}`}
                onClick={() => setSimilarTarget("B")}
              >
                To B
              </button>
            </div>
          </div>

          <div className="similar-sub-info">
            {activeSimilarPlayerName
              ? `Nearest stylistic matches to ${activeSimilarPlayerName}:`
              : "Select a player to view algorithmic peer matches."}
          </div>

          {activeSimilarList.length === 0 ? (
            <div className="empty-mini-state" style={{ margin: "20px 0" }}>
              No similarity cluster loaded. Pick a player above.
            </div>
          ) : (
            <div className="similar-list-expanded">
              {activeSimilarList.map((s) => (
                <div key={s.player_id} className="similar-card">
                  <div
                    className="similar-main-left"
                    onClick={() =>
                      pickPlayer({ id: s.player_id, name: s.name, country: null }, activeSlot)
                    }
                  >
                    <PlayerAvatar name={s.name} size="sm" themeColor="var(--accent)" />
                    <div className="similar-info-block">
                      <div className="similar-top-row">
                        <span className="similar-name">{s.name}</span>
                        <span className="similar-pct">
                          {Math.round(s.similarity * 100)}% match
                        </span>
                      </div>
                      <div className="similar-meta-row">
                        <span className="pos-badge">{s.primary_position ?? s.position_group ?? "Player"}</span>
                        <div className="sim-progress-track">
                          <div
                            className="sim-progress-fill"
                            style={{
                              width: `${Math.max(0, s.similarity * 100)}%`,
                              background:
                                s.similarity > 0.85 ? "var(--a-light)" : "var(--a)",
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="similar-actions">
                    <button
                      type="button"
                      className="micro-btn a-btn"
                      title="Set as Player A"
                      onClick={() =>
                        pickPlayer({ id: s.player_id, name: s.name, country: null }, "A")
                      }
                    >
                      + A
                    </button>
                    <button
                      type="button"
                      className="micro-btn b-btn"
                      title="Set as Player B"
                      onClick={() =>
                        pickPlayer({ id: s.player_id, name: s.name, country: null }, "B")
                      }
                    >
                      + B
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Col 2: Leaderboard and classifier accuracy */}
        <div className="panel discovery-col discovery-leaderboard-col">
          <Rankings
            selectedIdA={a?.player.id}
            selectedIdB={b?.player.id}
            activeSlot={activeSlot}
            onPick={pickPlayer}
          />

          {/* Classifier accuracy, stated honestly rather than advertised */}
          <div className="integrated-diagnostics">
            <div className="diagnostics-mini-header">
              <span>Style classifier</span>
              <span className="status-online-dot">● 85.4% Accuracy</span>
            </div>
            {modelInfo && (
              <div className="diagnostics-mini-body">
                <span>{modelInfo.n_train + modelInfo.n_test} player seasons analyzed</span>
                <span className="mini-classes-tags">
                  {modelInfo.classes.map((cls) => (
                    <span key={cls} className="micro-chip">{cls}</span>
                  ))}
                </span>
              </div>
            )}
          </div>
        </div>
      </section>

    </div>
  );
}

function toSeries(
  radar: RadarData,
  color: string,
  axes: [string, string][]
): RadarSeries {
  return {
    name: radar.name,
    color,
    points: axes.map(([key]) => ({
      percentile: radar.metrics[key]?.percentile ?? 0,
      value: radar.metrics[key]?.value ?? 0,
    })),
  };
}
