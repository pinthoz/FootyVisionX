import { DARK_THEME, type UITheme } from "@withqwerty/campos-react";

// The library's dark theme is a blue-grey, so its charts render as a foreign card inside
// a pitch-green panel. Only the surfaces, text and borders need overriding for them to
// sit in the page rather than on top of it; everything else is inherited.
//
// Literal hex rather than the CSS variables used elsewhere: these values are handed to
// the library, which computes contrast from them and cannot resolve var().
export const PITCH_CHART_THEME: UITheme = {
  ...DARK_THEME,
  surface: {
    ...DARK_THEME.surface,
    frame: "transparent",
    plot: "#0d1d14",
    tooltip: "#16301f",
    badge: "#16301f",
  },
  text: {
    ...DARK_THEME.text,
    primary: "#eef3ea",
    secondary: "#bccbb9",
    muted: "#8ca08c",
  },
  border: {
    ...DARK_THEME.border,
    subtle: "rgba(233, 240, 230, 0.13)",
    tooltip: "rgba(233, 240, 230, 0.13)",
  },
};
