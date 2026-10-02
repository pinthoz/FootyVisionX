"use client";

// In-memory runtime cache
const photoCache = new Map<string, string | null>();

/**
 * Returns a recognizable, clean display name for a footballer.
 * Handles Spanish/Portuguese multi-surname strings (e.g. "Lionel Andrés Messi Cuccittini" -> "Lionel Messi",
 * "Neymar da Silva Santos Junior" -> "Neymar Jr", "Cristiano Ronaldo dos Santos Aveiro" -> "Cristiano Ronaldo").
 */
export function getCleanPlayerName(fullName: string): string {
  if (!fullName) return "";
  const clean = fullName.trim();
  const lower = clean.toLowerCase();

  if (lower.includes("messi")) return "Lionel Messi";
  if (lower.includes("ronaldo") && (lower.includes("cristiano") || lower.includes("aveiro"))) return "Cristiano Ronaldo";
  if (lower.includes("neymar")) return "Neymar Jr";
  if (lower.includes("iniesta")) return "Andrés Iniesta";
  if (lower.includes("griezmann")) return "Antoine Griezmann";
  if (lower.includes("casemiro") || lower.includes("casimiro")) return "Casemiro";
  if (lower.includes("bale") && lower.includes("gareth")) return "Gareth Bale";
  if (lower.includes("laporte")) return "Aymeric Laporte";
  if (lower.includes("aduriz")) return "Aritz Aduriz";
  if (lower.includes("suárez") || lower.includes("suarez")) return "Luis Suárez";
  if (lower.includes("modrić") || lower.includes("modric")) return "Luka Modrić";
  if (lower.includes("kroos")) return "Toni Kroos";
  if (lower.includes("benzema")) return "Karim Benzema";
  if (lower.includes("busquets")) return "Sergio Busquets";
  if (lower.includes("piqué") || lower.includes("pique")) return "Gerard Piqué";
  if (lower.includes("ramos") && lower.includes("sergio")) return "Sergio Ramos";
  if (lower.includes("viera") && lower.includes("jonathan")) return "Jonathan Viera";
  if (lower.includes("orellana")) return "Fabián Orellana";
  if (lower.includes("konoplyanka")) return "Yevhen Konoplyanka";
  if (lower.includes("rodríguez") && lower.includes("jesé")) return "Jesé Rodríguez";
  if (lower.includes("rodríguez") && lower.includes("james")) return "James Rodríguez";
  if (lower.includes("agudo") && lower.includes("durán")) return "Nolito";

  const parts = clean.split(/\s+/);
  if (parts.length <= 2) return clean;

  // For 3+ names: return First + Primary surname (or First + Last)
  return `${parts[0]} ${parts[parts.length - 1]}`;
}

/**
 * Clean complex player names for higher query hit rates on public sports databases.
 * e.g. "Lionel Andrés Messi Cuccittini" -> ["Lionel Messi", "Lionel Andrés Messi Cuccittini"]
 */
export function getNameVariants(fullName: string): string[] {
  const clean = fullName.trim();
  const variants = new Set<string>();
  variants.add(clean);

  const cleanDisplay = getCleanPlayerName(fullName);
  if (cleanDisplay) variants.add(cleanDisplay);

  const parts = clean.split(/\s+/);
  if (parts.length > 2) {
    variants.add(`${parts[0]} ${parts[parts.length - 1]}`);
    variants.add(`${parts[0]} ${parts[1]}`);
    variants.add(`${parts[0]} ${parts[1]} ${parts[parts.length - 1]}`);
  }

  return Array.from(variants);
}

/**
 * Resolves player photograph URL across multiple public databases:
 * 1. Wikipedia Summary API
 * 2. Wikipedia Search generator with pageimages
 * 3. TheSportsDB Open Player API
 */
export async function resolvePlayerPhoto(fullName: string): Promise<string | null> {
  if (!fullName) return null;

  const cacheKey = fullName.toLowerCase().trim();
  if (photoCache.has(cacheKey)) {
    return photoCache.get(cacheKey) ?? null;
  }

  // Check sessionStorage if in browser
  if (typeof window !== "undefined") {
    const stored = window.sessionStorage.getItem(`pv_photo_${cacheKey}`);
    if (stored) {
      photoCache.set(cacheKey, stored === "null" ? null : stored);
      return stored === "null" ? null : stored;
    }
  }

  const nameVariants = getNameVariants(fullName);

  // Strategy 1: Wikipedia Direct Summary API
  for (const variant of nameVariants) {
    try {
      const slug = encodeURIComponent(variant.replace(/\s+/g, "_"));
      const res = await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${slug}`, {
        headers: { Accept: "application/json" },
      });
      if (res.ok) {
        const data = await res.json();
        if (data.thumbnail?.source && !data.thumbnail.source.includes("Disambig")) {
          const url = data.thumbnail.source;
          cachePhoto(cacheKey, url);
          return url;
        }
      }
    } catch {
      // Continue to next variant
    }
  }

  // Strategy 2: Wikipedia Search Generator API (matches full name and extracts pageimage)
  for (const variant of nameVariants.slice(0, 2)) {
    try {
      const q = encodeURIComponent(variant);
      const res = await fetch(
        `https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch=${q}&prop=pageimages&pithumbsize=400&format=json&origin=*`
      );
      if (res.ok) {
        const data = await res.json();
        const pages = data?.query?.pages;
        if (pages) {
          const pageList = Object.values(pages) as Array<{
            index?: number;
            title?: string;
            thumbnail?: { source: string };
          }>;
          pageList.sort((a, b) => (a.index ?? 99) - (b.index ?? 99));

          for (const p of pageList) {
            if (p.thumbnail?.source && !p.thumbnail.source.includes("logo") && !p.thumbnail.source.includes("Flag")) {
              const url = p.thumbnail.source;
              cachePhoto(cacheKey, url);
              return url;
            }
          }
        }
      }
    } catch {
      // Continue to Strategy 3
    }
  }

  // Strategy 3: TheSportsDB Free Open Search
  for (const variant of nameVariants.slice(0, 2)) {
    try {
      const q = encodeURIComponent(variant);
      const res = await fetch(`https://www.thesportsdb.com/api/v1/json/3/searchplayers.php?p=${q}`);
      if (res.ok) {
        const data = await res.json();
        const player = data?.player?.[0];
        const photo = player?.strCutout || player?.strThumb || player?.strRender;
        if (photo) {
          cachePhoto(cacheKey, photo);
          return photo;
        }
      }
    } catch {
      // Continue
    }
  }

  // No photo found across all sources
  cachePhoto(cacheKey, null);
  return null;
}

function cachePhoto(key: string, url: string | null) {
  photoCache.set(key, url);
  if (typeof window !== "undefined") {
    try {
      window.sessionStorage.setItem(`pv_photo_${key}`, url ?? "null");
    } catch {
      // Storage quota safety
    }
  }
}
