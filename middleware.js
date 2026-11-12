// Vercel Routing Middleware: link previews for shared site links.
//
// Link-preview bots (iMessage, Slack, WhatsApp, X, LinkedIn, Discord...) don't run
// JavaScript, so a shared /?site=<id> link would otherwise show only the app's
// generic preview. For those bots this answers with a tiny page whose Open Graph
// tags carry the site's name, score and place, with a detailed satellite photo of
// the site as the image. People (and anything else) get the app as usual.
//
// The facts come from share/<shard>.json, written by `rrisk export`.

export const config = { matcher: "/" };

const BOTS =
  /bot\b|bot\/|crawler|spider|facebookexternalhit|facebookcatalog|slack|discord|whatsapp|telegram|linkedin|twitter|skypeuripreview|teams|outlook|embedly|pinterest|vkshare|redditbot|applebot|iframely|mastodon|bluesky|signal|preview/i;

const SHARDS = 256; // keep in sync with export.py and fpShard() in the page

// 32-bit FNV-1a over UTF-16 code units, as in export.shard()
function shard(id) {
  let h = 0x811c9dc5;
  for (let k = 0; k < id.length; k++) {
    h ^= id.charCodeAt(k);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h % SHARDS;
}

const ENT = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ENT[c]);

const TECH = { solar: "Solar", wind: "Wind" };

export default async function middleware(request) {
  const url = new URL(request.url);
  const id = url.searchParams.get("site");
  if (!id || !BOTS.test(request.headers.get("user-agent") || "")) return;

  let s;
  try {
    const r = await fetch(new URL(`/share/${shard(id)}.json`, url.origin));
    if (!r.ok) return;
    s = (await r.json())[id];
  } catch {
    return;
  }
  if (!s) return;

  const tech = TECH[s.tech] || s.tech || "";
  const title = `${s.name} · ${tech}${s.mw ? ` ${s.mw} MW` : ""}`;
  const desc = [
    s.score != null && s.tier
      ? `${s.tier} weather-hazard exposure (${s.score}/100 among ${String(s.tech || "").toLowerCase()} sites)`
      : null,
    s.place,
    "Renewable Site Risk Atlas",
  ]
    .filter(Boolean)
    .join(" · ");
  const image =
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export" +
    `?bbox=${s.bb.join(",")}&bboxSR=3857&imageSR=3857&size=1200,630&format=jpg&f=image`;
  const page = url.href;

  const html = `<!doctype html>
<html lang="en"><head>
<meta charset="utf-8" />
<title>${esc(title)}</title>
<meta name="description" content="${esc(desc)}" />
<meta property="og:type" content="website" />
<meta property="og:site_name" content="Renewable Site Risk Atlas" />
<meta property="og:title" content="${esc(title)}" />
<meta property="og:description" content="${esc(desc)}" />
<meta property="og:url" content="${esc(page)}" />
<meta property="og:image" content="${esc(image)}" />
<meta property="og:image:type" content="image/jpeg" />
<meta property="og:image:width" content="1200" />
<meta property="og:image:height" content="630" />
<meta property="og:image:alt" content="${esc(`Satellite photo of ${s.name}. Imagery © Esri, Maxar, Earthstar Geographics`)}" />
<meta name="twitter:card" content="summary_large_image" />
<link rel="canonical" href="${esc(page)}" />
</head><body><a href="${esc(page)}">${esc(title)}</a></body></html>`;

  return new Response(html, {
    headers: {
      "content-type": "text/html; charset=utf-8",
      "cache-control": "public, max-age=3600",
    },
  });
}
