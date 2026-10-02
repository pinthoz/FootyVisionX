"""A documentation page that looks like the product it documents.

Swagger UI is generated from the OpenAPI schema and arrives in its default white, which
next to a dark green dashboard reads as a different application entirely — and the docs
page is often the first thing anyone sees of an API. The schema is left alone; only the
presentation is replaced, so every endpoint stays as discoverable and as executable as it
was, still generated rather than written by hand and therefore never out of date.

The palette is lifted from the frontend's own tokens (`globals.css`): a pitch is green,
its markings are matte chalk, and the single accent is the gold used for player A.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, Response

# Fetched from a CDN by Swagger UI itself; pinned so the page cannot change under us.
SWAGGER_JS = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui-bundle.js"
SWAGGER_CSS = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui.css"

# Grouping the tags gives the page a shape the flat list does not have: what the API is
# *for* reads off the section headings rather than out of twenty endpoint names.
SECTIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "Players",
        "Who is in the database, and how they compare within their position.",
        ("players", "similarity", "metrics"),
    ),
    (
        "Models",
        "Performance score, the position classifiers, team attack/defence, and market "
        "value — the last one a range rather than a price, and reported with what it is "
        "worth against a do-nothing baseline.",
        ("talent", "teams", "value"),
    ),
    (
        "Search & language",
        "Structured filters, natural-language queries, and the retrieval assistant.",
        ("search", "assistant", "reports"),
    ),
    (
        "Provenance",
        "What the data covers and how the assistant scored when it was evaluated.",
        ("coverage", "evaluation", "health", "meta"),
    ),
)

_THEME = """
@import url("https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap");

:root {
  --bg: #0a1710;
  --bg-subtle: #0d1d14; --panel: #10241a; --panel-2: #16301f;
  --chalk: #e9f0e6; --border: rgba(233,240,230,0.13);
  --border-soft: rgba(233,240,230,0.07); --text: #eef3ea;
  --text-dim: #bccbb9; --muted: #8ca08c; --accent: #b88e2d; --accent-light: #d9ab48;
  --pink: #e0607e; --radius: 4px;
  --sans: Inter, system-ui, -apple-system, sans-serif;
  --head: Archivo, Inter, system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
/* The ground and its mowing bands, copied from the dashboard rather than reinvented: a
   chalk wash at 1.6% in 72px columns over the flat green, fixed so it does not travel
   with the scroll. */
body { background: var(--bg);
  background-image: repeating-linear-gradient(90deg,
    rgba(233,240,230,0.016) 0px, rgba(233,240,230,0.016) 72px,
    transparent 72px, transparent 144px);
  background-attachment: fixed;
  margin: 0; font-family: var(--sans); }

/* The markings the dashboard draws behind itself: centre circle, halfway line, and the
   gold centre spot. The app hangs them on a real .pitch-markings div; this page is
   generated, so there is no element to hang them on and they go on the two pseudo-
   elements instead -- at z-index 0, with the header and Swagger itself at 1. */
body::before {
  content: ""; position: fixed; top: 50%; left: 50%; z-index: 0; pointer-events: none;
  width: min(48vmin, 560px); aspect-ratio: 1; transform: translate(-50%, -50%);
  border: 3px solid rgba(255,255,255,0.03); border-radius: 50%;
}
body::after {
  content: ""; position: fixed; top: 50%; left: 0; right: 0; height: 20px;
  z-index: 0; pointer-events: none; transform: translateY(-50%);
  /* Spot first so it paints over the line it sits on; gold is the one marked point. */
  background:
    radial-gradient(circle at 50% 50%, rgba(217,171,72,0.1) 0 8px, transparent 8.5px),
    linear-gradient(to bottom,
      transparent calc(50% - 1.5px), rgba(255,255,255,0.03) calc(50% - 1.5px),
      rgba(255,255,255,0.03) calc(50% + 1.5px), transparent calc(50% + 1.5px));
}

/* --- The map above the reference ---------------------------------------------------- */
.fv-head { position: relative; z-index: 1; max-width: 1160px;
  margin: 0 auto; padding: 52px 24px 10px; }
.fv-title { font-family: var(--head); font-size: 32px; font-weight: 700;
  letter-spacing: -0.02em; color: var(--text); margin: 0; }
.fv-title span { color: var(--accent-light); }
.fv-sub { color: var(--text-dim); font-size: 13.5px; margin: 10px 0 0; max-width: 62ch;
  line-height: 1.6; }
.fv-meta { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.fv-pill { border: 1px solid var(--border); background: rgba(255,255,255,0.03);
  border-radius: var(--radius); padding: 4px 10px; font-size: 11px; color: var(--muted);
  text-decoration: none; }
.fv-pill b { color: var(--text-dim); font-weight: 600; font-family: var(--mono); }
a.fv-pill:hover { border-color: rgba(184,142,45,0.5); color: var(--text-dim); }

.fv-map { display: grid; gap: 10px; margin-top: 22px;
  grid-template-columns: repeat(auto-fit, minmax(232px, 1fr)); }
.fv-card { border: 1px solid var(--border); background: var(--panel);
  border-radius: var(--radius); padding: 14px 16px; display: flex; flex-direction: column; }
.fv-card h4 { margin: 0 0 5px; font-size: 12.5px; color: var(--accent-light);
  letter-spacing: 0.03em; text-transform: uppercase; font-family: var(--head); }
.fv-card p { margin: 0 0 10px; font-size: 11.5px; color: var(--muted); line-height: 1.55; }
.fv-card .fv-links { margin-top: auto; }
.fv-card a { display: inline-block; margin: 0 5px 5px 0; font-size: 10.5px;
  color: var(--text-dim); border: 1px solid var(--border); border-radius: 3px;
  padding: 2px 7px; text-decoration: none; font-family: var(--mono); }
.fv-card a:hover { color: var(--text); border-color: rgba(184,142,45,0.5);
  background: rgba(184,142,45,0.07); }
.fv-count { margin: 9px 0 0 !important; font-size: 10.5px; color: var(--muted);
  font-family: var(--mono); }

/* --- Swagger UI, restyled rather than rebuilt ---------------------------------------- */
.swagger-ui { position: relative; z-index: 1; font-family: var(--sans); }
.swagger-ui .topbar, .swagger-ui .information-container { display: none; }
.swagger-ui .wrapper { max-width: 1160px; padding: 0 24px; }
.swagger-ui, .swagger-ui .info .title, .swagger-ui .opblock-tag,
.swagger-ui .opblock .opblock-summary-operation-id,
.swagger-ui .opblock .opblock-summary-path,
.swagger-ui .opblock .opblock-summary-path__deprecated,
.swagger-ui .parameter__name, .swagger-ui .response-col_status,
.swagger-ui .opblock-title_normal, .swagger-ui .tab li,
.swagger-ui label, .swagger-ui .parameter__extension { color: var(--text); }
.swagger-ui .opblock-description-wrapper p, .swagger-ui .renderedMarkdown p,
.swagger-ui .parameter__type, .swagger-ui .response-col_links,
.swagger-ui .opblock-summary-description,
.swagger-ui table thead tr td { color: var(--text-dim); }
.swagger-ui .parameter__in, .swagger-ui .parameter__extension { font-family: var(--mono);
  color: var(--muted); }
.swagger-ui .opblock .opblock-summary-path, .swagger-ui .response-col_status,
.swagger-ui .opblock-summary-operation-id { font-family: var(--mono); }
.swagger-ui hr { border-color: var(--border-soft); }

/* Servers / authorize strip, and the filter box. */
.swagger-ui .scheme-container { background: var(--panel); box-shadow: none;
  border: 1px solid var(--border); border-radius: var(--radius);
  padding: 14px 18px; margin: 6px 0 24px; }
.swagger-ui .scheme-container .schemes-title, .swagger-ui .servers-title { color: var(--text-dim); }
.swagger-ui .filter .operation-filter-input { background: var(--bg-subtle);
  border: 1px solid var(--border); color: var(--text); border-radius: var(--radius); }

/* Tag sections. */
.swagger-ui .opblock-tag { border-bottom: 1px solid var(--border); font-size: 15px;
  font-family: var(--head); letter-spacing: 0.01em; padding: 14px 0 10px; margin: 24px 0 10px; }
.swagger-ui .opblock-tag small { color: var(--muted); font-family: var(--sans); }
.swagger-ui .opblock-tag:hover { background: rgba(255,255,255,0.02); }

/* Operations. */
.swagger-ui .opblock { background: var(--panel); border-radius: var(--radius);
  border: 1px solid var(--border); box-shadow: none; margin: 0 0 9px; }
.swagger-ui .opblock .opblock-summary { border-color: var(--border-soft); padding: 8px 12px; }
.swagger-ui .opblock .opblock-summary-method { border-radius: 3px; font-family: var(--head);
  font-size: 12px; text-shadow: none; box-shadow: none; min-width: 74px; }
.swagger-ui .opblock.opblock-get { border-color: rgba(184,142,45,0.34);
  background: rgba(184,142,45,0.05); }
.swagger-ui .opblock.opblock-get .opblock-summary-method { background: var(--accent);
  color: #14210f; }
.swagger-ui .opblock.opblock-post { border-color: rgba(224,96,126,0.34);
  background: rgba(224,96,126,0.05); }
.swagger-ui .opblock.opblock-post .opblock-summary-method { background: var(--pink);
  color: #1a0a0f; }
.swagger-ui .opblock.is-open { background: var(--panel); }
.swagger-ui .opblock-body, .swagger-ui .opblock-section-header,
.swagger-ui .opblock .opblock-section-header { background: var(--bg-subtle);
  border-color: var(--border-soft); box-shadow: none; }
.swagger-ui .opblock-section-header h4, .swagger-ui .opblock-section-header > label,
.swagger-ui .tab li button.tablinks, .swagger-ui .btn { color: var(--text); }
.swagger-ui .opblock-section-header h4 { font-family: var(--head); font-size: 13px; }
.swagger-ui .responses-inner { padding: 18px 20px; }
/* The labels an executed request prints -- Curl, Request URL, Server response, Response
   body -- are bare h4/h5s Swagger leaves at its default navy, unreadable on the pitch. */
.swagger-ui .responses-inner h4, .swagger-ui .responses-inner h5 { color: var(--text-dim);
  font-family: var(--sans); font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.05em; }
.swagger-ui .servers h4.message, .swagger-ui .operation-servers h4.message
  { color: var(--muted); font-family: var(--sans); }

/* Swagger reaches for the method colour and for green on a handful of controls, through
   selectors more specific than the ones above. These are those selectors. */
.swagger-ui .opblock.opblock-get > .opblock-summary,
.swagger-ui .opblock.opblock-post > .opblock-summary { border-color: var(--border-soft); }
.swagger-ui .opblock .opblock-section-header h4,
.swagger-ui .opblock .opblock-section-header > label { color: var(--text);
  font-family: var(--head); font-size: 13px; }
.swagger-ui .opblock .opblock-summary-description { color: var(--text-dim);
  font-family: var(--sans); }
.swagger-ui .opblock .opblock-title_normal p,
.swagger-ui .opblock-description-wrapper p { color: var(--text-dim); }
.swagger-ui .response-control-media-type--accept-controller select
  { border-color: rgba(184,142,45,0.6); }
.swagger-ui .response-control-media-type__accept-message { color: var(--muted); }
/* The active tab's underline is drawn per method, in the method's own blue or green.
   Matching the attribute keeps one rule covering every verb. */
.swagger-ui .opblock[class*="opblock-"] .tab-header .tab-item.active h4 span::after
  { background: var(--accent); }

.swagger-ui .response-col_status { color: var(--accent-light); }

/* Controls. */
.swagger-ui .btn { background: var(--panel-2); border-color: var(--border);
  box-shadow: none; border-radius: 3px; font-family: var(--sans); }
.swagger-ui .btn:hover { border-color: rgba(184,142,45,0.55); }
.swagger-ui .btn.execute { background: var(--accent); border-color: var(--accent);
  color: #14210f; font-weight: 600; }
.swagger-ui .btn.execute:hover { background: var(--accent-light); }
.swagger-ui .btn.cancel { background: transparent; border-color: rgba(224,96,126,0.5);
  color: var(--pink); }
.swagger-ui .btn.authorize { color: var(--accent-light); border-color: rgba(184,142,45,0.5); }
.swagger-ui .btn.authorize svg { fill: var(--accent-light); }
.swagger-ui .copy-to-clipboard { background: var(--panel-2); border-radius: 3px; }
.swagger-ui .download-contents { background: var(--panel-2); color: var(--text); }
.swagger-ui .tab li button.tablinks { opacity: 0.6; font-family: var(--mono); font-size: 11px; }
.swagger-ui .tab li.active button.tablinks { opacity: 1; color: var(--accent-light); }

/* Fields. */
.swagger-ui input, .swagger-ui textarea, .swagger-ui select {
  background: var(--bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 3px; font-family: var(--mono); }
.swagger-ui input:focus, .swagger-ui textarea:focus, .swagger-ui select:focus {
  outline: none; border-color: rgba(184,142,45,0.6); }
.swagger-ui input[type=text].invalid, .swagger-ui textarea.invalid {
  background: rgba(224,96,126,0.12); border-color: var(--pink); }
.swagger-ui .microlight { background: #08120c !important; border-radius: 3px;
  font-family: var(--mono) !important; }
.swagger-ui .highlight-code > .microlight code { color: #cfe0cb !important;
  font-family: var(--mono) !important; }
.swagger-ui table thead tr th, .swagger-ui table thead tr td {
  color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
  border-bottom: 1px solid var(--border-soft); font-family: var(--sans); }

/* Dialogs and errors, which ship as white cards. */
.swagger-ui .dialog-ux .modal-ux { background: var(--panel); border-color: var(--border);
  border-radius: var(--radius); box-shadow: 0 18px 50px rgba(0,0,0,0.5); }
.swagger-ui .dialog-ux .modal-ux-header { border-color: var(--border); }
.swagger-ui .dialog-ux .modal-ux-header h3, .swagger-ui .dialog-ux .modal-ux-content h4,
.swagger-ui .dialog-ux .modal-ux-content p, .swagger-ui .dialog-ux .modal-ux-content code
  { color: var(--text); }
.swagger-ui .dialog-ux .backdrop-ux { background: rgba(4,10,7,0.72); }
.swagger-ui .errors-wrapper { background: rgba(224,96,126,0.08);
  border-color: rgba(224,96,126,0.4); border-radius: var(--radius); }
.swagger-ui .errors-wrapper .errors h4, .swagger-ui .errors-wrapper hgroup h4,
.swagger-ui .errors-wrapper .errors small { color: var(--text); }

/* Swagger draws several glyphs as inline SVG or as a background image with a hardcoded
   dark fill, invisible on a dark ground. Inverting them is cheaper and safer than
   overriding every icon. */
.swagger-ui svg.arrow, .swagger-ui .expand-methods svg, .swagger-ui .expand-operation svg,
.swagger-ui .opblock-summary-control svg,
.swagger-ui .copy-to-clipboard svg,
.swagger-ui .authorization__btn svg { filter: invert(0.88); }

/* --- Schemas -------------------------------------------------------------------------
   The generated schemas section is the weakest part of the default page: an unlabelled
   list of names in a white well, each field a line of prose. Because the API is OpenAPI
   3.1, Swagger renders it with its JSON-Schema-2020-12 component, whose markup shares no
   class names with the older `.model` tree -- these are the classes the page actually
   uses. Each schema is given the card treatment the operations above it get, a line
   saying what the section is for, and a monospace column so a field name and its type
   line up. The same component draws the request and response schemas inside an
   operation, so styling it here makes both read alike. */
.swagger-ui section.models { background: transparent; border: 0;
  border-top: 1px solid var(--border); border-radius: 0; margin: 40px 0 64px; padding: 0; }
.swagger-ui section.models.is-open h4 { border-bottom: 0; margin: 0; }
.swagger-ui section.models h4 { font-family: var(--head); font-size: 12.5px;
  text-transform: uppercase; letter-spacing: 0.08em; color: var(--accent-light);
  padding: 20px 0 8px; border: 0; }
.swagger-ui section.models h4 button, .swagger-ui section.models .models-control
  { color: inherit; background: transparent; }
.swagger-ui section.models h4 svg { fill: var(--accent-light); }

/* A line of orientation the generated page does not carry. */
.swagger-ui section.models.is-open > div:first-of-type::before {
  content: "{schemas-note}";
  display: block; max-width: 70ch; margin: 2px 0 18px; color: var(--muted);
  font-size: 12px; line-height: 1.6; }

/* One card per schema. Scoped to the top level so the same component stays compact when
   it is drawn inside an operation's request or response body. */
.swagger-ui section.models > div > article.json-schema-2020-12 {
  background: var(--panel); border: 1px solid var(--border-soft);
  border-radius: var(--radius); margin: 0 0 8px; padding: 3px 15px 5px;
  transition: border-color 0.12s ease; }
.swagger-ui section.models > div > article.json-schema-2020-12:hover
  { border-color: var(--border); }
.swagger-ui section.models > div > article > .json-schema-2020-12-head
  > .json-schema-2020-12-accordion .json-schema-2020-12__title { font-size: 13.5px; }

/* The head row: name, expand-all, type. */
.swagger-ui .json-schema-2020-12-head { display: flex; align-items: center; gap: 9px;
  padding: 7px 0; flex-wrap: wrap; }
.swagger-ui .json-schema-2020-12__title { font-family: var(--mono); font-size: 12.5px;
  font-weight: 500; color: var(--text); background: none; }
.swagger-ui .json-schema-2020-12-accordion { background: transparent; border: 0;
  padding: 0; display: flex; align-items: center; gap: 4px; }
.swagger-ui .json-schema-2020-12-accordion:hover .json-schema-2020-12__title
  { color: var(--accent-light); }
.swagger-ui .json-schema-2020-12-accordion__icon { color: var(--muted); }
.swagger-ui .json-schema-2020-12-accordion__icon svg { fill: var(--muted); }
.swagger-ui .json-schema-2020-12-expand-deep-button { background: transparent;
  border: 1px solid var(--border-soft); border-radius: 3px; color: var(--muted);
  font-family: var(--sans); font-size: 9.5px; text-transform: uppercase;
  letter-spacing: 0.06em; padding: 2px 7px; }
.swagger-ui .json-schema-2020-12-expand-deep-button:hover { color: var(--text-dim);
  border-color: var(--border); }

/* The type, and any constraint attached to it. */
.swagger-ui .json-schema-2020-12__attribute { font-family: var(--mono); font-size: 11px;
  font-weight: 400; }
.swagger-ui .json-schema-2020-12__attribute--primary { color: var(--accent-light); }
.swagger-ui .json-schema-2020-12__attribute--muted { color: var(--muted); }
.swagger-ui .json-schema-2020-12__constraint { background: rgba(184,142,45,0.12);
  color: var(--accent-light); border-radius: 3px; font-family: var(--mono);
  font-size: 10.5px; padding: 1px 6px; }

/* Properties, indented off a chalk hairline rather than off whitespace alone. */
.swagger-ui .json-schema-2020-12--embedded { background: transparent; border: 0;
  padding: 0; }
.swagger-ui .json-schema-2020-12-body { padding-left: 0; }
.swagger-ui .json-schema-2020-12-keyword--properties > ul { list-style: none;
  margin: 0 0 6px; padding: 0; }
.swagger-ui .json-schema-2020-12-property { border-left: 1px solid var(--border-soft);
  margin: 0 0 0 3px; padding: 0 0 0 14px; }
.swagger-ui .json-schema-2020-12-property > article > .json-schema-2020-12-head
  { padding: 4px 0; }
.swagger-ui .json-schema-2020-12-property .json-schema-2020-12__title { color: var(--text);
  font-family: var(--mono); font-weight: 500; }
.swagger-ui .json-schema-2020-12-property--required
  > .json-schema-2020-12:first-of-type > .json-schema-2020-12-head
  .json-schema-2020-12__title::after { color: var(--pink); }

/* Keywords: description, default, example, enum. */
.swagger-ui .json-schema-2020-12-keyword__name { font-family: var(--mono);
  font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.05em; }
.swagger-ui .json-schema-2020-12-keyword__name--primary { color: var(--text-dim); }
.swagger-ui .json-schema-2020-12-keyword__name--secondary { color: var(--muted); }
.swagger-ui .json-schema-2020-12-keyword__value { font-family: var(--mono);
  font-size: 11px; color: var(--muted); }
.swagger-ui .json-schema-2020-12-keyword__value--primary { color: var(--accent-light); }
.swagger-ui .json-schema-2020-12-keyword__value--warning { color: var(--pink); }
.swagger-ui .json-schema-2020-12__description,
.swagger-ui .json-schema-2020-12 .renderedMarkdown p { font-family: var(--sans);
  color: var(--muted); font-size: 12px; line-height: 1.55; margin: 2px 0 4px; }

/* A dark page deserves a dark scrollbar; the default light one cuts the page in half. */
* { scrollbar-color: rgba(233,240,230,0.18) transparent; scrollbar-width: thin; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: rgba(233,240,230,0.16); border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: rgba(233,240,230,0.28); }
::-webkit-scrollbar-track { background: transparent; }
"""

# The one sentence of prose in the stylesheet, substituted in rather than written inline:
# a CSS string cannot be wrapped, and this one is longer than a line.
_THEME = _THEME.replace(
    "{schemas-note}",
    "Every request body and response payload the endpoints above refer to. Each field "
    "carries its type, whether it is required, and the description written on the "
    "Pydantic model.",
)


def install(app: FastAPI) -> None:
    """Replace the default `/docs` with the themed one. Call after the routers are added."""

    @app.get("/static/docs.css", include_in_schema=False)
    def docs_css() -> Response:
        return Response(_THEME, media_type="text/css")

    @app.get("/docs", include_in_schema=False)
    def docs() -> HTMLResponse:
        swagger = get_swagger_ui_html(
            openapi_url=app.openapi_url or "/openapi.json",
            title=f"{app.title} — API",
            swagger_js_url=SWAGGER_JS,
            swagger_css_url=SWAGGER_CSS,
        )
        body = swagger.body.decode()
        header = _header(app)
        # Injected into the generated page rather than templated over it: Swagger UI mounts
        # itself into #swagger-ui, and anything placed before that div survives the mount.
        body = body.replace(
            '<div id="swagger-ui">',
            f'<link rel="stylesheet" href="/static/docs.css">{header}<div id="swagger-ui">',
        )
        return HTMLResponse(body)


def _header(app: FastAPI) -> str:
    """The map above the reference: what this API is for, before what it exposes."""
    paths = app.openapi().get("paths", {})
    operations = {
        tag: [
            f"{method.upper()} {path}"
            for path, methods in paths.items()
            for method, operation in methods.items()
            if tag in operation.get("tags", [])
        ]
        for tag in {t for m in paths.values() for o in m.values() for t in o.get("tags", [])}
    }
    endpoints = sum(len(v) for v in operations.values())

    cards = []
    for title, blurb, tags in SECTIONS:
        links = "".join(f'<a href="#/{tag}">{tag}</a>' for tag in tags if operations.get(tag))
        count = sum(len(operations.get(tag, [])) for tag in tags)
        cards.append(
            f'<div class="fv-card"><h4>{title}</h4><p>{blurb}</p>'
            f'<div class="fv-links">{links}</div>'
            f'<p class="fv-count">{count} endpoints</p></div>'
        )

    return f"""
    <div class="fv-head">
      <h1 class="fv-title">Footy<span>Vision</span> API</h1>
      <p class="fv-sub">{app.description}</p>
      <div class="fv-meta">
        <span class="fv-pill"><b>v{app.version}</b></span>
        <span class="fv-pill"><b>{endpoints}</b> endpoints</span>
        <span class="fv-pill">schema: <b>/openapi.json</b></span>
        <span class="fv-pill">alternative reference: <b>/redoc</b></span>
        <a class="fv-pill" href="#" onclick="document.querySelector('section.models')
          .scrollIntoView({{behavior:'smooth'}});return false">jump to <b>schemas</b></a>
      </div>
      <div class="fv-map">{"".join(cards)}</div>
    </div>
    """
