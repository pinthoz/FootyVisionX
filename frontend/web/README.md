# FootyVision dashboard

Next.js (App Router) frontend for the FootyVision API — player search, scouting radars and
the RAG assistant.

## Development

The dashboard is a thin client: start the backend first.

```bash
# from the repo root
docker compose up -d db
uvicorn footyvision.api.main:app --reload      # http://localhost:8000

# then, here
npm install
npm run dev                                    # http://localhost:3000
```

Point the client at a different backend with `NEXT_PUBLIC_API_URL` (defaults to
`http://localhost:8000`):

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

## Layout

```
app/
  page.tsx              # dashboard: search, player detail, assistant
  layout.tsx            # root layout
  components/Radar.tsx  # percentile radar chart
  lib/api.ts            # typed fetch helpers for the FastAPI backend
```

## Build

```bash
npm run build && npm start
```

The whole stack also runs with `docker compose up --build` from the repo root.
