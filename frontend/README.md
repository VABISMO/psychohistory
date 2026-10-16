# Psychohistory — Frontend

Next.js **16** frontend for **Psychohistory · Future Graph Dynamics** — an interactive
prediction graph (React Flow) over the FastAPI backend.

See the [root README](../README.md) for the full project overview, architecture,
mathematics and setup.

## Develop

```bash
npm install
npm run dev -- --port 3000
```

Requires the backend running at <http://localhost:8000> (see [`../backend`](../backend)).

## Build

```bash
npm run build
npm start -- --port 3000
```

## Checks

```bash
npx tsc --noEmit   # type-check
npm run lint       # eslint
```

The same checks run on every pull request via [`../.github/workflows/ci.yml`](../.github/workflows/ci.yml).
