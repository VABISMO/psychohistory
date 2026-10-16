"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  BrainCircuit,
  Sparkles,
  Loader2,
  GitBranch,
  Layers,
  Cpu,
  AlertCircle,
  Search,
  Plus,
  X,
  Link2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input, Textarea } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { PredictionGraph } from "@/components/prediction-graph";
import {
  generatePrediction,
  fetchModels,
  fetchPrediction,
  type PredictionResult,
  type UserSource,
} from "@/lib/api";

const EXAMPLES = [
  "US midterm elections 3 November 2026 and the road to 2028",
  "Russia–Ukraine war stalemate and post-Putin power succession",
  "China's 21st Party Congress 2027 and the succession question",
  "Iran's nuclear program after the 2026 US–Israel strikes and regime stability",
  "Israel's postwar Gaza governance and the future of Netanyahu's coalition",
  "European far-right surge and EU enlargement with Ukraine",
  "Spain snap general election 29 November 2026 and the role of Vox",
];

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [depth, setDepth] = useState(2);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [model, setModel] = useState<string>("");
  const [sources, setSources] = useState<UserSource[]>([]);
  const [showSources, setShowSources] = useState(false);

  useEffect(() => {
    fetchModels()
      .then((r) => {
        setModels(r.models);
        if (r.models.length && !model) setModel(r.models[0]);
      })
      .catch(() => {})
      .finally(() => setModelsLoaded(true));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("p");
    if (!id) return;
    let active = true;
    fetchPrediction(id)
      .then((r) => {
        if (active) setResult(r);
      })
      .catch((e) => {
        if (!active) return;
        const msg = e instanceof Error ? e.message : "Prediction not found";
        setError(
          /not found/i.test(msg)
            ? "This shared prediction is no longer available."
            : msg
        );
        setResult(null);
        // Drop the dead id so reloads and hot-reloads stop retrying it.
        if (/not found/i.test(msg)) {
          window.history.replaceState(null, "", window.location.pathname);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  async function run() {
    if (!prompt.trim() || loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await generatePrediction(prompt, depth, model || undefined, sources);
      setResult(res);
      window.history.replaceState(null, "", `?p=${res.prediction_id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid-bg flex h-screen flex-col overflow-hidden">
      {/* Header */}
      <header className="z-20 shrink-0 border-b border-border bg-background/70 backdrop-blur-xl">
        <div className="mx-auto flex w-full max-w-[1800px] items-center justify-between px-6 py-3">
          <div className="flex items-center gap-3">
            <div className="flex size-9 items-center justify-center rounded-xl bg-primary/15 text-primary glow-primary">
              <BrainCircuit className="size-5" />
            </div>
            <div>
              <h1 className="text-base font-bold leading-tight tracking-tight">
                Psychohistory <span className="text-primary">·</span> Future Graph Dynamics
              </h1>
              <p className="text-[11px] text-muted-foreground">
                Civilizational dynamics prediction · convergent graphs ·{" "}
                <Link
                  href="/paper"
                  className="text-muted-foreground/70 underline-offset-2 transition hover:text-primary hover:underline"
                >
                  whitepaper
                </Link>
              </p>
            </div>
          </div>
          <div className="hidden items-center gap-2 md:flex">
            {result && (
              <>
                <Badge variant="secondary" className="font-mono">
                  <Cpu className="size-3 mr-1" /> {result.model_used}
                </Badge>
                <Badge variant="outline" className="font-mono">
                  {countNodes(result.graph_nodes)} nodes
                </Badge>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Prompt / search bar */}
      <section className="z-10 shrink-0 border-b border-border bg-card/40 backdrop-blur-sm">
        <div className="mx-auto w-full max-w-[1800px] px-6 py-4">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
            <div className="relative flex-1">
              <Search className="pointer-events-none absolute left-3 top-3 size-4 text-muted-foreground" />
              <Textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run();
                }}
                placeholder="Describe the scenario to predict (e.g. Spain snap general election 29 November 2026)...  Cmd/Ctrl+Enter to predict"
                className="min-h-[52px] pl-9 pr-3"
              />
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2 rounded-lg border border-border bg-background/50 px-3 py-2">
                <Layers className="size-4 text-muted-foreground" />
                <span className="text-xs text-muted-foreground">Depth</span>
                <span className="w-4 text-center font-mono text-sm font-bold text-primary">
                  {depth}
                </span>
                <div className="w-24">
                  <Slider
                    value={[depth]}
                    min={1}
                    max={5}
                    step={1}
                    onValueChange={(v) => setDepth(v[0])}
                  />
                </div>
              </div>

              <div className="flex items-center gap-2 rounded-lg border border-border bg-background/50 px-3 py-2">
                <Cpu className="size-4 text-muted-foreground" />
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className="bg-transparent text-sm text-foreground outline-none"
                >
                  {models.length === 0 && (
                    <option value="">
                      {modelsLoaded ? "no local models" : "loading models..."}
                    </option>
                  )}
                  {models.map((m) => (
                    <option key={m} value={m} className="bg-card text-foreground">
                      {m}
                    </option>
                  ))}
                </select>
              </div>

              <button
                onClick={() => setShowSources(true)}
                className="flex items-center gap-2 rounded-lg border border-border bg-background/50 px-3 py-2 text-sm transition hover:bg-secondary"
                title="Add your own sources for this prediction"
              >
                <Link2 className="size-4 text-muted-foreground" />
                <span className="text-xs text-muted-foreground">Sources</span>
                {sources.length > 0 && (
                  <span className="flex size-4 items-center justify-center rounded-full bg-primary/20 font-mono text-[10px] font-bold text-primary">
                    {sources.length}
                  </span>
                )}
              </button>

              <Button onClick={run} disabled={loading || !prompt.trim()} size="lg">
                {loading ? (
                  <>
                    <Loader2 className="size-4 animate-spin" /> Computing...
                  </>
                ) : (
                  <>
                    <GitBranch className="size-4" /> Generate prediction
                  </>
                )}
              </Button>
            </div>
          </div>

          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            <Sparkles className="size-3.5 text-muted-foreground" />
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                onClick={() => setPrompt(ex)}
                className="rounded-full border border-border px-2.5 py-0.5 text-[11px] text-muted-foreground transition hover:bg-secondary hover:text-foreground"
              >
                {ex}
              </button>
            ))}
            {error && (
              <span className="ml-2 flex items-center gap-1 text-[11px] text-destructive-foreground">
                <AlertCircle className="size-3.5 text-destructive" /> {error}
              </span>
            )}
          </div>
        </div>
      </section>

      {/* Graph: fills the rest */}
      <main className="min-h-0 flex-1">
        {result ? (
          <PredictionGraph result={result} />
        ) : (
          <EmptyState loading={loading} />
        )}
      </main>

      {showSources && (
        <SourcesEditor
          sources={sources}
          onChange={setSources}
          onClose={() => setShowSources(false)}
        />
      )}
    </div>
  );
}

function SourcesEditor({
  sources,
  onChange,
  onClose,
}: {
  sources: UserSource[];
  onChange: (s: UserSource[]) => void;
  onClose: () => void;
}) {
  const [title, setTitle] = useState("");
  const [url, setUrl] = useState("");
  const [text, setText] = useState("");

  const canAdd = Boolean(title.trim() || url.trim() || text.trim());

  function add() {
    if (!canAdd) return;
    onChange([...sources, { title: title.trim(), url: url.trim(), text: text.trim() }]);
    setTitle("");
    setUrl("");
    setText("");
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-background/70 p-6 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="max-h-[85vh] w-full max-w-xl overflow-y-auto rounded-2xl border border-border bg-card/95 p-6 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 text-primary">
              <Link2 className="size-5" />
              <h2 className="text-lg font-semibold text-foreground">Add your own sources</h2>
            </div>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              Attach links, documents or notes you already have. They are treated as
              authoritative evidence for the prediction, alongside the live web search.
            </p>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="-mr-1 shrink-0 rounded-md p-1.5 text-muted-foreground transition hover:bg-secondary hover:text-foreground"
          >
            <X className="size-5" />
          </button>
        </div>

        <div className="space-y-2">
          <Input
            placeholder="Title (optional)"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />
          <Input
            placeholder="URL (optional)"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
          <Textarea
            placeholder="Notes or excerpt (optional)"
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="min-h-[70px]"
          />
          <div className="flex justify-end">
            <Button onClick={add} disabled={!canAdd} size="sm">
              <Plus className="size-4" /> Add source
            </Button>
          </div>
        </div>

        {sources.length > 0 ? (
          <ul className="mt-4 space-y-1.5">
            {sources.map((s, i) => (
              <li
                key={i}
                className="flex items-start justify-between gap-2 rounded-md border border-border bg-background/50 px-2.5 py-2 text-xs"
              >
                <div className="min-w-0">
                  <p className="truncate font-medium text-foreground">
                    {s.title || s.url || s.text.slice(0, 60)}
                  </p>
                  {s.url && (
                    <p className="truncate font-mono text-[10px] text-muted-foreground">{s.url}</p>
                  )}
                  {s.text && (
                    <p className="mt-0.5 line-clamp-2 text-[10px] text-muted-foreground">{s.text}</p>
                  )}
                </div>
                <button
                  onClick={() => onChange(sources.filter((_, j) => j !== i))}
                  aria-label="Remove source"
                  className="shrink-0 rounded-md p-1 text-muted-foreground transition hover:bg-secondary hover:text-foreground"
                >
                  <X className="size-3.5" />
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-4 text-center text-[11px] text-muted-foreground">
            No sources added yet. They will be sent with the next prediction.
          </p>
        )}
      </div>
    </div>
  );
}

function EmptyState({ loading }: { loading: boolean }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 p-10 text-center">
      {loading ? (
        <>
          <div className="relative">
            <div className="absolute inset-0 animate-ping rounded-full bg-primary/30" />
            <div className="relative flex size-16 items-center justify-center rounded-full bg-primary/15 text-primary">
              <Loader2 className="size-8 animate-spin" />
            </div>
          </div>
          <div>
            <p className="font-semibold">Retrieving context and computing trajectories...</p>
            <p className="text-sm text-muted-foreground">
              Searching news and data, generating the future tree. This can take 1-3 min on CPU.
            </p>
          </div>
        </>
      ) : (
        <>
          <div className="flex size-16 items-center justify-center rounded-2xl bg-secondary text-muted-foreground">
            <GitBranch className="size-8" />
          </div>
          <div className="max-w-md">
            <p className="font-semibold">No active prediction</p>
            <p className="text-sm text-muted-foreground">
              Type a scenario above and press <em>Generate prediction</em>. The system will
              search real news and data, then draw the future tree with its probabilities,
              trends and radiants.
            </p>
          </div>
        </>
      )}
    </div>
  );
}

function countNodes(nodes: PredictionResult["graph_nodes"]): number {
  return nodes.reduce((acc, n) => acc + 1 + countNodes(n.children), 0);
}
