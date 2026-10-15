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
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { PredictionGraph } from "@/components/prediction-graph";
import { generatePrediction, fetchModels, type PredictionResult } from "@/lib/api";

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

  async function run() {
    if (!prompt.trim() || loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await generatePrediction(prompt, depth, model || undefined);
      setResult(res);
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
          <PredictionGraph
            nodes={result.graph_nodes}
            meta={{
              coherence: result.coherence_index,
              model: result.model_used,
              contextCount: result.context_used.length,
              summary: result.summary,
            }}
          />
        ) : (
          <EmptyState loading={loading} />
        )}
      </main>
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
