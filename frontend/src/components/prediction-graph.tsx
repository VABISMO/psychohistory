"use client";

import { useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
  type NodeProps,
  Handle,
  Position,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Sparkles, TrendingUp, TrendingDown, Minus, Activity, X, BookOpen, Users, Link2, Calendar, GitBranch, Download } from "lucide-react";
import { cn } from "@/lib/utils";
import type { PredictionNode, PsychEvent, TemporalInfo, Actor, Citation, PredictionResult } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";

const BRANCH_COLORS: Record<string, string> = {
  same: "#8b5cf6",
  max: "#10b981",
  min: "#f43f5e",
};

const BRANCH_LABEL: Record<string, string> = {
  same: "Continuation",
  max: "Highest probability",
  min: "Lowest probability",
};

const TREND_ICON = {
  rising: TrendingUp,
  declining: TrendingDown,
  stable: Minus,
  volatile: Activity,
};

type EventNodeData = {
  event: PsychEvent;
  branch_type: string;
  stability_index: number;
  depth: number;
};

function EventNode({ data, selected }: NodeProps) {
  const d = data as unknown as EventNodeData;
  const { event, branch_type, stability_index, depth } = d;
  const TrendIcon = TREND_ICON[event.trend] || Minus;
  const color = BRANCH_COLORS[branch_type] || "#8b5cf6";
  const sameP = Math.round(stability_index * 100);
  const maxP = Math.min(100, Math.round(stability_index * (1.1 + 0.1 * event.magnitude) * 100));
  const minP = Math.max(0, Math.round(stability_index * 0.4 * 100));

  return (
    <div
      className={cn(
        "w-72 rounded-xl border bg-card/95 px-3 py-2.5 shadow-xl backdrop-blur-md transition-all",
        selected ? "ring-2 ring-primary" : "",
        event.radiant ? "radiant-pulse border-yellow-400/60" : ""
      )}
      style={{ borderColor: event.radiant ? undefined : `${color}66` }}
    >
      <Handle type="target" position={Position.Top} className="!bg-muted-foreground/50" />
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <span
            className="size-2 rounded-full"
            style={{ background: color, boxShadow: `0 0 8px ${color}` }}
          />
          <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
            D{depth} · {branch_type}
          </span>
        </div>
        {event.radiant && (
          <span className="flex items-center gap-1 text-[10px] text-yellow-300">
            <Sparkles className="size-3" /> radiant
          </span>
        )}
      </div>

      <p className="mt-1.5 text-sm font-semibold leading-snug text-foreground">
        {event.name}
      </p>

      <p className="mt-1 line-clamp-3 text-[11px] leading-snug text-muted-foreground">
        {event.description}
      </p>

      <div className="mt-2 flex flex-wrap gap-1">
        <Badge variant={event.trend as "rising"}>
          <TrendIcon className="size-3" />
          <span className="ml-1">{event.trend}</span>
        </Badge>
        {event.date && (
          <Badge variant="outline" className="font-mono">
            {event.date}
          </Badge>
        )}
        {event.duration && (
          <Badge variant="outline" className="font-mono">
            lasts {event.duration}
          </Badge>
        )}
      </div>

      <div className="mt-2.5 grid grid-cols-3 gap-1 text-center">
        <ProbCell label="same" value={sameP} color="#8b5cf6" />
        <ProbCell label="max" value={maxP} color="#10b981" />
        <ProbCell label="min" value={minP} color="#f43f5e" />
      </div>

      <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className="h-full rounded-full"
          style={{ width: `${event.magnitude * 100}%`, background: color }}
        />
      </div>
      <p className="mt-1 font-mono text-[10px] text-muted-foreground">
        magnitude {event.magnitude.toFixed(2)}
      </p>

      <Handle type="source" position={Position.Bottom} className="!bg-muted-foreground/50" />
    </div>
  );
}

function ProbCell({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="rounded-md bg-background/60 px-1 py-1">
      <div className="text-[9px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="font-mono text-xs font-bold" style={{ color }}>
        {value}%
      </div>
    </div>
  );
}

const nodeTypes = { event: EventNode };

const V_GAP = 280;
const H_GAP = 330;

function layout(nodes: PredictionNode[], overrides?: Record<string, number>) {
  const flowNodes: Node[] = [];
  const flowEdges: Edge[] = [];
  let leafCursor = 0;

  function visit(node: PredictionNode): number {
    const childrenX: number[] = node.children.map(visit);
    let x: number;
    if (childrenX.length === 0) {
      x = leafCursor * H_GAP;
      leafCursor += 1;
    } else {
      x = (childrenX[0] + childrenX[childrenX.length - 1]) / 2;
    }

    const data: EventNodeData = {
      event: node.event,
      branch_type: node.branch_type,
      stability_index: overrides?.[node.id] ?? node.stability_index,
      depth: node.depth,
    };

    flowNodes.push({
      id: node.id,
      type: "event",
      position: { x, y: (node.depth - 1) * V_GAP },
      data,
    });

    for (const child of node.children) {
      flowEdges.push({
        id: `${node.id}-${child.id}`,
        source: node.id,
        target: child.id,
        animated: child.event.radiant,
        style: {
          stroke: BRANCH_COLORS[child.branch_type] || "#8b5cf6",
          strokeWidth: child.event.radiant ? 2.5 : 1.5,
          opacity: 0.7,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: BRANCH_COLORS[child.branch_type] || "#8b5cf6",
        },
      });
    }

    return x;
  }

  for (const root of nodes) visit(root);
  return { flowNodes, flowEdges };
}

export interface GraphMeta {
  coherence: number;
  model: string;
  contextCount: number;
  summary: string;
}

export function PredictionGraph({
  result,
}: {
  result: PredictionResult;
}) {
  const nodes = result.graph_nodes;
  const meta: GraphMeta = {
    coherence: result.coherence_index,
    model: result.model_used,
    contextCount: result.context_used.length,
    summary: result.summary,
  };
  const [selected, setSelected] = useState<PredictionNode | null>(null);
  const [showSummary, setShowSummary] = useState(false);
  const [view, setView] = useState<"graph" | "timeline">("graph");
  const [overrides, setOverrides] = useState<Record<string, number>>({});

  const { flowNodes, flowEdges } = useMemo(() => layout(nodes, overrides), [nodes, overrides]);
  const radiants = countRadiants(nodes);

  function applyOverride(id: string, p: number) {
    setOverrides((o) => ({ ...o, [id]: Math.max(0, Math.min(1, p)) }));
  }
  function resetOverride(id: string) {
    setOverrides((o) => {
      const c = { ...o };
      delete c[id];
      return c;
    });
  }

  return (
    <div className="relative h-full w-full">
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        nodeTypes={nodeTypes}
        fitView
        minZoom={0.1}
        nodesDraggable
        onNodeClick={(_, n) => setSelected(findNode(nodes, n.id))}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#334155" gap={28} size={1.5} />
        <Controls
          showInteractive={false}
          className="!bottom-4 !left-4 !flex !flex-row !overflow-hidden !rounded-lg !border !border-border !bg-card/95 !shadow-xl [&_button]:!h-9 [&_button]:!w-9 [&_button]:!border-border [&_button]:!bg-card [&_button]:!text-foreground [&_button:hover]:!bg-secondary [&_button_svg]:!fill-foreground"
        />
      </ReactFlow>

      {/* Leyenda */}
      <div className="pointer-events-none absolute left-4 top-4 rounded-lg border border-border bg-card/90 px-3 py-2 backdrop-blur-md">
        <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          Branches
        </p>
        <div className="flex flex-col gap-1">
          {(["same", "max", "min"] as const).map((b) => (
            <div key={b} className="flex items-center gap-2 text-[11px] text-foreground">
              <span
                className="size-2.5 rounded-full"
                style={{ background: BRANCH_COLORS[b] }}
              />
              {BRANCH_LABEL[b]}
            </div>
          ))}
          <div className="mt-1 flex items-center gap-2 text-[11px] text-yellow-200">
            <Sparkles className="size-3" /> Radiant
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div className="pointer-events-none absolute right-4 top-4 flex flex-col items-end gap-1.5">
        <div className="rounded-lg border border-border bg-card/90 px-3 py-2 text-right backdrop-blur-md">
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
            Coherence index
          </p>
          <p className="font-mono text-lg font-bold text-primary">
            {meta.coherence.toFixed(3)}
          </p>
        </div>
        <div className="flex gap-1.5">
          <Badge variant="secondary" className="font-mono">
            {countAll(nodes)} nodes
          </Badge>
          <Badge variant="radiant" className="font-mono">
            {radiants} radiants
          </Badge>
          <Badge variant="outline" className="font-mono">
            {meta.contextCount} sources
          </Badge>
        </div>
        <button
          onClick={() => setShowSummary(true)}
          className="pointer-events-auto mt-1 flex items-center gap-1.5 rounded-lg border border-primary/40 bg-primary/15 px-3 py-1.5 text-xs font-medium text-primary transition hover:bg-primary/25"
        >
          <BookOpen className="size-3.5" /> Summary
        </button>
      </div>

      {/* Detail panel */}
      {selected && (
        <div className="absolute bottom-4 right-4 w-96 rounded-xl border border-border bg-card/95 p-4 shadow-2xl backdrop-blur-md">
          <div className="flex items-start justify-between gap-3">
            <h3 className="pr-2 text-sm font-semibold leading-snug">{selected.event.name}</h3>
            <button
              onClick={() => setSelected(null)}
              className="text-muted-foreground transition hover:text-foreground"
            >
              <X className="size-4" />
            </button>
          </div>
          <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
            {selected.event.description}
          </p>
          <dl className="mt-3 grid grid-cols-3 gap-2 text-xs">
            <Info label="Probability" value={`${(selected.event.probability * 100).toFixed(1)}%`} />
            <Info label="Magnitude" value={selected.event.magnitude.toFixed(2)} />
            <Info label="Date" value={selected.event.date || String(selected.event.year)} />
            <Info label="Trend" value={selected.event.trend} />
            <Info label="Duration" value={selected.event.duration || "—"} />
            <Info label="Radiant" value={selected.event.radiant ? "yes" : "no"} />
            <Info label="Stability" value={selected.stability_index.toFixed(3)} />
            {selected.event.probability_ci && selected.event.probability_ci.length === 2 && (
              <Info
                label="90% CI"
                value={`${(selected.event.probability_ci[0] * 100).toFixed(0)}–${(selected.event.probability_ci[1] * 100).toFixed(0)}%`}
              />
            )}
          </dl>

          {selected.event.actors && selected.event.actors.length > 0 && (
            <ActorDetails actors={selected.event.actors} />
          )}

          {selected.event.temporal && <TemporalDetails t={selected.event.temporal} />}

          {selected.event.citations && selected.event.citations.length > 0 && (
            <CitationDetails citations={selected.event.citations} />
          )}
        </div>
      )}

      {/* Global summary modal */}
      {showSummary && (
        <div
          className="absolute inset-0 z-30 flex items-center justify-center bg-background/70 p-6 backdrop-blur-sm"
          onClick={() => setShowSummary(false)}
        >
          <div
            className="max-h-[80vh] w-full max-w-2xl overflow-y-auto rounded-2xl border border-border bg-card/95 p-6 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-3 flex items-start justify-between gap-3">
              <div className="flex items-center gap-2 text-primary">
                <BookOpen className="size-5" />
                <h2 className="text-lg font-semibold text-foreground">
                  Most plausible future
                </h2>
              </div>
              <button
                onClick={() => setShowSummary(false)}
                className="text-muted-foreground transition hover:text-foreground"
              >
                <X className="size-5" />
              </button>
            </div>
            <p className="whitespace-pre-line text-sm leading-relaxed text-muted-foreground">
              {meta.summary}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function ActorDetails({ actors }: { actors: Actor[] }) {
  return (
    <div className="mt-3 border-t border-border pt-3">
      <div className="mb-2 flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wide text-primary">
        <Users className="size-3" /> Actors
      </div>
      <ul className="space-y-1.5">
        {actors.map((a, i) => (
          <li key={i} className="rounded-md bg-background/60 px-2 py-1.5 text-xs">
            <div className="flex items-center justify-between gap-2">
              <span className="font-medium text-foreground">{a.name}</span>
              <span className="font-mono text-[10px] text-muted-foreground">
                {a.role} · {(a.influence * 100).toFixed(0)}%
              </span>
            </div>
            {(a.age != null || a.tenure_years != null) && (
              <div className="mt-0.5 text-[10px] text-muted-foreground">
                {a.age != null && <>age {a.age}</>}
                {a.age != null && a.tenure_years != null && " · "}
                {a.tenure_years != null && <>tenure {a.tenure_years}y</>}
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function CitationDetails({ citations }: { citations: Citation[] }) {
  return (
    <div className="mt-3 border-t border-border pt-3">
      <div className="mb-2 flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wide text-primary">
        <Link2 className="size-3" /> Sources
      </div>
      <ul className="space-y-1.5">
        {citations.map((c, i) => (
          <li key={i} className="rounded-md bg-background/60 px-2 py-1.5 text-xs">
            <div className="flex items-start justify-between gap-2">
              {c.url ? (
                <a
                  href={c.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-medium text-primary underline-offset-2 hover:underline"
                >
                  {c.title}
                </a>
              ) : (
                <span className="font-medium text-foreground">{c.title}</span>
              )}
              {c.source && (
                <span className="shrink-0 font-mono text-[10px] text-muted-foreground">
                  {c.source}
                </span>
              )}
            </div>
            {c.snippet && (
              <p className="mt-0.5 line-clamp-2 text-[10px] text-muted-foreground">{c.snippet}</p>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function TemporalDetails({ t }: { t: TemporalInfo }) {
  const pct = (v: number | null) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
  const num = (v: number | null, digits = 2) => (v == null ? "—" : v.toFixed(digits));
  const years = (v: number | null) => (v == null ? "—" : `${v.toFixed(1)}y`);

  const overdue = t.overdue_factor ?? 1;
  const overdueLabel =
    overdue > 1.02 ? "rising (overdue)" : overdue < 0.98 ? "cooling off" : "memoryless";
  const tailLabel =
    t.tail_alpha == null
      ? "—"
      : t.tail_alpha < 2
        ? `${num(t.tail_alpha)} (black swan)`
        : `${num(t.tail_alpha)} (moderate)`;

  return (
    <div className="mt-3 border-t border-border pt-3">
      <div className="mb-2 flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wide text-primary">
        <Activity className="size-3" /> Temporal dynamics
      </div>
      <dl className="grid grid-cols-3 gap-2 text-xs">
        <Info label="Onset (median)" value={years(t.median_onset)} />
        <Info label="Occurs by horizon" value={pct(t.occurrence_prob)} />
        <Info label="Hazard shape k" value={num(t.shape_k)} />
        <Info label="Next window" value={pct(t.conditional_next)} />
        <Info label="Overdue factor" value={`${num(overdue)} · ${overdueLabel}`} />
        <Info label="Tail index α" value={tailLabel} />
        <Info label="Severity (mean)" value={num(t.severity_mean)} />
        <Info label="Tail risk (ES)" value={num(t.tail_es)} />
        <Info label="Risk score" value={num(t.risk_score)} />
      </dl>
    </div>
  );
}

function Info({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-background/60 px-2 py-1.5">
      <dt className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</dt>
      <dd className="font-mono font-medium text-foreground">{value}</dd>
    </div>
  );
}

function findNode(nodes: PredictionNode[], id: string): PredictionNode | null {
  for (const n of nodes) {
    if (n.id === id) return n;
    const found = findNode(n.children, id);
    if (found) return found;
  }
  return null;
}

function countAll(nodes: PredictionNode[]): number {
  return nodes.reduce((acc, n) => acc + 1 + countAll(n.children), 0);
}

function countRadiants(nodes: PredictionNode[]): number {
  return nodes.reduce(
    (acc, n) => acc + (n.event.radiant ? 1 : 0) + countRadiants(n.children),
    0
  );
}
