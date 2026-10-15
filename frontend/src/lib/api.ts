export interface PsychEvent {
  name: string;
  description: string;
  probability: number;
  magnitude: number;
  trend: "rising" | "stable" | "declining" | "volatile";
  radiant: boolean;
  year: number;
  date: string;
  duration: string;
}

export interface PredictionNode {
  id: string;
  depth: number;
  parent_id: string | null;
  branch_type: "same" | "max" | "min";
  event: PsychEvent;
  stability_index: number;
  children: PredictionNode[];
}

export interface PredictionResult {
  prediction_id: string;
  prompt_used: string;
  prediction_depth: number;
  coherence_index: number;
  summary: string;
  graph_nodes: PredictionNode[];
  context_used: string[];
  model_used: string;
  timestamp: number;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function generatePrediction(
  prompt: string,
  depth: number,
  model?: string
): Promise<PredictionResult> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, prediction_depth: depth, model }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  const json = await res.json();
  return json.data as PredictionResult;
}

export async function fetchModels(): Promise<{ models: string[]; ollama_available: boolean }> {
  const res = await fetch(`${API_BASE}/models`);
  return res.json();
}
