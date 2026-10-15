import Link from "next/link";
import type { Metadata } from "next";
import type { ReactNode } from "react";
import { ArrowLeft, BrainCircuit } from "lucide-react";

const YEAR = 2026;

export const metadata: Metadata = {
  title: "Whitepaper · Psychohistory · Future Graph Dynamics",
  description:
    "Technical whitepaper: architecture, live grounding and the mathematical model behind the Psychohistory forecasting engine.",
};

function Formula({ children, label }: { children: string; label?: string }) {
  return (
    <div className="avoid-break my-4 overflow-x-auto rounded-lg border border-border bg-background/60 px-4 py-3">
      <div className="whitespace-pre font-mono text-[13px] leading-relaxed text-primary">
        {children}
      </div>
      {label && (
        <div className="mt-1 text-right text-[11px] text-muted-foreground">({label})</div>
      )}
    </div>
  );
}

function Section({ n, title, children }: { n: string; title: string; children: ReactNode }) {
  return (
    <section className="mt-10 scroll-mt-6" id={`s${n}`}>
      <h2 className="mb-3 border-b border-border pb-2 text-xl font-bold tracking-tight">
        <span className="text-primary">{n}.</span> {title}
      </h2>
      <div className="space-y-3 text-sm leading-relaxed text-foreground/90">{children}</div>
    </section>
  );
}

function Code({ children }: { children: string }) {
  return (
    <code className="rounded bg-secondary px-1.5 py-0.5 font-mono text-[12px] text-primary">
      {children}
    </code>
  );
}

export default function PaperPage() {
  return (
    <div className="grid-bg min-h-screen">
      <div className="print-sheet mx-auto max-w-3xl px-5 py-10">
        <Link
          href="/"
          className="mb-8 inline-flex items-center gap-1.5 text-xs text-muted-foreground transition hover:text-foreground"
        >
          <ArrowLeft className="size-3.5" /> Back to the app
        </Link>

        <article className="rounded-2xl border border-border bg-card/80 p-8 shadow-2xl backdrop-blur-md md:p-12">
          {/* Title block */}
          <header className="mb-8 border-b border-border pb-6">
            <div className="mb-3 flex items-center gap-2 text-primary">
              <BrainCircuit className="size-5" />
              <span className="font-mono text-[11px] uppercase tracking-[0.2em]">
                Technical Whitepaper
              </span>
            </div>
            <h1 className="text-3xl font-black leading-tight tracking-tight md:text-4xl">
              Psychohistory <span className="text-primary">·</span> Future Graph Dynamics
            </h1>
            <p className="mt-2 text-base text-muted-foreground">
              A hybrid mathematical–LLM engine for grounded socio-political forecasting
            </p>
            <p className="mt-4 text-xs text-muted-foreground">
              Future Graph Dynamics · Version 1.0 · {YEAR}
            </p>
          </header>

          {/* Abstract */}
          <div className="mb-8 rounded-xl border border-primary/30 bg-primary/5 p-5">
            <p className="mb-2 font-mono text-[11px] uppercase tracking-widest text-primary">
              Abstract
            </p>
            <p className="text-sm italic leading-relaxed text-foreground/90">
              We present Psychohistory · Future Graph Dynamics, an open forecasting system
              inspired by Asimov&apos;s fictional science. The engine projects branching futures
              from any user-supplied socio-political scenario by combining a transparent
              probabilistic core with a large language model operating under strict factual
              grounding. Real news and encyclopedic sources are retrieved live from the web;
              institutional maxima, totals and records are harvested online and injected as hard
              numeric constraints, so the model can never emit impossible figures (e.g. a seat
              count exceeding a chamber&apos;s real size). The output is a convergent tree of
              events, each annotated with probability, magnitude, trend, an estimated date and
              duration, plus a global narrative summarising the most plausible future. This
              paper specifies the full architecture and every mathematical component.
            </p>
          </div>

          <Section n="1" title="Introduction">
            <p>
              Psychohistory, in Asimov&apos;s Foundation, treats the behaviour of large human
              populations as a statistical system whose aggregate dynamics can be predicted even
              though individuals cannot. The system operationalises this intuition: it does not
              attempt to divine discrete events with certainty, but to enumerate the space of
              plausible trajectories and weigh them probabilistically.
            </p>
            <p>
              The central engineering problem is <em>grounding</em>. A purely generative model
              will confidently invent impossible facts — a party winning 400 seats in a
              350-seat chamber, a percentage above 100%, a record that never existed. The system
              therefore couples two subsystems: (i) a <strong>deterministic mathematical core</strong>{" "}
              that guarantees well-defined probabilities, invariants and a bounded tree; and (ii)
              a <strong>semantic layer</strong> driven by a local LLM that fills the nodes with
              concrete, named, real-world content — always constrained by facts fetched from the
              live web.
            </p>
          </Section>

          <Section n="2" title="System Architecture">
            <p>The system is composed of four cooperating components:</p>
            <ul className="ml-4 list-disc space-y-1.5">
              <li>
                <strong>Frontend</strong> — Next.js + React Flow. A full-width prompt bar, a
                depth control, a model selector populated from the live Ollama registry, and an
                interactive graph of the forecast tree with a global-summary modal.
              </li>
              <li>
                <strong>Backend</strong> — FastAPI. Exposes <Code>/predict</Code>,{" "}
                <Code>/models</Code> and <Code>/predictions</Code>, orchestrates retrieval,
                constraint discovery, LLM inference and persistence.
              </li>
              <li>
                <strong>Inference</strong> — a local Ollama server (CPU). No data ever leaves the
                machine.
              </li>
              <li>
                <strong>Persistence</strong> — MongoDB, with an automatic in-memory{" "}
                <Code>mongomock</Code> fallback when no database is reachable.
              </li>
            </ul>
            <p>
              The request lifecycle is: retrieve real context → discover real-world bounds →
              single constrained LLM call → resilient parse → assemble the tree → attach
              mathematical indices → persist.
            </p>
          </Section>

          <Section n="3" title="Data Acquisition and Grounding">
            <p>
              Before any inference, two live web searches run concurrently. The first gathers
              <strong> situational context</strong>: recent news headlines via the Google News RSS
              endpoint and encyclopedic extracts via the Wikipedia REST API, in both Spanish and
              English. The second gathers <strong>hard bounds</strong>.
            </p>
            <p>
              Because a hypothetical future scenario (&quot;the November 2029 election&quot;) has no
              article of its own, the bound search <em>de-temporalises</em> the query by removing
              years, month names and ordinal day suffixes, so it matches the standing institutional
              article that states the real total. For the Spanish case this surfaces the fact that
              the Congress of Deputies has <strong>350 seats</strong> — a cap the model must never
              exceed.
            </p>
            <p>
              Retrieved lines are retained only if they contain a concrete number, then ranked by a
              relevance score that rewards on-topic overlap and the presence of capacity words
              (seats, total, majority, record, …):
            </p>
            <Formula label="3.1">{`score(fact) = |W(scenario) ∩ W(fact)| + 2 · |W_capacity ∩ W(fact)|`}</Formula>
            <p>
              where <Code>W(·)</Code> is the set of content words (&gt;3 characters). The highest
              scoring facts are injected into the prompt as hard caps. <strong>No bound is ever
              hardcoded</strong>: every maximum is the product of a live query derived from the
              user&apos;s own scenario.
            </p>
          </Section>

          <Section n="4" title="The Forecast Tree">
            <p>
              A forecast of depth <Code>D</Code> is a rooted tree in which every node spawns three
              children, each representing a distinct qualitative branch of the future:
            </p>
            <ul className="ml-4 list-disc space-y-1.5">
              <li>
                <strong>same</strong> — continuation of the current trend;
              </li>
              <li>
                <strong>max</strong> — the highest-probability (optimistic) variant;
              </li>
              <li>
                <strong>min</strong> — the lowest-probability (pessimistic) variant.
              </li>
            </ul>
            <p>
              Hence the branching factor is 3. The number of nodes at depth <Code>d</Code> is{" "}
              <Code>3^d</Code>, and the total size of a tree of depth <Code>D</Code> is the
              geometric sum
            </p>
            <Formula label="4.1">{`N(D) = Σ_{d=1}^{D} 3^d = (3^(D+1) − 3) / 2`}</Formula>
            <p>
              giving <Code>N(1)=3</Code>, <Code>N(2)=12</Code>, <Code>N(3)=39</Code>,{" "}
              <Code>N(4)=120</Code>. Each node carries a full event record (name, description,
              magnitude, trend, radiant flag, year, date, duration) plus a mathematical{" "}
              <em>stability index</em> λ.
            </p>
          </Section>

          <Section n="5" title="Mathematical Model">
            <p>
              The core is a fully transparent, deterministic function mapping an event&apos;s
              intrinsic properties to three calibrated branch probabilities. It relies only on the
              standard normal CDF.
            </p>

            <h3 className="mt-5 font-semibold text-foreground">5.1 The normal CDF operator</h3>
            <p>
              We use the exact error-function representation of the standard normal cumulative
              distribution:
            </p>
            <Formula label="5.1">{`Φ(x) = ½ · (1 + erf(x / √2))`}</Formula>

            <h3 className="mt-5 font-semibold text-foreground">
              5.2 Collective state and base probabilities
            </h3>
            <p>
              Each event has an impact magnitude <Code>m ∈ [0,1]</Code>. We map it to a
              &quot;collective state&quot; pressure that measures how strongly the event pushes the
              system beyond a neutral threshold. With threshold <Code>C_c = 0.5</Code>, coupling
              coefficient <Code>0.3</Code> and dispersion <Code>σ_C = 0.15</Code>:
            </p>
            <Formula label="5.2">{`C_t = 0.5 + 0.3 · m`}</Formula>
            <p>The continuation probability is the mass of the normal above the threshold:</p>
            <Formula label="5.3">{`b_same = Φ( (C_t − C_c) / σ_C )`}</Formula>
            <p>The optimistic and pessimistic branches are proportionality factors of it:</p>
            <Formula label="5.4">{`b_max = b_same · ( 0.7 + 0.2 · (1 − 0.15 · depth) )`}</Formula>
            <Formula label="5.5">{`b_min = max( 0.05 ,  b_same · ( 0.15 + 0.01 · depth ) )`}</Formula>
            <p>
              The depth term expresses the epistemic principle that uncertainty grows with the
              horizon: deeper branches are progressively less extreme in their optimistic weight.
            </p>

            <h3 className="mt-5 font-semibold text-foreground">5.3 Trend modulation</h3>
            <p>Let <Code>τ</Code> denote the event trend. The probabilities are shifted:</p>
            <div className="my-4 overflow-hidden rounded-lg border border-border">
              <table className="w-full text-[13px]">
                <thead className="bg-secondary/60 text-left text-muted-foreground">
                  <tr>
                    <th className="px-3 py-2 font-medium">τ</th>
                    <th className="px-3 py-2 font-medium">b_same</th>
                    <th className="px-3 py-2 font-medium">b_max</th>
                    <th className="px-3 py-2 font-medium">b_min</th>
                  </tr>
                </thead>
                <tbody className="font-mono text-primary">
                  <tr className="border-t border-border">
                    <td className="px-3 py-2 text-foreground">rising</td>
                    <td className="px-3 py-2">+ 0.15</td>
                    <td className="px-3 py-2">+ 0.20</td>
                    <td className="px-3 py-2">—</td>
                  </tr>
                  <tr className="border-t border-border">
                    <td className="px-3 py-2 text-foreground">declining</td>
                    <td className="px-3 py-2">− 0.10</td>
                    <td className="px-3 py-2">—</td>
                    <td className="px-3 py-2">+ 0.15</td>
                  </tr>
                  <tr className="border-t border-border">
                    <td className="px-3 py-2 text-foreground">volatile</td>
                    <td className="px-3 py-2">× 0.70</td>
                    <td className="px-3 py-2">+ 0.10</td>
                    <td className="px-3 py-2">+ 0.10</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <h3 className="mt-5 font-semibold text-foreground">5.4 Radiant reinforcement</h3>
            <p>
              A <em>radiant</em> is an event judged near-inevitable by the semantic layer. Radiants
              concentrate mass on continuation and pull mass away from the pessimistic branch:
            </p>
            <Formula label="5.6">{`b_same += 0.25 ,   b_max += 0.15 ,   b_min −= 0.10`}</Formula>

            <h3 className="mt-5 font-semibold text-foreground">5.5 Depth decay</h3>
            <p>
              Finally, every branch is attenuated by an exponential confidence decay over depth,
              so distant futures are systematically less certain:
            </p>
            <Formula label="5.7">{`δ(depth) = exp( −0.3 · depth )`}</Formula>
            <Formula label="5.8">{`b_* ← clamp01( b_* · δ(depth) )`}</Formula>
            <p>
              with <Code>clamp01(x) = min(1, max(0, x))</Code>. The triple{" "}
              <Code>(b_same, b_max, b_min)</Code> is the model&apos;s probability profile for the
              node. An invariant enforced by the construction (and covered by the test-suite) is{" "}
              <Code>b_max ≥ b_min</Code>.
            </p>

            <h3 className="mt-5 font-semibold text-foreground">5.6 Stability index</h3>
            <p>
              Each node&apos;s <strong>stability index</strong> λ is the continuation probability
              of its own event evaluated one level above its position:
            </p>
            <Formula label="5.9">{`λ(node) = b_same( magnitude, trend, radiant, depth = level − 1 )`}</Formula>

            <h3 className="mt-5 font-semibold text-foreground">5.7 Coherence index</h3>
            <p>
              The global <strong>coherence index</strong> κ is the arithmetic mean of the stability
              indices over all nodes in the tree — a single scalar summarising how self-consistent
              the forecast is:
            </p>
            <Formula label="5.10">{`κ = (1 / N) · Σ_{i=1}^{N} λ_i ,   κ ∈ [0,1]`}</Formula>

            <h3 className="mt-5 font-semibold text-foreground">5.8 Magnitude normalisation</h3>
            <p>
              Because the language model may express magnitudes on different scales, every numeric
              field is normalised onto <Code>[0,1]</Code> before use. For a raw value <Code>v</Code>:
            </p>
            <Formula label="5.11">{`ν(v) = v/10 if 1 < v ≤ 10 ;  v/100 if 10 < v ≤ 100 ;  1 if v > 100 ;  v otherwise`}</Formula>
            <p>followed by <Code>clamp01</Code>.</p>
          </Section>

          <Section n="6" title="The Semantic Layer (LLM Integration)">
            <p>
              All semantic content — names, descriptions, dates, durations, the global summary — is
              produced by a single constrained LLM call. Asking for the entire tree at once, rather
              than node-by-node, keeps inference tractable on CPU.
            </p>
            <p>The model receives a strict JSON contract:</p>
            <Formula>{`{ "summary": "...", "events": [
  { "level": 1, "name": "...", "description": "...",
    "magnitude": 0.0, "trend": "rising", "radiant": true,
    "year": 2029, "date": "Nov 2029", "duration": "6 months" }, ... ] }`}</Formula>
            <p>
              The events form a flat, ordered list: <Code>3^1</Code> entries at level 1, then{" "}
              <Code>3^2</Code> at level 2, grouped in threes by parent, each triple ordered{" "}
              [continuation, highest, lowest]. The prompt forbids generic filler, requires real
              named actors, and — critically — instructs the model to treat the retrieved
              real-world caps as inviolable, never emitting a quantity above 100% of an established
              total.
            </p>
            <p>
              Because CPU generation can be truncated mid-document, parsing is{" "}
              <strong>resilient</strong>. Rather than a single <Code>json.loads</Code>, the engine
              walks the raw text with a brace-and-string-aware scanner and recovers every balanced{" "}
              <Code>{"{…}"}</Code> object containing a <Code>name</Code> key. Complete events are
              salvaged even from an unterminated stream. Missing slots are back-filled by the
              mathematical model, so a forecast is <em>always</em> returned.
            </p>
          </Section>

          <Section n="7" title="Global Summary">
            <p>
              Alongside the tree, the model produces a plain-language{" "}
              <strong>global summary</strong>: one paragraph stating the single most plausible
              trajectory — who wins, who loses, and the key societal consequences with their dates.
              It is surfaced through a discreet <em>Summary</em> modal over the graph. When the LLM
              is unavailable, the summary is synthesised from the root events instead.
            </p>
          </Section>

          <Section n="8" title="Concurrency, Rate Limiting and Resilience">
            <p>
              The inference resource is a single CPU-bound model, so the backend protects it with a
              layered control scheme.
            </p>
            <p>
              <strong>Bottleneck (serialisation).</strong> A semaphore of capacity 1 admits one LLM
              call at a time, with a minimum inter-call interval to avoid thermal throttling.
            </p>
            <p>
              <strong>Sliding-window rate limiter.</strong> Per client key, at most <Code>M</Code>{" "}
              requests are permitted within a window of <Code>T</Code> seconds. Excess requests
              receive HTTP 429 with a computed <Code>Retry-After</Code>:
            </p>
            <Formula label="8.1">{`Retry_After = T − (t_now − t_oldest)`}</Formula>
            <p>Prediction and general endpoints use distinct budgets.</p>
            <p>
              <strong>Retry with exponential backoff.</strong> The inference call is retried on
              failure with delay <Code>base · 2^i</Code> for attempt <Code>i</Code>, so transient
              faults recover without overwhelming the model.
            </p>
          </Section>

          <Section n="9" title="Parameter Reference">
            <div className="overflow-hidden rounded-lg border border-border">
              <table className="w-full text-[13px]">
                <thead className="bg-secondary/60 text-left text-muted-foreground">
                  <tr>
                    <th className="px-3 py-2 font-medium">Symbol</th>
                    <th className="px-3 py-2 font-medium">Meaning</th>
                    <th className="px-3 py-2 font-medium">Value</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {[
                    ["C_c", "Collective threshold", "0.5"],
                    ["σ_C", "Threshold dispersion", "0.15"],
                    ["—", "Magnitude coupling", "0.3"],
                    ["δ", "Depth decay rate", "0.3"],
                    ["—", "Optimistic factor (base)", "0.70"],
                    ["—", "Pessimistic factor (base)", "0.15"],
                    ["—", "Radiant reinforcement", "+0.25 / +0.15 / −0.10"],
                    ["M", "Prediction rate limit", "6 / 60 s"],
                    ["—", "Branching factor", "3"],
                  ].map(([s, m, v]) => (
                    <tr key={m} className="border-t border-border">
                      <td className="px-3 py-2 text-primary">{s}</td>
                      <td className="px-3 py-2 font-sans text-foreground">{m}</td>
                      <td className="px-3 py-2">{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section n="10" title="Limitations and Ethics">
            <p>
              Psychohistory is fiction; this system is an exploratory reasoning aid, not an oracle.
              Probabilities are heuristic and reflect modelling choices, not measured frequencies.
              LLM narratives can inherit bias from their training data and from retrieved sources.
              The engine reduces — but cannot eliminate — hallucination through grounding and
              numeric caps. Outputs should inform human judgement, never replace it, and must not
              be used to defame real individuals or influence sensitive decisions.
            </p>
          </Section>

          <Section n="11" title="Future Work and Roadmap">
            <p>
              The current engine is deliberately minimal and fully transparent. A candid
              assessment identifies the following as the highest-value next steps, tracked in
              the project&apos;s <code className="rounded bg-secondary px-1.5 py-0.5 font-mono text-[12px] text-primary">ROADMAP.md</code>:
            </p>
            <p>
              <strong>A real time axis and hazard models.</strong> The core gap is that the
              mathematics is currently time‑blind. We should attach a continuous time axis and a{" "}
              <em>hazard rate</em> <span className="font-mono">h(t)</span> per event, so the engine
              can express &quot;overdue&quot; phenomena: the longer an improbable event fails to
              occur, the higher its conditional probability becomes. A Weibull hazard
            </p>
            <Formula label="11.1">{`h(t) = (k/λ)(t/λ)^(k−1)`}</Formula>
            <p>
              with shape <span className="font-mono">k &gt; 1</span> models an increasing hazard
              (a wearing‑out / renewal process), while <span className="font-mono">k = 1</span>{" "}
              recovers the memoryless Poisson case and <span className="font-mono">k &lt; 1</span>{" "}
              the &quot;risk decays with calm&quot; regime. Rare, high‑impact (&quot;black swan&quot;)
              events should have their impact modelled with fat‑tailed distributions and Extreme
              Value Theory, with probability and severity treated as separate axes.
            </p>
            <ul className="ml-4 list-disc space-y-1.5">
              <li>
                <strong>Actors and &quot;heroes&quot;.</strong> Model key individuals and institutions as
                first‑class entities — power, tenure, ideology — and give them a leader‑departure
                hazard and an influence weight that can shift a node&apos;s branch probabilities.
                A single decision by a pivotal actor can redefine the whole trajectory.
              </li>
              <li>
                <strong>Event coupling and cascades.</strong> Replace independent branching with
                conditioned probabilities <span className="font-mono">P(child | parent)</span> and
                contagion dynamics (self‑exciting Hawkes processes, SIR‑like diffusion) so events
                genuinely cause and amplify one another.
              </li>
              <li>
                <strong>Calibration and evaluation.</strong> Add Bayesian priors, Monte Carlo
                uncertainty intervals, and a backtesting harness scored with Brier score and
                calibration curves — without which no claim of improvement can be made.
              </li>
              <li>
                <strong>Structured decoding and grounding.</strong> Constrain the LLM with
                JSON‑schema / grammar decoding, and attach verifiable source citations to every
                event.
              </li>
            </ul>
          </Section>

          <Section n="12" title="References">
            <ol className="ml-4 list-decimal space-y-1.5 text-[13px] text-muted-foreground">
              <li>I. Asimov, <em>Foundation</em>. Gnome Press, 1951.</li>
              <li>G. Marsaglia, &quot;Evaluating the Normal Distribution,&quot; <em>J. Stat. Soft.</em>, 2004.</li>
              <li>Google News RSS Search API. news.google.com</li>
              <li>Wikimedia REST API and Action API. wikimedia.org</li>
              <li>Ollama — local large-language-model runtime. ollama.com</li>
            </ol>
          </Section>

          <footer className="mt-12 border-t border-border pt-5 text-center text-[11px] text-muted-foreground">
            © {YEAR} Psychohistory · Future Graph Dynamics · Generated locally,
            deterministically documented.
          </footer>
        </article>
      </div>
    </div>
  );
}
