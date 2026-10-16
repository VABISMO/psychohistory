# 🗺️ Roadmap & Critical Review — Psychohistory · Future Graph Dynamics

This document is an honest, critical assessment of the current engine and a
prioritised checklist of everything we should build next. It is meant to be
living: check items off as they land.

Legend: **P0** = foundational / unlocks the rest · **P1** = high value · **P2** = nice to have.

---

## 🔍 Critical review of the current model (why it is still "poor")

The present engine is an impressive *plumbing* demo — grounding, salvage parsing,
a bounded tree, a global summary — but as a *forecasting model* it is shallow.
Concretely:

1. **The "mathematics" is decorative, not predictive.** Branch probabilities are a
   monotone function of a self-assigned `magnitude`, plus hand-tuned bumps. They are
   not calibrated against anything and carry no information from data.
2. **No real time axis.** Events have a free-text `date`/`duration` for display only.
   The math never uses *when* something happens, so it cannot express "this is overdue".
3. **Events are treated as independent.** A child does not actually depend on its
   parent's outcome; the tree *looks* causal but the probabilities are computed in
   isolation. There is no conditioning (`P(child | parent)`), no coupling, no cascades.
4. **No actors.** "Heroes", institutions, coalitions and leaders — who decide and can
   single-handedly change trajectories — are absent. Only the LLM's prose mentions them.
5. **No rare-event machinery.** Improbable, high-impact events (coups, defaults,
   assassinations, pandemics) are exactly the ones that matter, and they are modelled
   with the same thin curve as everything else. No heavy tails, no hazard rates.
6. **No uncertainty quantification.** A single number per branch, no intervals, no
   ensembles, no sensitivity to parameters.
7. **No evaluation.** Nothing measures whether the forecasts are any good (Brier score,
   calibration, backtesting). Without this, every "improvement" is guesswork.
8. **Grounding is thin and untyped.** A few scraped lines injected into a prompt; no
   source citation per event, no verification, no structured numeric database.
9. **Single LLM, single shot.** No self-consistency, no ensemble, no grammar‑constrained
   decoding — which is why we needed a salvage parser in the first place.
10. **Explainability is post-hoc.** The UI shows probabilities but not *why*; the user
    cannot inspect the contribution of each factor.

The checklist below addresses all of the above.

---

## ✅ Checklist

### A. Temporal dynamics & rare events *(the "overdue black swan")* — **P0**

The core missing capability: model *time*. The user's intuition — *"the longer an
improbable event fails to occur, the more likely it becomes"* — is a **hazard rate**
with *increasing* baseline hazard (a renewal / wearing‑out process).

- [ ] Introduce an explicit continuous time axis for every event (start, end, expected onset).
- [x] Add a **hazard function** `h(t)` per event class; survival `S(t)=exp(−∫₀ᵗ h)`.
- [x] Use a **Weibull** hazard for "overdue" phenomena: `h(t) = (k/λ)(t/λ)^(k−1)` with
      `k > 1` (increasing) — probability rises the longer the event is absent. `k = 1`
      recovers the memoryless Poisson case; `k < 1` models "the risk decays with calm".
- [x] Conditional update: `P(event in (t, t+Δt] | none so far) = 1 − exp(−∫_t^{t+Δt} h)`.
- [x] **Renewal process** for recurring events (elections, cycles): track time since last
      occurrence and modulate the rate.
- [x] Self-exciting processes (**Hawkes**) for clustering: one event raises the short-term
      rate of others (protests, bank runs, conflict escalation).
- [x] **Heavy tails / black swans**: model impact (not just probability) with a fat-tailed
      distribution (Pareto / log-normal); apply **Extreme Value Theory** for the tail.
- [x] Separate **probability** from **impact severity** into two dimensions; a rare × huge
      event must be able to dominate the ranking.
- [ ] "Stress" clock per region: cumulative absence of an expected shock raises its odds.

### B. Actors & "heroes" (individual-level modelling) — **P0**

- [x] Model **key actors** (leaders, parties, generals, central bankers, CEOs) as first-class
      entities with attributes: power, tenure, popularity, ideology, constraints. *(`Actor` model + parsing)*
- [x] **Leader-departure hazard** (death/exit/resignation as a function of age, tenure,
      stress) — the "hero transition" event. *(Gompertz-Makeham + political risk)*
- [x] **Great-man influence weight**: scale an actor's leverage to shift branch probabilities
      (a single decision can move `b_same`, `b_max`, `b_min`).
- [ ] Actor → action → consequence graph, so predictions can say *"X does Y"*. *(actors attached per event, not yet a graph)*
- [x] Coalition / **game-theoretic** layer (who can form a majority, who blocks whom). *(winning coalitions + Banzhaf index)*
- [x] Succession modelling (heir presence/absence, purge cycles) — e.g. deferred-succession
      regimes. *(`succession_viability`)*
- [ ] Optional **agent-based simulation** to explore emergent outcomes.

### C. Event coupling & cascades — **P0**

- [x] Replace independent branching with **conditioned** probabilities: `P(child | parent)`.
- [x] Event dependency graph (causes, enables, prevents, correlates). *(coupling coefficients + noisy-OR/AND)*
- [x] **Contagion models** (SIR-like / network diffusion) for protests, violence, adoption.
- [ ] Feedback loops (economic ↔ political ↔ social) with resource constraints.
- [ ] Conservation/constraint checks across the whole tree (e.g. vote/seat totals, budgets). *(helpers in `coupling.py`, not yet enforced tree-wide)*

### D. Probabilistic rigour & calibration — **P1**

- [x] **Bayesian updating**: informative priors from base rates; update on new evidence. *(Beta-Bernoulli + Normal-Normal in `uncertainty.py`)*
- [x] **Monte Carlo** propagation of uncertainty; output **credible intervals**, not point values. *(per-event 90% CI + `mc_summary`/`propagate`)*
- [x] **Ensemble** over models and over LLM samples; aggregate by proper scoring rules. *(log-odds / linear pools + spread; scored by `evaluation/scoreboard`)*
- [x] **Markov chains / HMMs** for state transitions (regime shifts). *(Markov step/stationary, HMM forward + Viterbi)*
- [x] Sensitivity analysis: which parameters move the answer most. *(`one_at_a_time`)*
- [ ] Explicit, documented priors; remove magic constants (or justify them). *(priors documented in `uncertainty.py`; predictor constants still to be revisited)*

### E. Grounding & data — **P1**

- [x] Per-event **citations** (source URL + snippet) surfaced in the UI. *(`grounding.py` attribution + `Citation` model + Sources panel)*
- [x] Add sources: **GDELT**, **ACLED**, **Wikidata**, polling aggregators, IMF/World Bank,
      election databases, market/odds data. *(Wikidata entities, GDELT global news (best-effort),
      World Bank + IMF macro figures, PredictIt market/odds & election markets in `datasources.py`;
      ACLED conflict events behind optional `ACLED_API_KEY`/`ACLED_EMAIL`. Polling aggregators and
      generic election databases expose no keyless API — partly covered by Wikidata + PredictIt.)*
- [x] **Read the top result pages** (RAG over real news): Bing News/Web search → publisher URL →
      fetch → HTML→text, excerpts passed to the model as SOURCE DOCUMENTS. *(`web.py`
      `parse_bing_rss`/`fetch_page`/`extract_readable`; top 5 pages read in `search.retrieve_sources`)*
- [x] Structured **numeric constraint store** (chambers, budgets, populations, records) with
      automatic validation of every number the LLM emits. *(`constraints.py`, enforced on each event)*
- [x] Temporal grounding: resolve "next election", "in N years" to real ISO dates. *(relative refs → ISO dates in `grounding.py`)*
- [ ] Fact verification / contradiction detection across sources.
- [x] Caching & provenance for retrieved evidence. *(provenance via citations with URL + `retrieved_at`; response caching still pending)*
- [x] **Sources modal**: show *all* sources consulted for a prediction — a single panel with
      every retrieved document / citation, its snippet, URL and retrieval time, so the user can
      audit exactly what the forecast was built from. *("Sources (N)" button + modal in `prediction-graph.tsx`)*
- [x] **Add your own sources**: let the user append their own documents, URLs or notes before
      running a prediction, so the query has *all* the information they hold for that calculation.
      These become first-class evidence in grounding and citations. *(`extra_sources` → `source="USER"` citations)*

### F. LLM & inference — **P1**

- [x] **Grammar-constrained / JSON-schema decoding** (GBNF / structured outputs) so output
      is valid by construction; keep the salvage parser only as a fallback. *(`schema.py` + Ollama `format`)*
- [ ] **Self-consistency** (sample k, vote) and **multi-step / chain-of-thought** planning.
- [ ] Multi-model **ensemble** and **critique/debate** between models.
- [x] Better **RAG**: reranking, dedup, chunked evidence. *(attribution/citations pending — see section E)*
- [ ] Streaming progress to the UI (token/step level).
- [ ] Optional fine-tuning or distilling a small domain model; quantised CPU inference.

### G. Product & UX — **P2**

- [x] Timeline / chronological view alongside the graph. *(`TimelineView` toggle in `prediction-graph.tsx`)*
- [ ] Scenario **comparison** and **what-if** (perturb a node, re-run that branch).
- [x] **Export**: JSON + shareable permalinks to a prediction. *(PDF/PNG pending; JSON download + copy link + `GET /predictions/{id}`)*
- [x] **Explainability panel**: show the math contributions and cited evidence per node. *(branch, stability, override, citations presence; temporal/actors/citations already present)*
- [x] **Readable nodes at every depth**: the LLM only writes levels 1-2, so deeper nodes are
      templated from the parent's topic + a real actor with a concrete description; names are
      guaranteed unique across the tree and no node is left showing a bare actor name with no
      explanation of what happens. *(`_synthetic_event`/`_polish_event` + `_TOPIC_NAMES`/`_TOPIC_DESCS`
      in `predictor.py`; empty/duplicate/bare-name LLM events are repaired on assembly)*
- [x] Editable nodes with manual probability overrides. *(stability override slider in detail panel; applied client-side)*
- [ ] Internationalisation (the app is English-only today).
- [ ] Saved scenarios, history, user accounts.
- [ ] Accessibility pass (keyboard nav, ARIA, contrast).

### H. Engineering & quality — **P1**

- [x] **CI** (GitHub Actions): run backend tests + frontend tsc/lint/build on every pull request. *(`.github/workflows/ci.yml`, single `tests` job)*
- [ ] Frontend + integration + end‑to‑end tests (currently only backend math is tested).
- [ ] Pre-commit hooks (ruff/black, eslint/prettier).
- [ ] Structured logging + observability; request tracing.
- [ ] Robust error taxonomy and user-facing error messages.
- [ ] Security: input validation, stricter CORS, dependency scanning.
- [ ] Performance: cache retrieval, parallelise inference where possible.

### I. Evaluation — **P0** *(without this we cannot claim progress)*

- [x] Build a **backtesting harness**: run the engine on past-dated scenarios, compare to
      what actually happened. *(timing backtest in section A; general harness in `backend/evaluation`)*
- [x] Score with **Brier score**, **log loss**, **calibration curves**, rank histograms.
- [x] Track lead-time accuracy (predicted date vs actual).
- [x] Maintain a labelled dataset of resolved events for regression testing. *(30 events)*
- [x] Report a scoreboard per model/prompt version. *(`evaluation/scoreboard.py`)*

### K. Documentation & whitepaper — **P2**

- [ ] **Update the whitepaper** (`/paper`) so it reflects what is *actually implemented*.
      Its §11 "Future Work" still describes as future work several things that have since
      landed: hazard/Weibull time model, actors, event coupling, Bayesian/Monte-Carlo
      uncertainty, evaluation/backtesting, JSON-schema decoding and per-event citations.
      New sections are needed for the time axis, actors, coupling, uncertainty, evaluation,
      grounding, plus the product features (timeline, permalink, export, explainability).
      The §10 "Limitations and Ethics" disclaimer should be removed (no-disclaimers policy).
- [ ] **Rewrite the whitepaper references** — §12 currently has only 5 weak, mostly placeholder
      entries. Cite the real literature behind the model: renewal/Weibull & hazard functions,
      Hawkes self-exciting processes, Extreme Value Theory, Markov/HMMs, Brier/calibration
      scoring, JSON-schema constrained decoding, RAG and the cited data APIs.
- [ ] Periodically **re-sync `README.md`** with the code: features, endpoints, dependency/runtime
      table, test counts, project structure and license. *(first pass done: test count, license,
      endpoints, features and structure updated)*

---

## 🧭 Suggested sequencing

1. **A + I** (time/hazard model + evaluation) — makes the math meaningful and testable.
2. **C + B** (coupling + actors) — turns a tree into a causal, agentic model.
3. **F** (structured decoding + ensemble) — reliability and quality.
4. **E + G** (richer grounding + UX/explainability) — trust and usability.
5. **H** — hardening and quality.

---

*Status: living document. Start at section A — the engine is currently time-blind,
which is the single biggest gap versus the Asimovian ambition.*
