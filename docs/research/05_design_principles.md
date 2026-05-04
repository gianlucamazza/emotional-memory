# Design Principles: Affective Field Theory

> This document translates the philosophical and neuroscientific foundations into concrete
> architectural principles for the library. Each principle is traced to its theoretical sources.

---

## The Central Model: Affective Field Theory (AFT)

The model we propose is an original synthesis that transcends the limitations of all existing
frameworks. We call it **Affective Field Theory** because emotion is not a point in a space,
but a **field** — distributed, dynamic, multi-layered, permeating every cognitive operation.

No existing model (PAD, OCC, Barrett, Plutchik, Panksepp) captures the complexity of emotion in
memory on its own. AFT integrates them as distinct layers of a unified representation.

---

## The Five Layers of the Affective Field

### Layer 1 — Core Affect
**Inspiration**: Mehrabian & Russell (1974), Russell (1980), Barrett (2017)
**Structure**: Continuous coordinate `(valence: float, arousal: float, dominance: float)` in PAD space

The fundamental affective substrate — always active, like the background hum of the system.
Corresponds to Barrett's interoception: a continuous stream of valence, arousal, and dominance
before any categorization.

```python
CoreAffect:
    valence: float    # [-1.0, +1.0] from very negative to very positive
    arousal: float    # [0.0, 1.0] from completely calm to maximum excitement
    dominance: float  # [0.0, 1.0] from helpless to full control (PAD third axis)
```

**Principle**: Discrete emotion labels (joy, fear, anger) are not primitives — they are **regions
in Core Affect space**. They are computed contextually from the full PAD vector + contextual
information.

---

### Layer 2 — Affective Momentum
**Inspiration**: Spinoza (*Ethica*, III: affects as transitions), classical physics (kinematics)
**Structure**: Temporal derivatives of Core Affect

Spinoza's fundamental insight: "Laetitia est transitio hominis a minore ad maiorem perfectionem"
— joy **is** the passage, not the destination. A system in valence=+0.8 and rising is
qualitatively different from one in valence=+0.8 and falling.

```python
AffectiveMomentum:
    d_valence: float   # dv/dt — first derivative of valence
    d_arousal: float   # da/dt — first derivative of arousal
    dd_valence: float  # d²v/dt² — acceleration of valence
    dd_arousal: float  # d²a/dt² — acceleration of arousal
```

Momentum informs:
- **Prediction**: where affect will be at t+1
- **Retrieval**: memories on a similar trajectory are more congruent
- **Reconsolidation**: an abrupt change in momentum (emotional surprise) triggers the opening
  of the reconsolidation window

---

### Layer 3 — MoodField
**Inspiration**: Heidegger (*Being and Time*, §29–30) — *Stimmung* as a mode of being-in-the-world;
Bower (1981) mood-congruent memory
**Structure**: Slowly varying PAD vector — the "background tone" of the system

Heideggerian *Stimmung* is not an emotion but a **mode of being-in-the-world** — it does not
change with individual events but through accumulation. It is the gravitational field that orients
all cognitive operations. In the library this concept is embodied by `MoodField`.

```python
MoodField:
    valence: float         # [-1.0, +1.0] weighted moving average of affective history
    arousal: float         # [0.0, 1.0]
    dominance: float       # [0.0, 1.0] — PAD extension
    inertia: float         # [0.0, 1.0] resistance to change (0=volatile, 1=stable)
    timestamp: datetime    # when it was last updated
```

**Update**: Exponentially weighted moving average of Core Affect over time:
```
Mood(t) = α * CoreAffect(t) + (1-α) * Mood(t-1)
```
where `α` is small (0.05–0.15) — Mood changes slowly.

**Effect on retrieval**: Mood weights all retrieval signals — in a negative Mood state, the bias
toward negative memories increases (mood-congruent memory).

---

### Layer 4 — Appraisal Vector
**Inspiration**: Scherer CPM (2009), Lazarus (1991), Stoics (evaluation as generator of emotion)
**Structure**: Multi-component evaluation vector of an event

Emotion is not given — it is **generated** by the evaluation of the event against the system's
goals and norms. Layer 4 is the engine that produces Core Affect (Layer 1) from semantic input.

```python
AppraisalVector:
    novelty: float              # [-1.0, +1.0] from completely expected to entirely novel
    goal_relevance: float       # [-1.0, +1.0] from obstructing goals to furthering them
    coping_potential: float     # [0.0, 1.0] from no coping ability to full control
    norm_congruence: float      # [-1.0, +1.0] from norm-violating to norm-conforming
    self_relevance: float       # [0.0, 1.0] how much it concerns the system's "self"
```

**Appraisal → Core Affect mapping**:
```python
# coping_potential [0,1] is converted to signed [-1,+1]:
# coping=1 → +1 (confidence), coping=0 → -1 (helplessness)
coping_signed = 2.0 * coping_potential - 1.0

valence = 0.4*goal_relevance + 0.3*norm_congruence + 0.2*coping_signed + 0.1*novelty
arousal = 0.5*abs(novelty) + 0.3*(1-coping_potential) + 0.2*self_relevance
# coping_potential maps directly to the dominance (PAD) axis:
dominance = coping_potential   # 0 = helpless, 1 = full control
```

The mapping is configurable — this is the most empirically open component of the system.

---

### Layer 5 — Associative Resonance
**Inspiration**: Aristotle (associationism by similarity/contiguity/contrast), Hume (sympathy),
Bower (affective associative network)
**Structure**: Graph of connections between memories based on emotional resonance

When a memory is encoded or retrieved, it **resonates** with others through:
1. **Semantic similarity**: similar content
2. **Emotional congruence**: similar Core Affect at encoding time
3. **Temporal contiguity**: temporally proximate event
4. **Perceived causality**: causally connected event
5. **Emotional contrast**: affective opposition (anxiety ↔ relief)

```python
ResonanceLink:
    source_id: str
    target_id: str
    strength: float             # [0.0, 1.0]
    link_type: Literal['semantic', 'emotional', 'temporal', 'causal', 'contrastive']
```

Resonance links form emotional clusters that mutually reinforce each other during retrieval
(spreading activation).

---

## The EmotionalTag: Complete Structure

Every memory carries an `EmotionalTag` that captures all 5 layers at encoding time:

```python
class EmotionalTag(BaseModel):  # Pydantic, frozen=True
    # Layer 1: Core Affect at encoding time
    core_affect: CoreAffect                   # (valence, arousal, dominance)

    # Layer 2: Momentum at encoding time
    momentum: AffectiveMomentum               # (d_valence, d_arousal, dd_valence, dd_arousal)

    # Layer 3: Mood snapshot at encoding time
    mood_snapshot: MoodField                  # (valence, arousal, dominance, inertia, timestamp)

    # Layer 4: Appraisal that generated this emotion
    appraisal: AppraisalVector | None         # None if exogenous emotion

    # Layer 5: Resonance links with other memories
    resonance_links: list[ResonanceLink]

    # Metadata
    timestamp: datetime
    consolidation_strength: float             # [0.0, 1.0] — modulated by arousal
    last_retrieved: datetime | None
    retrieval_count: int
    reconsolidation_count: int                # number of reconsolidations that have occurred

    # Dual-path encoding (LeDoux, 1996)
    pending_appraisal: bool                   # True when fast-path; elaborate() will complete the tag

    # APE-gated reconsolidation window (Pearce & Hall, 1980)
    window_opened_at: datetime | None         # window open timestamp; None = closed

    # Pearce-Hall predictive learning
    expected_affect: CoreAffect | None        # EMA prediction of future core_affect
    prediction_learning_rate: float           # adaptive rate in [0.05, 0.80]

    # Plutchik discrete emotion label (auto_categorize)
    emotion_label: EmotionLabel | None        # None if auto_categorize=False
```

---

## Encoding Principles

### Principle 1 — Emotional Tagging Is Pervasive
*Source: Colombetti (2014), Richter-Levin (2004)*

Every memory — without exception — carries an `EmotionalTag`. There is no "emotionally neutral"
memory. If there is no explicit appraisal, the tag uses the current Core Affect and Mood as
a baseline.

### Principle 2 — Consolidation Strength Is a Function of Arousal
*Source: McGaugh (2004), Brown & Kulik (1977), Yerkes-Dodson*

```python
def consolidation_strength(arousal: float, mood_arousal: float) -> float:
    # Inverted-U relationship — non-linear
    # Peak around arousal=0.7, degradation at extreme values
    effective_arousal = 0.7 * arousal + 0.3 * mood_arousal
    return -4 * (effective_arousal - 0.7)**2 + 1.0
    # or: parametric Yerkes-Dodson curve
```

Memories with high arousal (but not maximum) have the highest consolidation_strength → slower
decay → higher retrieval priority.

### Principle 3 — Dual-Speed Encoding
*Source: LeDoux (1996), dual pathway theory*

- **Fast path**: Immediate encoding with coarse affective tag (valence + arousal from current Core
  Affect, no appraisal). Occurs within the next cycle.
- **Slow path**: Tag update with full appraisal (Layer 4) after contextual elaboration. Opens the
  reconsolidation window.

---

## Retrieval Principles

### Principle 4 — Multi-Signal Retrieval
*Source: original synthesis, Emotional RAG (2024), Bower (1981)*

```python
def retrieval_score(
    query_embedding: np.ndarray,
    query_affect: Tuple[float, float],        # current Core Affect
    current_mood: MoodField,
    current_momentum: AffectiveMomentum,
    memory: Memory,
    activation_map: dict[str, float]          # pre-computed by spreading_activation()
) -> float:

    w = adaptive_weights(current_mood)        # weights modulated by Mood

    s1 = cosine_similarity(query_embedding, memory.embedding)
    s2 = emotional_congruence(current_mood, memory.tag.mood_snapshot)
    s3 = core_affect_proximity(query_affect, memory.tag.core_affect)
    s4 = momentum_alignment(current_momentum, memory.tag.momentum)
    s5 = recency_decay(memory.tag.timestamp, memory.tag.consolidation_strength)
    s6 = activation_map.get(memory.id, 0.0)   # spreading activation boost

    return w[0]*s1 + w[1]*s2 + w[2]*s3 + w[3]*s4 + w[4]*s5 + w[5]*s6
```

### Principle 5 — Adaptive Weights Modulated by Mood
*Source: Heidegger (disclosure through mood), Bower (1981)*

The weights `w` are not fixed — they depend on the current Mood:

```python
def adaptive_weights(mood: MoodField, base: list[float], config: AdaptiveWeightsConfig) -> np.ndarray:
    # Weights are modulated with smooth sigmoid (tanh) — no hard threshold
    w = np.array(base, dtype=float)

    # Negative Mood → emotional signals dominate (s2, s3 ↑, s1 ↓)
    neg = smooth_gate(mood.valence, center=-0.3, sharpness=5.0)  # activates below center
    w[1] += config.negative_mood_strength * neg * (2/3)   # mood congruence
    w[2] += config.negative_mood_strength * neg * (1/3)   # affect proximity
    w[0] -= config.negative_mood_strength * neg           # semantic

    # High arousal → momentum alignment dominates (s4 ↑, s1 ↓)
    high_a = smooth_gate(mood.arousal, center=0.6, sharpness=5.0)  # activates above center
    w[3] += config.high_arousal_strength * high_a
    w[0] -= config.high_arousal_strength * high_a

    # Calm/neutral Mood → semantic retrieval prevails (s1 ↑, s2, s3 ↓)
    calm = smooth_gate(mood.arousal, center=0.3, sharpness=-5.0) * \
           (1.0 - abs(mood.valence))  # low arousal + neutral valence
    w[0] += config.calm_semantic_strength * calm
    w[1] -= config.calm_semantic_strength * calm * (2/3)
    w[2] -= config.calm_semantic_strength * calm * (1/3)

    w = w.clip(0.0)  # no negative weights
    total = w.sum()
    return w / total if total > 0 else np.full(len(w), 1.0 / len(w))  # uniform fallback
```

---

## Decay Principles

### Principle 6 — Non-Linear Decay Modulated by Arousal
*Source: ACT-R (Anderson, 1983), McGaugh (2004), Merleau-Ponty (habits)*

```python
def effective_decay_rate(
    base_decay: float,
    arousal_at_encoding: float,
    retrieval_count: int
) -> float:
    # Power-law base (ACT-R style)
    # Modulation by arousal: high arousal = slower decay
    arousal_factor = 1.0 - 0.5 * arousal_at_encoding

    # Retrieval reinforces memory (spacing effect)
    retrieval_factor = 1.0 / (1.0 + 0.1 * retrieval_count)

    return base_decay * arousal_factor * retrieval_factor


def consolidation_at_time_t(
    initial_strength: float,
    decay_rate: float,
    time_delta_seconds: float,
    floor: float = 0.0
) -> float:
    # Power-law decay with minimum floor for important memories
    decayed = initial_strength * (time_delta_seconds ** (-decay_rate))
    return max(decayed, floor)
```

**Minimum floor**: Memories with arousal above threshold maintain a non-zero minimum
`consolidation_strength` — they are never completely forgotten.

---

## Principle 7 — Reconsolidation at Retrieval
*Source: Nader, Schiller (2000, 2010), prediction error theory*

Retrieval does not always open the reconsolidation window — it is the **Affective Prediction Error
(APE)** that determines it. Only when the APE exceeds a configurable threshold does the window
open (`window_opened_at` is set). Any subsequent retrieval during the open window performs
reconsolidation. Tags are immutable (Pydantic `frozen=True`): `model_copy(update=...)` is used
to produce new instances.

```python
def on_retrieval(memory: Memory, current_state: AffectiveState) -> Memory:
    # Update metadata (returns new immutable instance)
    new_tag = memory.tag.model_copy(update={
        "last_retrieved": datetime.now(tz=UTC),
        "retrieval_count": memory.tag.retrieval_count + 1,
    })

    # Compute APE and update prediction (Pearce & Hall, 1980)
    ape = compute_ape(new_tag, current_state.core_affect)
    new_tag = update_prediction(new_tag, current_state.core_affect)

    # Open window only if APE exceeds threshold
    if ape >= APE_THRESHOLD and new_tag.window_opened_at is None:
        new_tag = new_tag.model_copy(update={"window_opened_at": datetime.now(tz=UTC)})

    # Reconsolidate if window is open (any retrieval within the window)
    if new_tag.window_opened_at is not None:
        alpha = min(ape * RECONSOLIDATION_LEARNING_RATE, 0.5)
        blended = CoreAffect(
            valence=(1-alpha)*new_tag.core_affect.valence + alpha*current_state.core_affect.valence,
            arousal=(1-alpha)*new_tag.core_affect.arousal + alpha*current_state.core_affect.arousal,
        )
        new_tag = new_tag.model_copy(update={
            "core_affect": blended,
            "reconsolidation_count": new_tag.reconsolidation_count + 1,
            "window_opened_at": None,  # window closes after reconsolidation
        })

    return memory.model_copy(update={"tag": new_tag})
```

---

## Principle 8 — Appraisal as Generator of Emotion
*Source: Lazarus (1991), Scherer (2009), Stoics*

Emotion is not an input — it is an **output** of the appraisal of the event against the system's
goals. The system must be able to define a **goal set** (*conatus*) that orients the appraisal.

```python
class AppraisalEngine(Protocol):
    """Duck-typed protocol — no inheritance required.

    Available implementations:
      - LLMAppraisalEngine  : uses an LLM (OpenAI-compatible) with Scherer CPM prompt
      - KeywordAppraisalEngine: keyword-based rules, no external dependency
      - StaticAppraisalEngine : returns a fixed vector (for testing)
    """

    def appraise(
        self, event_text: str, context: dict | None = None
    ) -> AppraisalVector: ...
```

---

## Principle 9 — Associative Resonance
*Source: Aristotle (De Memoria), Hume (association), Bower (spreading activation)*

At encoding time, automatically build resonance links:

```python
def build_resonance_links(
    new_memory: Memory,
    existing_memories: list[Memory],
    top_k: int = 5
) -> list[ResonanceLink]:
    links = []

    for mem in existing_memories:
        # Semantic similarity
        sem_sim = cosine_similarity(new_memory.embedding, mem.embedding)

        # Emotional congruence
        emo_sim = emotional_congruence(
            new_memory.tag.core_affect,
            mem.tag.core_affect
        )

        # Temporal contiguity
        temporal_prox = temporal_proximity(new_memory.tag.timestamp, mem.tag.timestamp)

        # Composite score
        resonance = 0.5*sem_sim + 0.3*emo_sim + 0.2*temporal_prox

        if resonance > RESONANCE_THRESHOLD:
            link_type = classify_link_type(sem_sim, emo_sim, temporal_prox)
            links.append(ResonanceLink(
                source_id=new_memory.id,
                target_id=mem.id,
                strength=resonance,
                link_type=link_type
            ))

    return sorted(links, key=lambda l: l.strength, reverse=True)[:top_k]
```

### Principle 9b — Bidirectional Links and Spreading Activation
*Source: Collins & Loftus (1975), "A spreading-activation theory of semantic processing"*

Resonance links are **bidirectional**: when memory A connects to memory B, the inverse link on B
is also created. This ensures that activation propagates in both directions in the associative
network.

During retrieval, **spreading activation** BFS multi-hop propagates activation from seed memories
toward adjacent ones: each hop multiplies the activation by the strength of the traversed link.
If two paths converge on the same memory, the maximum is taken (not the sum) to avoid artificial
inflation from the number of converging paths.

```python
def spreading_activation(
    seeds: dict[str, float],          # memory_id -> initial activation
    store: MemoryStore,
    max_hops: int = 2,
) -> dict[str, float]:                # memory_id -> max activation reached
    ...
```

### Principle 9c — Hebbian Co-Retrieval Strengthening
*Source: Hebb (1949), "The Organization of Behavior"*

> "Neurons that fire together, wire together."

Every time a group of memories is retrieved together in the same query, the existing links
between them are reinforced by a fixed increment (`hebbian_increment`, default 0.05). This models
the fact that association consolidates with use: frequently co-activated memories become
progressively more connected, emerging spontaneously as thematic or emotional clusters without
explicit supervision.

---

## System Architecture

### Memory Lifecycle

```
INPUT: event/conversation
         |
         v
    [1. Appraisal Engine]
    Compute AppraisalVector
         |
         v
    [2. Core Affect Update]
    Update the system's affective state
         |
         v
    [3. Momentum Update]
    Compute temporal derivatives
         |
         v
    [4. Mood Update]
    Weighted moving average (small α)
         |
         v
    [5. Encoding]
    Create EmotionalTag with snapshot of all 5 layers
    Compute consolidation_strength
         |
         v
    [6. Storage]
    Save in vector store with embedding + tag
         |
         v
    [7. Resonance Building]
    Build links with existing memories
         |
         v
    OUTPUT: memory with full EmotionalTag
```

### Retrieval Cycle

```
QUERY: text + current affective state
         |
         v
    [1. Query Processing]
    Embedding + current Core Affect + Mood
         |
         v
    [2. Adaptive Weights]
    Compute weights as a function of Mood
         |
         v
    [3. Multi-Signal Scoring]
    Compute score for each candidate
         |
         v
    [4. Spreading Activation]
    Multi-hop BFS from seed set (Collins & Loftus 1975)
    Build activation_map for Pass 2
         |
         v
    [5. Pass 2 Scoring + Top-K Selection]
    Re-score with resonance boost from activation_map
    Select the most relevant memories
         |
         v
    [6. Reconsolidation Check]
    For each retrieved memory: open window,
    compute APE, update tag if necessary
         |
         v
    [7. Hebbian Strengthening]
    Reinforce links between co-retrieved memories (Hebb 1949)
         |
         v
    OUTPUT: retrieved memories + updated tags
```

---

## Comparison with Neuroscience and Philosophy Principles

| Principle | Theoretical source | Implementation |
|-----------|--------------------|----------------|
| Affect as transition | Spinoza, *Ethica* III | Layer 2: Affective Momentum |
| *Stimmung* as global field | Heidegger, BT §29 | Layer 3: MoodField |
| Emotion as appraisal | Stoics, Scherer, Lazarus | Layer 4: Appraisal Vector |
| Associative resonance | Aristotle, Hume, Bower | Layer 5: Resonance Links |
| Consolidation ∝ arousal | McGaugh, LeDoux | `consolidation_strength` formula |
| Reconsolidation | Nader, Schiller | `on_retrieval()` with APE |
| Mood-congruent retrieval | Bower (1981) | `adaptive_weights()` |
| Non-linear decay | ACT-R, Yerkes-Dodson | `effective_decay_rate()` |
| Dual-speed encoding | LeDoux dual pathway | fast path + slow path |
| Pervasive tagging | Colombetti, Memory Bear AI | `EmotionalTag` on every memory |
