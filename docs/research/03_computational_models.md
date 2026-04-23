# Computational Models of Emotion

> Overview of the main computational models of emotion, with a critical analysis of their
> strengths, limitations, and relevance to our "Affective Field Theory" approach.

---

## 1. Dimensional Models

### Russell — Circumplex Model of Affect (1980)

James Russell proposed that all affective states can be mapped onto a two-dimensional circular
space:
- **X axis**: Valence (pleasure — displeasure)
- **Y axis**: Arousal (activation — deactivation)

Labels for discrete emotions (happy, sad, angry, calm, etc.) are distributed around this circle
as **regions** in continuous space, not as discrete points.

```
                    HIGH AROUSAL
                         |
         excited         |        alarmed
    elated     -------- +A --------     afraid
  happy                  |                 annoyed
         --------- +V ---+--- -V ---------
  content                |                  miserable
    serene     -------- -A --------     sad
         calm            |        depressed
                    LOW AROUSAL
```

**Strengths**:
- Continuous: no artificial discretization
- Well validated empirically across cultures
- Computationally efficient (2 coordinates)
- Captures the fundamental structure of affect

**Limitations**:
- Lacks the Dominance dimension (anger and fear have similar valence/arousal but opposite
  dominance)
- Does not capture dynamics (it is a state model, not a process model)
- Some complex emotions are difficult to position

**Relevance to the library**: The circumplex provides the substrate for **Layer 1 — Core Affect**.
Every affective state of the system has coordinates (valence, arousal) that position it in the
circumplex.

---

### Mehrabian & Russell — PAD Model (1974)

Extends the circumplex to three dimensions:
- **Pleasure** (pleasant — unpleasant): ≈ valence
- **Arousal** (excited — calm): ≈ arousal
- **Dominance** (dominant/in-control — submissive/controlled)

The third dimension **Dominance** distinguishes emotions that are conflated in the 2D circumplex:
- **Anger**: high arousal, low valence, **high dominance**
- **Fear**: high arousal, low valence, **low dominance**

The PAD model claims these three dimensions are necessary and sufficient to characterize any
emotional state.

**Empirical validation**: Mapping hundreds of emotion terms from 50+ languages onto these three
axes shows significant cross-cultural consistency.

**Relevance**: The PAD model is the natural candidate for **Core Affect** representation in our
architecture — three continuous axes covering the affective space with sufficient granularity
without over-engineering.

---

## 2. Evolutionary Discrete Models

### Plutchik — Wheel of Emotions (1980)

Robert Plutchik proposed a hybrid evolutionary-dimensional model.

**Structure**:
- **8 primary emotions** in opposing pairs: Joy/Sadness, Trust/Disgust, Fear/Anger,
  Surprise/Anticipation
- Arranged in a **3D cone** with **intensity** on the vertical axis (e.g. annoyance → anger →
  rage)
- **Dyads** (blending): adjacent emotions combine (e.g. joy + trust = love; fear + surprise = awe)

```
                    Ecstasy
                 Joy       Admiration
           Serenity           Trust
      Anticipation               Acceptance
   Vigilance                           Apprehension
        Anger                       Fear
             Disgust           Surprise
                  Loathing   Amazement
                      Grief
                        Sadness
                          Pensiveness
```

**Evolutionary function of each emotion**:
- Fear → Protection
- Anger → Destruction of barriers
- Joy → Reproduction
- Sadness → Reintegration
- Trust → Affiliation
- Disgust → Rejection
- Surprise → Orientation
- Anticipation → Exploration

**Strengths**:
- Supports emotional blending
- Intensity dimension
- Evolutionary grounding
- Intuitive and visually communicable

**Limitations**:
- The choice of 8 primitives is arbitrary
- Blending is additive; it does not capture the complexity of emotional emergence

**Relevance**: The Wheel provides a natural taxonomy for the **discrete emotion categories** that
emerge from the continuous PAD space. The regions of Plutchik's cone can be mapped onto PAD space,
providing human-readable labels for numerical coordinates.

---

### Panksepp — 7 Primary Affective Systems (1998)

The seven emotional command systems identified by Panksepp through experimental neurobiology
(see `02_neuroscience.md` for details):

| System | Approx. PAD | Behavior |
|--------|-------------|----------|
| SEEKING | P+, A+, D+ | Exploration, curiosity |
| RAGE | P-, A+, D+ | Aggression, frustration |
| FEAR | P-, A+, D- | Avoidance, freeze/flight |
| LUST | P+, A+, D+ | Sexual behavior |
| CARE | P+, A-, D+ | Nurturance, attachment |
| PANIC/GRIEF | P-, A-, D- | Isolation, crying |
| PLAY | P+, A+, D- | Social bonding, play |

**Relevance**: These 7 systems can function as **primary clusters** in PAD space — regions around
which basic emotional responses organize. Biologically more robust than Plutchik's Wheel, but less
granular for linguistic use.

---

## 3. Cognitive and Appraisal Models

### OCC Model — Ortony, Clore & Collins (1988)

The most influential computational model of emotion in symbolic AI. Generates 22 emotion types
from appraisals of three categories:

- **Events** → pleased/displeased (e.g. joy, distress, hope, fear, satisfaction, disappointment)
- **Agents** (actions) → approving/disapproving (e.g. pride, shame, admiration, reproach)
- **Objects** → liking/disliking (e.g. love, hate)

Compound emotions emerge from combinations (e.g. gratification = joy + pride; remorse = distress +
shame).

```
                          EMOTIONS
                         /    |    \
                      Events Agents Objects
                      /  \    / \    / \
                pleased disp. app. repr. like dislike
                  |       |
           hope distress joy fear ...
```

**Strengths**:
- Rule-based → interpretable and transparent
- Compositional structure
- De facto standard for virtual agents (NPCs in video games, social chatbots)

**Limitations**:
- Discrete → loses the continuity of emotional experience
- Rule-based → rigid, does not learn
- Does not model temporal dynamics
- Difficult to scale to open-ended scenarios

**Relevance**: OCC provides a structured **vocabulary of emotion labels** that can be used as a
semantic layer on top of PAD coordinates. An event with (valence=+0.7, arousal=0.4) in response
to a goal-furthering event → OCC label "joy" or "satisfaction".

---

### Scherer — Component Process Model (CPM)

(Detailed in `02_neuroscience.md` — section 5)

The CPM is the richest among appraisal models. The sequence of 5 SECs (Stimulus Evaluation
Checks) generates emotional states in a processual way.

**Computational format** (original Scherer CPM):
```
appraisal_result = {
    novelty_check: float,          # 0=completely expected, 1=entirely novel
    intrinsic_pleasantness: float, # -1=unpleasant, +1=pleasant
    goal_significance: float,      # -1=obstructs, +1=furthers
    coping_potential: float,       # 0=no coping, 1=full control
    norm_compatibility: float      # -1=violates norms, +1=conforms
}
```

These 5 values determine the resulting emotional state through compositional rules.

**Relevance**: The CPM is the model that most directly informs **Layer 4 — Appraisal Vector**
in our architecture.

**Implementation note**: In the library, `intrinsic_pleasantness` is replaced by `self_relevance`
(how much the event concerns the system's "self", `[0.0, 1.0]`). This deviation from the original
CPM is intentional: `self_relevance` is more operationalizable in an LLM agent and contributes to
arousal (not to valence), preserving the 5 SECs while redistributing their roles.

---

## 4. Architectural Cognitive Models

### ACT-R — Adaptive Control of Thought—Rational (Anderson, 1983+)

ACT-R is a unified cognitive architecture. Retrieval is governed by activation levels:

```
Ai = Bi + Σ(Wj * Sji) + εi
```

Where:
- `Bi` = base-level activation (recency and frequency with power-law decay)
- `Σ(Wj * Sji)` = spreading context activation
- `εi` = Gaussian noise

**Base-level decay**:
```
Bi = ln(Σ(t_j^(-d)))
```
Where `t_j` are the elapsed times since each access and `d` is the decay parameter.

**Emotional extensions of ACT-R**: Researchers have proposed extensions where:
- Emotion modifies the `d` parameter (decay) — emotionally intense memories decay more slowly
- Emotion modifies `Wj` — the current emotional context gives greater weight to emotionally
  congruent memories

**Relevance**: ACT-R's base-level activation mechanism is the foundation for our differentiated
decay system. We adopt the power law but add modulation by arousal.

---

### SOAR with Emotion (Marinier, Laird & Lewis, 2009)

"A Computational Unification of Cognitive Behavior and Emotion" integrates appraisal theory into
SOAR. Emotional evaluations serve as **intrinsic rewards for reinforcement learning**.

Phases:
- Relevance appraisal during Perceive/Encode
- Evaluative appraisal during Comprehend
- Outcome probability during Intend

**Relevance**: The **emotion-as-intrinsic-reward** principle is applicable to our architecture:
emotions not only tag memories but also modulate the reinforcement of system behavior.

---

### CLARION — Motivational Subsystem (Sun, 2016)

CLARION includes an explicit motivational subsystem with:
- **Low-level drives** (survival, curiosity)
- **High-level drives** (purpose, focus, adaptation)

Emotion is generated by drive activations and action potentials, with appraisal and metacognition
as secondary factors.

**Relevance**: The **Spinozian conatus** in our architecture corresponds to CLARION's drives —
the system's goals that generate the evaluative plane of appraisal.

---

## 5. Recent Neural-Symbolic Models (2023–2026)

### Chain-of-Emotion Architecture (2024)

Architecture for affective LLM agents in video games. Uses a separate LLM call for emotional
appraisal before every response:
```
prompt = system_instruction + message_history + emotion_history + user_input
```
Evaluated as significantly more natural and emotionally sensitive than controls (PLOS One, 2024).

**Relevance**: The **two-pass** structure (first appraisal, then response) is the pattern we
adopt in our appraisal engine.

### Emotional RAG (Huang et al., 2024)

Two mood-dependent retrieval strategies:
1. **Combination**: joint weighting of semantics + emotional state
2. **Sequential**: filtering by emotion then semantic ranking (or vice versa)

Outperforms standard RAG on role-playing datasets.

**Relevance**: Confirms the utility of the `emotional_congruence` signal in retrieval scoring.
Our approach extends this with momentum, the Mood field, and spreading activation.

### Dynamic Affective Memory Management (2025)

Bayesian memory updating with entropy minimization. Uses **memory entropy** as a quality metric —
the system minimizes global entropy through Bayesian updates to the affective memory vector.

**Relevance**: The **entropy-as-memory-quality** principle is promising for our consolidation
system. More precise memories (low entropy) should have higher `consolidation_strength`.

---

## 6. Comparative Analysis: Why No Existing Model Is Sufficient

| Model | Static/Dynamic | Continuous/Discrete | Appraisal | Temporal | Global |
|-------|----------------|---------------------|-----------|----------|--------|
| Russell Circumplex | Static | Continuous | No | No | No |
| PAD | Static | Continuous | No | No | No |
| OCC | Process | Discrete | Yes | No | No |
| Plutchik Wheel | Static | Hybrid | No | Yes (intensity) | No |
| Panksepp 7 | Static | Discrete | No | No | No |
| Barrett Core Affect | Static | Continuous | Partial | No | No |
| CPM (Scherer) | Process | Continuous | Yes | Partial | No |
| ACT-R ext. | Dynamic | Continuous | No | Yes | No |
| Emotional RAG | Process | Hybrid | No | No | No |

**Systematic gaps identified**:
1. **No model** captures the **temporal dynamics** of affect (where you are going, not just where
   you are)
2. **No model** has a **global field** (mood field) that colors all operations
3. **No model** integrates **appraisal + memory + retrieval** in a unified framework
4. **No model** implements **reconsolidation** as a native operation

These four gaps motivate our **Affective Field Theory**.

---

## Bibliographic Notes

- Russell, J.A. (1980). "A circumplex model of affect." *Journal of Personality and Social
  Psychology*, 39, 1161-1178.
- Mehrabian, A. & Russell, J.A. (1974). *An Approach to Environmental Psychology*. MIT Press.
- Plutchik, R. (1980). *Emotion: A Psychoevolutionary Synthesis*. Harper & Row.
- Panksepp, J. (1998). *Affective Neuroscience*. Oxford University Press.
- Ortony, A., Clore, G.L. & Collins, A. (1988). *The Cognitive Structure of Emotions*. Cambridge
  University Press.
- Anderson, J.R. et al. (2004). "An integrated theory of the mind." *Psychological Review*,
  111(4), 1036-1060.
- Marinier, R.P., Laird, J.E. & Lewis, R.L. (2009). "A computational unification of cognitive
  behavior and emotion." *Cognitive Systems Research*.
- Sun, R. (2016). *Anatomy of the Mind*. Oxford University Press.
- Huang et al. (2024). "Emotional RAG: Enhancing Role-Playing Agents through Emotional
  Retrieval." arXiv:2410.23041.
- Zhang & Zhong (2025). "Decoding Emotion in the Deep." arXiv:2510.04064.
- Dynamic Affective Memory Management (2025). arXiv:2510.27418.
