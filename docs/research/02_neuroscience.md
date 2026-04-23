# Neuroscience and Cognitive Science of Emotional Memory

> This document collects the neuroscientific and cognitive foundations that inform the library's
> architecture. Each section ends with the operational implications for the system.

---

## 1. The Fundamental Neural Circuit: Amygdala–Hippocampus

### LeDoux's Dual Pathway

Joseph LeDoux established the neural architecture of fear memory through Pavlovian conditioning.

**Two parallel pathways**:
- **Fast path** (*low road*): Sensory thalamus → Amygdala directly. ~150–200 ms. Coarse processing,
  sufficient for immediate defensive responses before conscious awareness.
- **Slow path** (*high road*): Sensory thalamus → Sensory cortex → Amygdala. Slower but enables
  detailed, nuanced evaluation. Reduces false alarms.

The lateral nucleus of the amygdala receives convergent input from both pathways and is critical
for associating stimuli with threats.

**Implication for encoding**: The system must support dual-speed encoding:
- **Fast path**: immediate emotional tag (coarse valence + arousal) based on surface patterns
- **Slow path**: elaborated emotional tag with full post-processing appraisal

---

### Hippocampal Modulation by the Amygdala (McGaugh)

James McGaugh demonstrated experimentally that the basolateral amygdala (BLA) mediates the
modulation of memory consolidation by stress hormones.

**Mechanism**: After emotionally significant learning, extracellular norepinephrine in the
amygdala rises to ~300% above baseline and correlates with retention performance. The BLA then
modulates consolidation in downstream regions (caudate, nucleus accumbens, cortex).

**The emotional tagging hypothesis** (Richter-Levin & Akirav): The amygdala, activated by
emotional arousal, "tags" a hippocampal memory trace by reinforcing synaptic consolidation at
activated neurons.

**Arousal vs. valence**: Emotionally arousing memories depend on the amygdala-hippocampus
network; emotionally valenced but non-arousing memories depend on the prefrontal cortex-hippocampus
network (Kensinger & Corkin, 2004).

**Implication**: The `consolidation_strength` of each memory is primarily a function of
**arousal** at encoding time, not valence. High arousal → preferential consolidation →
slower decay.

---

## 2. The Somatic Marker Theory (Damasio)

Antonio Damasio proposed that emotions, experienced as bodily states ("somatic markers"), play a
critical role in decisional reasoning under conditions of complexity and uncertainty.

**Core principles**:
- Somatic markers are changes in bodily state (accelerated heartbeat, visceral sensation)
  associated with the past emotional outcomes of decisions
- They are represented and regulated primarily in the ventromedial prefrontal cortex (vmPFC)
  and the amygdala
- During decision-making, somatic markers guide choices by reactivating the emotional "tag" of
  past outcomes, consciously or unconsciously

**Iowa Gambling Task**: Patients with vmPFC damage do not develop somatic markers and make
systematically disadvantageous choices despite intact logical ability.

**Implication**: **Somatic markers** are the neuroscientific precedent for our `EmotionalTag`.
A retrieval system that uses the emotional tags of past memories as a decision-making bias
computationally implements Damasio's hypothesis.

---

## 3. Panksepp's Seven Affective Systems

Jaak Panksepp mapped seven primary emotional command systems through electrical brain stimulation,
pharmacological challenges, and lesion studies in mammalian brains.

| System | Valence | Function |
|--------|---------|----------|
| **SEEKING** (Expectancy) | Positive | Exploration, curiosity, anticipatory enthusiasm |
| **RAGE** (Anger) | Negative | Frustration, aggression when goals are blocked |
| **FEAR** (Fear) | Negative | Threat detection, freeze/flight responses |
| **LUST** (Sexual desire) | Positive | Reproductive drive |
| **CARE** (Nurturance) | Positive | Attachment, maternal behavior |
| **PANIC/GRIEF** (Panic/Grief) | Negative | Separation distress, loss of social bond |
| **PLAY** (Social play) | Positive | Social bonding, rough-and-tumble play |

Each system is subcortically generated, evolutionarily conserved across species, and evaluative
(approach for positive, avoidance for negative).

**Implication**: These 7 systems offer a taxonomy of **biologically grounded emotional primitives**
that can serve as the basis for discrete categorization, distinct from continuous coordinates
(valence, arousal). The SEEKING system corresponds functionally to Spinozian *conatus* — the
fundamental exploratory drive.

---

## 4. Memory Consolidation and Emotional Modulation

### Flashbulb Memory (Brown & Kulik, 1977)

*Flashbulb memories* are vivid, detailed, highly confident recollections of the circumstances
under which one learned a significant, emotionally intense piece of news (e.g. September 11th,
a personal traumatic event).

**Mechanism**: Amygdala activation triggers the release of stress hormones that potentiate
hippocampal consolidation — producing unusually durable memory traces.

**Critical limitation**: Despite their subjective vividness, flashbulb memories are **not
necessarily more accurate** than ordinary memories. They are more consistent and confidently
held, but subject to distortion over time (Talarico & Rubin, 2003).

**Implication for the library**: High arousal → high `consolidation_strength` → slow decay →
high retrieval priority. But not necessarily high **accuracy**. The system must not treat
high-arousal memories as ground truth.

---

### Stress Hormones and the Yerkes-Dodson Curve

**Inverted-U relationship** between stress hormone levels and mnemonic performance:
- Moderate levels of cortisol and norepinephrine **enhance** encoding and consolidation
- Extreme or chronic stress (elevated cortisol) **impairs** hippocampal function and retrieval

**Timing matters**:
- Hormones released **during or immediately after** encoding → enhance consolidation
- Stress **before retrieval** → impairs it

**Implication**: The system must model the non-linear relationship between arousal and
`consolidation_strength`. A very high arousal value does not necessarily produce maximum
consolidation strength.

---

### Reconsolidation (Nader, Schiller)

Karim Nader, Glenn Schafe, and Daniela Schiller demonstrated that reactivated memories
temporarily become labile and must be "reconsolidated" to persist.

**Fundamental finding**: When a consolidated memory is reactivated (retrieved), it enters
a labile state for a limited time window. During this window:
- The memory can be **weakened** (amnesic agents such as propranolol or anisomycin)
- The memory can be **updated** with new information
- The **emotional tag** can be modified

**Prediction error as trigger**: Prediction error (mismatch between expectation and experience)
appears to be necessary to trigger reconsolidation.

**Clinical application**: Fear memories can be modified during reconsolidation through behavioral
interventions (extinction during the reconsolidation window) or pharmacological agents.

**Critical implication for the library**: Retrieval is not a read-only operation — the **Affective
Prediction Error (APE)** determines whether the emotional tag becomes labile. Only when the APE
exceeds a configurable threshold does the reconsolidation window open (`window_opened_at`), during
which the memory's `EmotionalTag` can be updated. This APE-gated model is also inspired by
**Pearce & Hall (1980)**: associability increases after high prediction errors, and decreases
after accurate predictions.

---

## 5. Cognitive Appraisal Theories

### Lazarus — Primary and Secondary Appraisal (1966, 1991)

Richard Lazarus proposed that emotions arise not from events themselves but from the **cognitive
evaluation** of those events.

**Two-phase appraisal**:
1. **Primary appraisal**: Is this event relevant to me? Is it threatening, beneficial, or
   irrelevant?
2. **Secondary appraisal**: Can I cope? What resources do I have?

The combination determines the specific emotion generated (e.g. threat + low coping =
fear/anxiety; threat + high coping = challenge).

---

### Scherer — Component Process Model (CPM)

Klaus Scherer expanded appraisal theory into a sequential multi-component process model.

**Five Stimulus Evaluation Checks (SECs)**:
1. **Novelty**: Is something new or unexpected happening?
2. **Intrinsic pleasantness**: Is it intrinsically pleasant or unpleasant?
3. **Goal significance**: Does it further or obstruct my goals?
4. **Coping potential**: Can I handle it?
5. **Normative significance**: Is it consistent with my norms and self-concept?

Emotion is defined as the synchronization of five components: cognitive appraisal, physiological
arousal, motor expression, action tendency, and subjective feeling.

**Implication for Layer 4**: Our **Appraisal Vector** implements a computational version of
Scherer's SECs. The five appraisal dimensions in our model (novelty, goal_relevance,
coping_potential, norm_congruence, self_relevance) derive directly from this framework.

---

## 6. The Theory of Constructed Emotion (Barrett)

Lisa Feldman Barrett challenges the classical view of basic emotions with a constructionist account.

**Core claims**:
- Emotions are not hard-wired, universal categories triggered by dedicated brain circuits.
  Instead, they are **constructed** in the moment by the brain's predictive machinery.
- The brain continuously generates predictions about incoming sensory signals based on
  **interoception** (internal bodily sensing), prior experience, and culturally learned
  emotional concepts.
- **Core affect** — a continuous stream of valence (pleasant/unpleasant) and arousal
  (activated/deactivated) — is the fundamental building block. Discrete emotions (anger,
  fear, etc.) emerge when the brain categorizes core affect using conceptual knowledge.

**Implication**: There is no single brain signature for any emotion; emotions emerge from
domain-general brain networks. For the library: discrete emotion labels (joy, fear, anger)
are **emergent patterns** in the continuous affective space, not hard-wired primitives.
This justifies a hybrid approach: continuous coordinates + contextual categorization.

---

## 7. Mood-Congruent Memory and State-Dependent Learning

### Mood-Congruent Memory (Bower, 1981)

People preferentially encode and retrieve information that matches their current mood state.

**Theoretical basis**: Gordon Bower's **Associative Network Theory** — affect exists as a central
node in an associative memory network. When a mood is active, it spreads activation to
mood-congruent nodes, facilitating encoding and retrieval of matching material.

**Empirical evidence**: Happy individuals better recall positive material; sad individuals better
recall negative material. MCM effects are more robust for positive than negative moods.

**Implication**: The `emotional_congruence` signal in multi-signal retrieval must measure the
match between the system's current Mood and the memory's `mood_snapshot` at encoding time.

---

### State-Dependent Learning (Eich, 1989)

Retrieval is enhanced when the internal state (mood, physiological arousal, pharmacological state)
at retrieval matches the state during encoding, independently of the emotional valence of
the material.

Distinct from mood-congruent memory: state-dependent memory concerns the **match between
encoding and retrieval contexts**, not the match between mood and material valence.

**Implication**: The Mood at encoding time must be captured as part of the tag (`mood_snapshot`).
Retrieval with a Mood very different from the `mood_snapshot` may penalize the score of an
otherwise relevant memory.

---

## 8. Prediction Error and Affective Learning

### Dopaminergic Reward Prediction Error (Schultz, 1997)

The dopaminergic reward prediction error (RPE) signal, characterized by Wolfram Schultz, is a
central mechanism linking emotion to learning:
- Dopaminergic neurons encode the **difference between expected and received reward**
- Positive RPE (better than expected) → dopamine increase → reinforces behavior
- Negative RPE (worse than expected) → dopamine suppression → guides adjustment

### Separable Neural Signals for RPE and Affective Prediction Error (2025)

Recent research (Nature Communications, 2025) has identified **separable** neural signals for RPE
and **affective prediction errors** — deviations from emotional expectations that drive learning
independently of reward value, through striatum-amygdala interactions.

**Implication for reconsolidation**: When a memory is retrieved and the current event generates a
significant affective prediction error (reality differs from the emotional expectation encoded in
the memory), the reconsolidation window opens. This is the computational mechanism that triggers
the update of the `EmotionalTag`.

---

## 9. Recent Developments (2020–2026)

### Generalizable Neural Signatures of Emotional Memory (2025)

A preprint (bioRxiv, 2025) identified cross-individual patterns of brain activity that predict
emotional memory formation — potentially opening the path toward biomarkers for successful
emotional encoding.

### Affective Computing: State of the Art 2026

The systematic review by Fang et al. (2025) demonstrates that multimodal fusion (facial
expression + speech prosody + text + physiological signals) outperforms unimodal models.
Cross-modal transformers dominate.

**The MemEmo gap (2026)**: The MemEmo benchmark demonstrated that no current LLM memory system
correctly handles emotion across extraction, updating, and question-answering. This is the gap
our library aims to address.

---

## Synthesis: From Neuron to Architecture

| Neuroscientific phenomenon | Parameter in the library |
|----------------------------|--------------------------|
| Amygdala activation → hippocampal tagging | `consolidation_strength = f(arousal_at_encoding)` |
| LeDoux dual pathway (fast/slow) | Fast encoding (coarse) + Slow encoding (full appraisal) |
| Yerkes-Dodson curve | Non-linear `consolidation_strength` with respect to arousal |
| Reconsolidation at retrieval | `EmotionalTag` update window post-retrieval |
| Somatic Marker Hypothesis | `EmotionalTag` as decisional navigation bias |
| Mood-Congruent Memory | `emotional_congruence` weight in retrieval scoring |
| State-Dependent Learning | `mood_snapshot` in `EmotionalTag` + match at retrieval |
| Affective Prediction Error | Trigger for opening the reconsolidation window |
| Panksepp 7 systems | Possible basis for discrete emotion categories |
| Barrett constructionism | Discrete labels as emergent patterns in continuous space |
| Scherer's SECs | Layer 4: Appraisal Vector (5 dimensions) |

---

## Bibliographic Notes

- LeDoux, J.E. (1996). *The Emotional Brain*. Simon & Schuster.
- LeDoux, J.E. (2000). "Emotion circuits in the brain." *Annual Review of Neuroscience*, 23, 155-184.
- McGaugh, J.L. (2004). "The amygdala modulates the consolidation of memories of emotionally
  arousing experiences." *Annual Review of Neuroscience*.
- McGaugh, J.L. (2013). "Making lasting memories: Remembering the significant." *PNAS*.
- Kensinger, E.A. & Corkin, S. (2004). "Two routes to emotional memory: Distinct neural processes
  for valence and arousal." *PNAS*.
- Richter-Levin, G. (2004). "The Amygdala, the Hippocampus, and Emotional Modulation of Memory."
  *The Neuroscientist*.
- Damasio, A.R. (1994). *Descartes' Error: Emotion, Reason, and the Human Brain*. Putnam.
- Damasio, A.R. (1996). "The somatic marker hypothesis and the possible functions of the prefrontal
  cortex." *Phil. Trans. Royal Society B*.
- Panksepp, J. (1998). *Affective Neuroscience: The Foundations of Human and Animal Emotions*.
  Oxford University Press.
- Brown, R. & Kulik, J. (1977). "Flashbulb memories." *Cognition*, 5(1), 73-99.
- Talarico, J.M. & Rubin, D.C. (2003). "Confidence, not consistency, characterizes flashbulb
  memories." *Psychological Science*, 14(5), 455-461.
- Nader, K., Schafe, G.E. & LeDoux, J.E. (2000). "Fear memories require protein synthesis in
  the amygdala for reconsolidation after retrieval." *Nature*, 406, 722-726.
- Schiller, D. et al. (2010). "Preventing the return of fear in humans using reconsolidation
  update mechanisms." *Nature*, 463, 49-53.
- Lee, J.L.C., Nader, K. & Schiller, D. (2017). "An update on memory reconsolidation updating."
  *Trends in Cognitive Sciences*, 21(7), 531-545.
- Lazarus, R.S. (1991). *Emotion and Adaptation*. Oxford University Press.
- Scherer, K.R. (2009). "The dynamic architecture of emotion: Evidence for the component process
  model." *Cognition and Emotion*.
- Barrett, L.F. (2017). *How Emotions Are Made: The Secret Life of the Brain*. Houghton Mifflin
  Harcourt.
- Barrett, L.F. (2017). "The theory of constructed emotion: an active inference account of
  interoception and categorization." *Social Cognitive and Affective Neuroscience*, 12(1), 1-23.
- Bower, G.H. (1981). "Mood and memory." *American Psychologist*, 36(2), 129-148.
- Eich, E. & Metcalfe, J. (1989). "Mood dependent memory for internal versus external events."
  *Journal of Experimental Psychology: Learning, Memory, and Cognition*.
- Schultz, W., Dayan, P. & Montague, P.R. (1997). "A neural substrate of prediction and reward."
  *Science*, 275, 1593-1599.
- Nature Communications (2025). "Separable neural signals for reward and emotion prediction errors."
