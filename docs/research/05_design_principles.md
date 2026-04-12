# Principi di Design: Affective Field Theory

> Questo documento traduce le fondazioni filosofiche e neuroscientifiche in principi architetturali concreti per la libreria. Ogni principio è tracciato alle sue sorgenti teoriche.

---

## Il Modello Centrale: Affective Field Theory (AFT)

Il modello che proponiamo è una sintesi originale che supera i limiti di tutti i framework esistenti. Lo chiamiamo **Affective Field Theory** perché l'emozione non è un punto in uno spazio, ma un **campo** — distribuito, dinamico, multi-strato, che permea ogni operazione cognitiva.

Nessun modello esistente (PAD, OCC, Barrett, Plutchik, Panksepp) cattura da solo la complessità dell'emozione nella memoria. L'AFT li integra come layer distinti di una rappresentazione unificata.

---

## I Cinque Layer dell'Affective Field

### Layer 1 — Core Affect
**Ispirazione**: Barrett (2017), Russell (1980)
**Struttura**: Coordinata continua `(valence: float, arousal: float)` nello spazio circumplex

Il substrato affettivo fondamentale — sempre attivo, come il rumore di fondo del sistema. Corrisponde all'interocezione di Barrett: uno stream continuo di valenza e arousal prima di qualsiasi categorizzazione.

```python
CoreAffect:
    valence: float   # [-1.0, +1.0] da molto negativo a molto positivo
    arousal: float   # [0.0, 1.0] da completamente calmo a massima eccitazione
```

**Principio**: Le etichette emotive discrete (joy, fear, anger) non sono primitive — sono **regioni nello spazio del Core Affect**. Si calcolano contestualmente dal vettore PAD completo + informazioni contestuali.

---

### Layer 2 — Affective Momentum
**Ispirazione**: Spinoza (*Ethica*, III: affetti come transizioni), fisica classica (cinematica)
**Struttura**: Derivate temporali del Core Affect

L'insight fondamentale di Spinoza: "Laetitia est transitio hominis a minore ad maiorem perfectionem" — la gioia **è** il passaggio, non la destinazione. Un sistema che si trova in valence=+0.8 in crescita è qualitativamente diverso da uno in valence=+0.8 in diminuzione.

```python
AffectiveMomentum:
    velocity: Tuple[float, float]      # (dv/dt, da/dt) — velocità nel piano valence-arousal
    acceleration: Tuple[float, float]  # (d²v/dt², d²a/dt²) — accelerazione
```

Il momentum informa:
- **Predizione**: dove sarà l'affetto tra t+1
- **Retrieval**: memorie in traiettoria simile sono più congruenti
- **Riconsolidamento**: un cambio brusco di momentum (sorpresa emotiva) triggera l'apertura della finestra

---

### Layer 3 — MoodField
**Ispirazione**: Heidegger (*Essere e Tempo*, §29-30) — Stimmung come modo di essere nel mondo; Bower (1981) mood-congruent memory
**Struttura**: Vettore PAD a variazione lenta — il "tono di fondo" del sistema

La Stimmung heideggeriana non è un'emozione ma un **modo di essere nel mondo** — non cambia per singoli eventi ma per accumulo. È il campo gravitazionale che orienta tutte le operazioni cognitive. Nella libreria questo concetto è incarnato da `MoodField`.

```python
MoodField:
    valence: float         # [-1.0, +1.0] media mobile pesata della storia affettiva
    arousal: float         # [0.0, 1.0]
    dominance: float       # [0.0, 1.0] — estensione PAD
    inertia: float         # [0.0, 1.0] resistenza al cambiamento (0=volatile, 1=stabile)
    timestamp: datetime    # quando è stato aggiornato l'ultima volta
```

**Aggiornamento**: Media esponenzialmente pesata del Core Affect nel tempo:
```
Mood(t) = α * CoreAffect(t) + (1-α) * Mood(t-1)
```
dove `α` è piccolo (0.05-0.15) — il Mood cambia lentamente.

**Effetto sul retrieval**: Il Mood pesa tutti i segnali di retrieval — in uno stato di Mood negativo, il bias verso memorie negative aumenta (mood-congruent memory).

---

### Layer 4 — Appraisal Vector
**Ispirazione**: Scherer CPM (2009), Lazarus (1991), Stoici (valutazione come generatrice di emozione)
**Struttura**: Vettore di valutazione multi-componente di un evento

L'emozione non è data — è **generata** dalla valutazione dell'evento rispetto ai goal e alle norme del sistema. Il Layer 4 è l'engine che produce il Core Affect (Layer 1) da input semantici.

```python
AppraisalVector:
    novelty: float              # [-1.0, +1.0] da completamente previsto a totalmente nuovo
    goal_relevance: float       # [-1.0, +1.0] da ostacola i goal a li favorisce
    coping_potential: float     # [0.0, 1.0] da nessuna capacità di coping a pieno controllo
    norm_congruence: float      # [-1.0, +1.0] da viola norme a conforme alle norme
    self_relevance: float       # [0.0, 1.0] quanto riguarda il "sé" del sistema
```

**Mapping appraisal → Core Affect**:
```python
# coping_potential [0,1] viene convertito a signed [-1,+1]:
# coping=1 → +1 (fiducia), coping=0 → -1 (helplessness)
coping_signed = 2.0 * coping_potential - 1.0

valence = 0.4*goal_relevance + 0.3*norm_congruence + 0.2*coping_signed + 0.1*novelty
arousal = 0.5*abs(novelty) + 0.3*(1-coping_potential) + 0.2*self_relevance
```

Il mapping è configurabile — questa è la componente più empiricamente aperta del sistema.

---

### Layer 5 — Associative Resonance
**Ispirazione**: Aristotele (associazionismo per somiglianza/contiguità/contrasto), Hume (simpathy), Bower (rete associativa affettiva)
**Struttura**: Grafo di connessioni tra memorie basate su risonanza emotiva

Quando una memoria viene encodata o recuperata, **risuona** con altre per:
1. **Similarità semantica**: contenuto simile
2. **Congruenza emotiva**: Core Affect simile al momento dell'encoding
3. **Contiguità temporale**: evento vicino nel tempo
4. **Causalità percepita**: evento causalmente connesso
5. **Contrasto emotivo**: opposizione affettiva (ansia ↔ sollievo)

```python
ResonanceLink:
    source_memory_id: str
    target_memory_id: str
    strength: float             # [0.0, 1.0]
    link_type: Literal['semantic', 'emotional', 'temporal', 'causal', 'contrastive']
```

I link di risonanza formano cluster emotivi che si rinforzano reciprocamente durante il retrieval (spreading activation).

---

## Il EmotionalTag: Struttura Completa

Ogni memoria porta un `EmotionalTag` che cattura tutti e 5 i layer al momento dell'encoding:

```python
@dataclass
class EmotionalTag:
    # Layer 1: Core Affect al momento dell'encoding
    core_affect: Tuple[float, float]          # (valence, arousal)

    # Layer 2: Momentum al momento dell'encoding
    momentum: Tuple[Tuple[float, float], Tuple[float, float]]  # (velocity, acceleration)

    # Layer 3: Mood snapshot al momento dell'encoding
    mood_snapshot: Tuple[float, float, float]  # (valence, arousal, dominance)

    # Layer 4: Appraisal che ha generato questa emozione
    appraisal: AppraisalVector | None         # None se emozione esogena

    # Layer 5: Link di risonanza con altre memorie
    resonance_links: list[ResonanceLink]

    # Metadata
    timestamp: datetime
    consolidation_strength: float             # [0.0, 1.0] — modulata da arousal
    reconsolidation_window_open: bool         # True per T_recon secondi dopo retrieval
    last_retrieved: datetime | None
    retrieval_count: int
```

---

## Principi di Encoding

### Principio 1 — Emotional Tagging è Pervasivo
*Fonte: Colombetti (2014), Richter-Levin (2004)*

Ogni memoria — senza eccezione — porta un `EmotionalTag`. Non esiste memoria "emotivamente neutra". Se non c'è appraisal esplicito, il tag usa il Core Affect e il Mood correnti come baseline.

### Principio 2 — Consolidation Strength è Funzione dell'Arousal
*Fonte: McGaugh (2004), Brown & Kulik (1977), Yerkes-Dodson*

```python
def consolidation_strength(arousal: float, mood_arousal: float) -> float:
    # Relazione a U invertita — non lineare
    # Peak attorno ad arousal=0.7, degradazione per valori estremi
    effective_arousal = 0.7 * arousal + 0.3 * mood_arousal
    return -4 * (effective_arousal - 0.7)**2 + 1.0
    # oppure: curva di Yerkes-Dodson parametrica
```

Memorie ad alta arousal (ma non massima) hanno la massima consolidation_strength → decay più lento → priorità di retrieval maggiore.

### Principio 3 — Dual-Speed Encoding
*Fonte: LeDoux (1996), dual pathway theory*

- **Fast path**: Encoding immediato con tag affettivo grezzo (valence + arousal dal Core Affect corrente, nessun appraisal). Avviene entro il prossimo ciclo.
- **Slow path**: Aggiornamento del tag con appraisal completo (Layer 4) dopo elaborazione contestuale. Apre la finestra di riconsolidamento.

---

## Principi di Retrieval

### Principio 4 — Retrieval Multi-Segnale
*Fonte: sintesi originale, Emotional RAG (2024), Bower (1981)*

```python
def retrieval_score(
    query_embedding: np.ndarray,
    query_affect: Tuple[float, float],        # Core Affect corrente
    current_mood: MoodField,
    current_momentum: AffectiveMomentum,
    memory: Memory,
    activation_map: dict[str, float]          # pre-calcolato da spreading_activation()
) -> float:

    w = adaptive_weights(current_mood)        # pesi modulati dal Mood

    s1 = cosine_similarity(query_embedding, memory.embedding)
    s2 = emotional_congruence(current_mood, memory.tag.mood_snapshot)
    s3 = core_affect_proximity(query_affect, memory.tag.core_affect)
    s4 = momentum_alignment(current_momentum, memory.tag.momentum)
    s5 = recency_decay(memory.tag.timestamp, memory.tag.consolidation_strength)
    s6 = activation_map.get(memory.id, 0.0)   # spreading activation boost

    return w[0]*s1 + w[1]*s2 + w[2]*s3 + w[3]*s4 + w[4]*s5 + w[5]*s6
```

### Principio 5 — Pesi Adattivi Modulati dal Mood
*Fonte: Heidegger (disclosure attraverso il mood), Bower (1981)*

I pesi `w` non sono fissi — dipendono dal Mood corrente:

```python
def adaptive_weights(mood: MoodField) -> np.ndarray:
    # Base weights
    w = np.array([0.35, 0.25, 0.15, 0.10, 0.10, 0.05])

    # Mood negativo intenso → più peso all'emotional congruence
    if mood.valence < -0.5:
        w[1] += 0.1; w[2] += 0.05
        w[0] -= 0.15  # meno peso semantico

    # Alta arousal → più sensibilità al momentum
    if mood.arousal > 0.7:
        w[3] += 0.1
        w[0] -= 0.1

    # Mood neutro/calmo → retrieval più razionale/semantico
    if abs(mood.valence) < 0.2 and mood.arousal < 0.3:
        w[0] += 0.15
        w[1] -= 0.1; w[2] -= 0.05

    return w / w.sum()  # normalizza
```

---

## Principi di Decay

### Principio 6 — Decay Non-Lineare Modulato da Arousal
*Fonte: ACT-R (Anderson, 1983), McGaugh (2004), Merleau-Ponty (abitudini)*

```python
def effective_decay_rate(
    base_decay: float,
    arousal_at_encoding: float,
    retrieval_count: int
) -> float:
    # Power law base (ACT-R style)
    # Modulazione per arousal: alta arousal = decay più lento
    arousal_factor = 1.0 - 0.5 * arousal_at_encoding

    # Il retrieval rinforza la memoria (spacing effect)
    retrieval_factor = 1.0 / (1.0 + 0.1 * retrieval_count)

    return base_decay * arousal_factor * retrieval_factor


def consolidation_at_time_t(
    initial_strength: float,
    decay_rate: float,
    time_delta_seconds: float,
    floor: float = 0.0
) -> float:
    # Power law decay con floor minimo per memorie importanti
    decayed = initial_strength * (time_delta_seconds ** (-decay_rate))
    return max(decayed, floor)
```

**Floor minimo**: Memorie con arousal > threshold mantengono una consolidation_strength minima non nulla — non vengono mai completamente dimenticate.

---

## Principio 7 — Riconsolidamento al Retrieval
*Fonte: Nader, Schiller (2000, 2010), prediction error theory*

Ogni retrieval apre una finestra di riconsolidamento:

```python
RECONSOLIDATION_WINDOW_SECONDS = 300  # 5 minuti (configurabile)

def on_retrieval(memory: Memory, current_state: AffectiveState) -> Memory:
    # Apre la finestra di riconsolidamento
    memory.tag.reconsolidation_window_open = True
    memory.tag.last_retrieved = datetime.now()
    memory.tag.retrieval_count += 1

    # Calcola l'affective prediction error
    ape = affective_prediction_error(
        expected=memory.tag.core_affect,
        observed=current_state.core_affect
    )

    # Se APE è significativo, aggiorna il tag emotivo
    if abs(ape) > APE_THRESHOLD:
        memory.tag = update_emotional_tag(memory.tag, current_state, ape)

    return memory


def update_emotional_tag(
    old_tag: EmotionalTag,
    new_state: AffectiveState,
    ape: float
) -> EmotionalTag:
    # Interpolazione pesata: il tag si aggiorna verso lo stato corrente
    # proporzionalmente all'entità dell'APE
    alpha = min(abs(ape) * LEARNING_RATE, 0.5)  # max 50% update
    new_core_affect = (
        (1-alpha) * old_tag.core_affect[0] + alpha * new_state.core_affect[0],
        (1-alpha) * old_tag.core_affect[1] + alpha * new_state.core_affect[1]
    )
    return replace(old_tag, core_affect=new_core_affect)
```

---

## Principio 8 — L'Appraisal come Generatore di Emozione
*Fonte: Lazarus (1991), Scherer (2009), Stoici*

L'emozione non è un input — è un **output** dell'appraisal dell'evento rispetto ai goal del sistema. Il sistema deve poter definire un **goal set** (conatus) che orienta l'appraisal.

```python
class AppraisalEngine:
    def __init__(self, goal_set: list[Goal], norm_set: list[Norm]):
        self.goals = goal_set
        self.norms = norm_set

    def appraise(self, event: Event, context: Context) -> AppraisalVector:
        novelty = self._compute_novelty(event, context)
        goal_relevance = self._compute_goal_relevance(event)
        coping = self._compute_coping_potential(event, context)
        norm_congruence = self._compute_norm_congruence(event)
        self_relevance = self._compute_self_relevance(event)
        return AppraisalVector(novelty, goal_relevance, coping, norm_congruence, self_relevance)

    def appraise_to_affect(self, event: Event, context: Context) -> Tuple[CoreAffect, AppraisalVector]:
        vec = self.appraise(event, context)
        affect = self._map_appraisal_to_affect(vec)
        return affect, vec
```

---

## Principio 9 — Risonanza Associativa
*Fonte: Aristotele (De Memoria), Hume (associazione), Bower (spreading activation)*

Al momento dell'encoding, costruire automaticamente link di risonanza:

```python
def build_resonance_links(
    new_memory: Memory,
    existing_memories: list[Memory],
    top_k: int = 5
) -> list[ResonanceLink]:
    links = []

    for mem in existing_memories:
        # Similarità semantica
        sem_sim = cosine_similarity(new_memory.embedding, mem.embedding)

        # Congruenza emotiva
        emo_sim = emotional_congruence(
            new_memory.tag.core_affect,
            mem.tag.core_affect
        )

        # Contiguità temporale
        temporal_prox = temporal_proximity(new_memory.tag.timestamp, mem.tag.timestamp)

        # Score composito
        resonance = 0.5*sem_sim + 0.3*emo_sim + 0.2*temporal_prox

        if resonance > RESONANCE_THRESHOLD:
            link_type = classify_link_type(sem_sim, emo_sim, temporal_prox)
            links.append(ResonanceLink(
                source_memory_id=new_memory.id,
                target_memory_id=mem.id,
                strength=resonance,
                link_type=link_type
            ))

    return sorted(links, key=lambda l: l.strength, reverse=True)[:top_k]
```

### Principio 9b — Link Bidirezionali e Spreading Activation
*Fonte: Collins & Loftus (1975), "A spreading-activation theory of semantic processing"*

I link di risonanza sono **bidirezionali**: quando la memoria A si connette alla memoria B, viene
creato anche il link inverso su B. Questo garantisce che l'attivazione si propaghi in entrambe le
direzioni nella rete associativa.

Durante il retrieval, **spreading activation** BFS a multi-hop propaga l'attivazione dalle memorie
seed verso le memorie adiacenti: ogni hop moltiplica l'attivazione per la forza del link
attraversato. Se due percorsi convergono sulla stessa memoria, si usa il massimo (non la somma) per
evitare inflazione artificiale dovuta al numero di percorsi convergenti.

```python
def spreading_activation(
    seeds: dict[str, float],          # memory_id -> activation iniziale
    store: MemoryStore,
    max_hops: int = 2,
) -> dict[str, float]:                # memory_id -> max activation raggiunta
    ...
```

### Principio 9c — Hebbian Co-Retrieval Strengthening
*Fonte: Hebb (1949), "The Organization of Behavior"*

> "Neurons that fire together, wire together."

Ogni volta che un gruppo di memorie viene recuperato insieme nella stessa query, i link esistenti
tra di esse vengono rinforzati di un incremento fisso (`hebbian_increment`, default 0.05). Questo
modella il fatto che l'associazione si consolida con l'uso: memorie co-attivate frequentemente
diventano progressivamente più connesse, emergendo spontaneamente come cluster tematici o
emotivi senza supervisione esplicita.

---

## Architettura di Sistema

### Ciclo di Vita di una Memoria

```
INPUT: evento/conversazione
         |
         v
    [1. Appraisal Engine]
    Calcola AppraisalVector
         |
         v
    [2. Core Affect Update]
    Aggiorna stato affettivo del sistema
         |
         v
    [3. Momentum Update]
    Calcola derivate temporali
         |
         v
    [4. Mood Update]
    Media mobile pesata (α piccolo)
         |
         v
    [5. Encoding]
    Crea EmotionalTag con snapshot dei 5 layer
    Calcola consolidation_strength
         |
         v
    [6. Storage]
    Salva in vector store con embedding + tag
         |
         v
    [7. Resonance Building]
    Costruisce link con memorie esistenti
         |
         v
    OUTPUT: memoria con full EmotionalTag
```

### Ciclo di Retrieval

```
QUERY: testo + stato affettivo corrente
         |
         v
    [1. Query Processing]
    Embedding + Core Affect + Mood correnti
         |
         v
    [2. Adaptive Weights]
    Calcola pesi in funzione del Mood
         |
         v
    [3. Multi-Signal Scoring]
    Calcola score per ogni candidato
         |
         v
    [4. Spreading Activation]
    BFS multi-hop dalla seed set (Collins & Loftus 1975)
    Costruisce activation_map per il Pass 2
         |
         v
    [5. Pass 2 Scoring + Top-K Selection]
    Re-score con resonance boost da activation_map
    Seleziona le memorie più rilevanti
         |
         v
    [6. Reconsolidation Check]
    Per ogni memoria recuperata: apre finestra,
    calcola APE, aggiorna tag se necessario
         |
         v
    [7. Hebbian Strengthening]
    Rinforza i link tra memorie co-recuperate (Hebb 1949)
         |
         v
    OUTPUT: memorie recuperate + tag aggiornati
```

---

## Confronto con i Principi di Neuroscienze e Filosofia

| Principio | Fonte teorica | Implementazione |
|-----------|---------------|-----------------|
| Affetto come transizione | Spinoza, *Ethica* III | Layer 2: Affective Momentum |
| Stimmung come campo globale | Heidegger, BT §29 | Layer 3: MoodField |
| Emozione come appraisal | Stoici, Scherer, Lazarus | Layer 4: Appraisal Vector |
| Risonanza associativa | Aristotele, Hume, Bower | Layer 5: Resonance Links |
| Consolidation ∝ arousal | McGaugh, LeDoux | consolidation_strength formula |
| Riconsolidamento | Nader, Schiller | on_retrieval() con APE |
| Mood-congruent retrieval | Bower (1981) | adaptive_weights() |
| Decay non-lineare | ACT-R, Yerkes-Dodson | effective_decay_rate() |
| Dual-speed encoding | LeDoux dual pathway | fast path + slow path |
| Tagging pervasivo | Colombetti, Memory Bear AI | EmotionalTag su ogni memoria |
