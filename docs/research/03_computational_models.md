# Modelli Computazionali dell'Emozione

> Panoramica dei principali modelli computazionali dell'emozione, con analisi critica dei loro punti di forza, limiti, e rilevanza per il nostro approccio "Affective Field Theory".

---

## 1. Modelli Dimensionali

### Russell — Circumplex Model of Affect (1980)

James Russell ha proposto che tutti gli stati affettivi possano essere mappati su uno spazio circolare bidimensionale:
- **Asse X**: Valenza (piacere — dispiacere)
- **Asse Y**: Arousal (attivazione — deattivazione)

Le etichette di emozioni discrete (felice, triste, arrabbiato, calmo, ecc.) sono distribuite intorno a questo cerchio come **regioni** nello spazio continuo, non come punti discreti.

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

**Punti di forza**:
- Continuo: nessuna discretizzazione artificiale
- Ben validato empiricamente attraverso culture
- Computazionalmente efficiente (2 coordinate)
- Cattura la struttura fondamentale dell'affetto

**Limiti**:
- Manca la dimensione Dominance (anger e fear hanno valenza/arousal simili ma dominance opposta)
- Non cattura la dinamica (è un modello di stato, non di processo)
- Alcune emozioni complesse difficili da posizionare

**Rilevanza per la libreria**: Il circumplex fornisce il substrato del **Layer 1 — Core Affect**. Ogni stato affettivo del sistema ha coordinate (valence, arousal) che lo posizionano nel circumplex.

---

### Mehrabian & Russell — PAD Model (1974)

Estende il circumplex a tre dimensioni:
- **Pleasure** (piacevole — spiacevole): ≈ valenza
- **Arousal** (eccitato — calmo): ≈ arousal
- **Dominance** (dominante/in-controllo — sottomesso/controllato)

La terza dimensione **Dominance** distingue emozioni che nel circumplex 2D sono confuse:
- **Anger**: alta arousal, bassa valenza, **alta dominance**
- **Fear**: alta arousal, bassa valenza, **bassa dominance**

Il PAD model afferma che queste tre dimensioni sono necessarie e sufficienti per caratterizzare qualsiasi stato emotivo.

**Validazione empirica**: Mappatura di centinaia di termini emotivi in 50+ lingue su questi tre assi mostra consistenza cross-culturale significativa.

**Rilevanza**: Il PAD è il candidato naturale per la rappresentazione del **Core Affect** nella nostra architettura — tre assi continui che coprono lo spazio affettivo con sufficiente granularità senza over-engineering.

---

## 2. Modelli Discreti Evolutivi

### Plutchik — Wheel of Emotions (1980)

Robert Plutchik ha proposto un modello evolutivo-dimensionale ibrido.

**Struttura**:
- **8 emozioni primarie** in coppie opposte: Joy/Sadness, Trust/Disgust, Fear/Anger, Surprise/Anticipation
- Disposte in un **cono 3D** con l'**intensità** sull'asse verticale (es. annoyance → anger → rage)
- **Dyadi** (blending): le emozioni adiacenti si combinano (es. joy + trust = love; fear + surprise = awe)

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

**Funzione evolutiva di ciascuna emozione**:
- Fear → Protezione
- Anger → Distruzione di barriere
- Joy → Riproduzione
- Sadness → Reintegrazione
- Trust → Affiliazione
- Disgust → Rifiuto
- Surprise → Orientamento
- Anticipation → Esplorazione

**Punti di forza**:
- Supporta il blending emotivo
- Dimensione di intensità
- Fondamento evolutivo
- Intuitivo e visivamente comunicabile

**Limiti**:
- La scelta degli 8 primitivi è arbitraria
- Il blending è additivo, non cattura la complessità dell'emergenza emotiva

**Rilevanza**: Il Wheel offre una tassonomia naturale per le **categorie emotive discrete** che emergono dallo spazio continuo PAD. Le regioni del cono di Plutchik possono essere mappate sullo spazio PAD, fornendo etichette human-readable per coordinate numeriche.

---

### Panksepp — 7 Sistemi Affettivi Primari (1998)

I sette sistemi di comando emotivo identificati da Panksepp attraverso la neurobiologia sperimentale (vedi doc 02_neuroscience.md per dettagli):

| Sistema | PAD appross. | Comportamento |
|---------|-------------|--------------|
| SEEKING | P+, A+, D+ | Esplorazione, curiosità |
| RAGE | P-, A+, D+ | Aggressione, frustrazione |
| FEAR | P-, A+, D- | Evitamento, freeze/fuga |
| LUST | P+, A+, D+ | Comportamento sessuale |
| CARE | P+, A-, D+ | Nutrimento, attaccamento |
| PANIC/GRIEF | P-, A-, D- | Isolamento, pianto |
| PLAY | P+, A+, D- | Legame sociale, gioco |

**Rilevanza**: Questi 7 sistemi possono funzionare come **cluster primari** nello spazio PAD — regioni attorno alle quali si organizzano le risposte emotive di base. Più robusti biologicamente rispetto al Wheel di Plutchik, ma meno granulari per usi linguistici.

---

## 3. Modelli Cognitivi e di Appraisal

### OCC Model — Ortony, Clore & Collins (1988)

Il modello computazionale dell'emozione più influente nell'AI simbolica. Genera 22 tipi di emozione da valutazioni (appraisal) di tre categorie:

- **Eventi** → pleased/displeased (es. joy, distress, hope, fear, satisfaction, disappointment)
- **Agenti** (azioni) → approving/disapproving (es. pride, shame, admiration, reproach)
- **Oggetti** → liking/disliking (es. love, hate)

Le emozioni composte emergono dalle combinazioni (es. gratification = joy + pride; remorse = distress + shame).

```
                          EMOTIONS
                         /    |    \
                      Events Agents Objects
                      /  \    / \    / \
                pleased disp. app. repr. like dislike
                  |       |
           hope distress joy fear ...
```

**Punti di forza**:
- Rule-based → interpretabile e trasparente
- Struttura compositiva
- Standard de facto per agenti virtuali (NPC nei videogiochi, chatbot sociali)

**Limiti**:
- Discreto → perde la continuità dell'esperienza emotiva
- Rule-based → rigido, non apprende
- Non modella la dinamica temporale
- Difficile scalare a scenari aperti

**Rilevanza**: L'OCC fornisce un **vocabolario di etichette emotive** strutturato che può essere usato come layer semantico sopra le coordinate PAD. Un evento con (valence=+0.7, arousal=0.4) in risposta a un goal-furthering event → etichetta OCC "joy" o "satisfaction".

---

### Scherer — Component Process Model (CPM)

(Dettagliato in doc 02_neuroscience.md — sezione 5)

Il CPM è il più ricco tra i modelli di appraisal. La sequenza dei 5 SECs (Stimulus Evaluation Checks) genera stati emotivi in modo processuale.

**Formato computazionale** (CPM originale di Scherer):
```
appraisal_result = {
    novelty_check: float,          # 0=completamente previsto, 1=totalmente nuovo
    intrinsic_pleasantness: float, # -1=spiacevole, +1=piacevole
    goal_significance: float,      # -1=ostacola, +1=favorisce
    coping_potential: float,       # 0=nessuna coping, 1=pieno controllo
    norm_compatibility: float      # -1=viola norme, +1=conforme
}
```

Questi 5 valori determinano lo stato emotivo risultante attraverso regole compositive.

**Rilevanza**: Il CPM è il modello che più direttamente informa il **Layer 4 — Appraisal Vector** della nostra architettura.

**Nota di implementazione**: nella libreria `intrinsic_pleasantness` è sostituita da `self_relevance` (quanto l'evento riguarda il "sé" del sistema, `[0.0, 1.0]`). Questa deviazione rispetto al CPM originale è intenzionale: `self_relevance` è più operazionalizzabile in un agente LLM e contribuisce all'arousal (non alla valence), preservando i 5 SEC ma ridistribuendo i ruoli.

---

## 4. Modelli Cognitivi Architetturali

### ACT-R — Adaptive Control of Thought—Rational (Anderson, 1983+)

ACT-R è un'architettura cognitiva unificata. Il retrieval è governato da livelli di attivazione:

```
Ai = Bi + Σ(Wj * Sji) + εi
```

Dove:
- `Bi` = attivazione di base (recency e frequenza con decadimento a legge di potenza)
- `Σ(Wj * Sji)` = attivazione per spreading context
- `εi` = rumore gaussiano

**Decadimento base-level**:
```
Bi = ln(Σ(t_j^(-d)))
```
Dove `t_j` sono i tempi trascorsi da ogni accesso e `d` è il parametro di decadimento.

**Estensioni emotive di ACT-R**: Ricercatori hanno proposto estensioni dove:
- L'emozione modifica il parametro `d` (decadimento) — memorie emotivamente intense decadono più lentamente
- L'emozione modifica `Wj` — il contesto emotivo corrente pesa maggiormente i ricordi emotivamente congruenti

**Rilevanza**: Il meccanismo di base-level activation di ACT-R è il fondamento del nostro sistema di decay differenziato. Adottiamo la legge di potenza ma aggiungiamo la modulazione per arousal.

---

### SOAR con Emozione (Marinier, Laird & Lewis, 2009)

"A Computational Unification of Cognitive Behavior and Emotion" integra la teoria dell'appraisal in SOAR. Le valutazioni emotive servono come **ricompense intrinseche per il reinforcement learning**.

Fasi:
- Appraisal di rilevanza durante Perceive/Encode
- Appraisal di valutazione durante Comprehend
- Probabilità di esito durante Intend

**Rilevanza**: Il principio di **emotion-as-intrinsic-reward** è applicabile alla nostra architettura: le emozioni non solo taggano le memorie ma modulano il reinforcement del comportamento del sistema.

---

### CLARION — Motivational Subsystem (Sun, 2016)

CLARION include un sottosistema motivazionale esplicito con:
- **Drive di basso livello** (sopravvivenza, curiosità)
- **Drive di alto livello** (scopo, focalizzazione, adattamento)

L'emozione è generata dalle attivazioni dei drive e dai potenziali d'azione, con appraisal e metacognizione come fattori secondari.

**Rilevanza**: Il **conatus spinoziano** nella nostra architettura corrisponde ai drive di CLARION — i goal del sistema che generano il piano valutativo dell'appraisal.

---

## 5. Modelli Neural-Simbolici Recenti (2023-2026)

### Chain-of-Emotion Architecture (2024)

Architettura per agenti LLM affettivi nei videogiochi. Usa una chiamata LLM separata per l'appraisal emotivo prima di ogni risposta:
```
prompt = system_instruction + message_history + emotion_history + user_input
```
Valutato come significativamente più naturale ed emotivamente sensibile dei controlli (PLOS One, 2024).

**Rilevanza**: La struttura a **two-pass** (prima appraisal, poi risposta) è il pattern che adottiamo nel nostro appraisal engine.

### Emotional RAG (Huang et al., 2024)

Due strategie di retrieval mood-dependent:
1. **Combination**: peso congiunto semantica + stato emotivo
2. **Sequential**: filtraggio per emozione poi ranking semantico (o viceversa)

Supera RAG standard su dataset di role-playing.

**Rilevanza**: Conferma l'utilità del segnale `emotional_congruence` nel retrieval scoring. Il nostro approccio estende questo con il momentum, il Mood field e la spreading activation.

### Dynamic Affective Memory Management (2025)

Aggiornamento Bayesiano della memoria con minimizzazione dell'entropia. Usa la **memoria entropy** come metrica di qualità — il sistema minimizza l'entropia globale attraverso aggiornamenti Bayesiani al vettore di memoria affettiva.

**Rilevanza**: Il principio di **entropia come qualità della memoria** è promettente per il nostro sistema di consolidamento. Memorie più precise (bassa entropia) dovrebbero avere consolidation_strength maggiore.

---

## 6. Analisi Comparativa: Perché Nessun Modello Esistente è Sufficiente

| Modello | Statico/Dinamico | Continuo/Discreto | Appraisal | Temporale | Globale |
|---------|-----------------|-------------------|-----------|-----------|---------|
| Russell Circumplex | Statico | Continuo | No | No | No |
| PAD | Statico | Continuo | No | No | No |
| OCC | Process | Discreto | Si' | No | No |
| Plutchik Wheel | Statico | Ibrido | No | Si' (intensita') | No |
| Panksepp 7 | Statico | Discreto | No | No | No |
| Barrett Core Affect | Statico | Continuo | Parziale | No | No |
| CPM (Scherer) | Process | Continuo | Si' | Parziale | No |
| ACT-R ext. | Dinamico | Continuo | No | Si' | No |
| Emotional RAG | Process | Ibrido | No | No | No |

**Gap sistematici identificati**:
1. **Nessun modello** cattura la **dinamica temporale** dell'affetto (dove stai andando, non solo dove sei)
2. **Nessun modello** ha un **campo globale** (mood field) che colori tutte le operazioni
3. **Nessun modello** integra **appraisal + memoria + retrieval** in un framework unificato
4. **Nessun modello** implementa il **riconsolidamento** come operazione nativa

Questi quattro gap motivano la nostra **Affective Field Theory**.

---

## Note Bibliografiche

- Russell, J.A. (1980). "A circumplex model of affect." *Journal of Personality and Social Psychology*, 39, 1161-1178.
- Mehrabian, A. & Russell, J.A. (1974). *An Approach to Environmental Psychology*. MIT Press.
- Plutchik, R. (1980). *Emotion: A Psychoevolutionary Synthesis*. Harper & Row.
- Panksepp, J. (1998). *Affective Neuroscience*. Oxford University Press.
- Ortony, A., Clore, G.L. & Collins, A. (1988). *The Cognitive Structure of Emotions*. Cambridge University Press.
- Anderson, J.R. et al. (2004). "An integrated theory of the mind." *Psychological Review*, 111(4), 1036-1060.
- Marinier, R.P., Laird, J.E. & Lewis, R.L. (2009). "A computational unification of cognitive behavior and emotion." *Cognitive Systems Research*.
- Sun, R. (2016). *Anatomy of the Mind*. Oxford University Press.
- Huang, et al. (2024). "Emotional RAG: Enhancing Role-Playing Agents through Emotional Retrieval." arXiv:2410.23041.
- Zhang & Zhong (2025). "Decoding Emotion in the Deep." arXiv:2510.04064.
- Dynamic Affective Memory Management (2025). arXiv:2510.27418.
