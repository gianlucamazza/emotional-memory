# Stato dell'Arte: Sistemi di Memoria per LLM e Affective Computing

> Analisi dei sistemi esistenti, con focus sui gap nell'implementazione della memoria emotiva. Questo documento motiva il posizionamento della nostra libreria nel panorama attuale.

---

## 1. Sistemi di Memoria per LLM

### 1.1 MemGPT / Letta (Packer et al., 2023)

**Paper**: "MemGPT: Towards LLMs as Operating Systems" (arXiv:2310.08560)
**GitHub**: https://github.com/cpacker/MemGPT → diventato https://research.memgpt.ai/

**Architettura — Tre tier ispirati ai SO**:
- **Core Memory** (sempre in-context, come RAM): persona del sistema, informazioni utente chiave
- **Recall Memory** (storia conversazionale ricercabile): database vettoriale delle conversazioni passate
- **Archival Memory** (storage a lungo termine, come disco): memoria espandibile illimitatamente

**Innovazione chiave**: L'LLM stesso decide cosa memorizzare, riassumere e dimenticare via tool calls — auto-gestione della memoria.

**Gap emotivo**: Nessuna rappresentazione emotiva. L'importanza di un ricordo è determinata dall'LLM in modo opaco, senza una dimensione affettiva strutturata. Non esiste decay differenziato basato su caratteristiche emotive.

---

### 1.2 Mem0 (2024)

**Paper**: arXiv:2504.19413
**GitHub**: https://github.com/mem0ai/mem0 (~48K stars, $24M funding)

**Approccio**: Layer di memoria universale per agenti AI. Estrae dinamicamente informazioni salienti dalle conversazioni, le consolida, e le recupera.

**Metriche**: 26% maggiore accuracy, 91% minore latency, 90% risparmio di token rispetto a approcci full-context.

**Gap emotivo**: Estrae fatti e preferenze, non stati emotivi. Non distingue tra "l'utente preferisce il caffè" e "l'utente era angosciato quando ha parlato del suo lavoro."

---

### 1.3 Zep (2024)

**GitHub**: https://github.com/getzep/zep

**Approccio**: Grafo di conoscenza temporale che connette interazioni passate, dataset strutturati e cambiamenti di contesto. Il motore Graphiti combina memoria episodica (chat), semantica (entità) e subgraph a livello di gruppo.

**Gap emotivo**: Il grafo di conoscenza cattura relazioni semantiche ma non dimensioni affettive. Non esiste nodo o arco che rappresenti lo stato emotivo associato a un'interazione.

---

### 1.4 Generative Agents (Park et al., 2023)

**Paper**: "Generative Agents: Interactive Simulacra of Human Behavior" (arXiv:2304.03442)
**Citazioni**: ~3000+ — paper fondativo del campo

**Formula di retrieval**:
```
score = recency(t) + importance(m) + relevance(q, m)
```
- `recency`: decadimento esponenziale dal tempo dell'ultimo accesso
- `importance`: valutazione LLM su scala 1-10 ("quanto è importante questo ricordo per il personaggio?")
- `relevance`: coseno similarity tra embedding query e embedding memoria

**Gap emotivo**: `importance` è un proxy opaco che può catturare rilevanza emotiva, ma:
1. Non è strutturata — non distingue valenza, arousal, tipo di emozione
2. Non modula il decay — una memoria molto importante decade alla stessa velocità di una banale
3. Non ha mood-congruent retrieval — il mood del personaggio non pesa nel retrieval
4. Non esiste riconsolidamento

Questo paper ha definito lo standard de facto che la maggior parte dei sistemi successivi riproduce con variazioni minori.

---

### 1.5 OpenMemory / Cavira (2025)

**GitHub**: https://github.com/CaviraOSS/OpenMemory
**Docs**: https://openmemory.cavira.app/

**Architettura — 5 settori con decay differenziato**:
1. **Semantic** (fatti, concetti): decay standard
2. **Episodic** (eventi specifici): decay moderato
3. **Procedural** (procedure, skill): decay lento (abitudini persistono)
4. **Emotional** (stati emotivi, esperienze affettive): decay lento con floor minimo
5. **Reflective** (auto-valutazioni, insight): decay molto lento

**Innovazione chiave**: Il primo sistema che include un **settore emotivo esplicito** con decay differenziato. I "reinforcement pulses" alzano ricordi critici sopra la soglia di ritenzione.

**Gap**: Emozione come settore separato, non come dimensione pervasiva. La rappresentazione emotiva interna non è documentata in dettaglio. Non esiste appraisal engine.

---

### 1.6 Memory Bear AI (2025-2026)

**Paper**: arXiv:2512.20651 (Dec 2025) + Technical Report arXiv:2603.22306 (Mar 2026)

**Innovazione**: **Emotion Memory Units (EMUs)** — primo sistema che tratta l'emozione come **dimensione nativa** della memoria, non come settore separato o etichetta post-hoc.

Ogni EMU combina:
- Contenuto semantico
- Stato emotivo al momento dell'encoding
- Tracce di impatto emotivo successive

**Rilevanza**: L'approccio EMU è il più vicino alla nostra visione. Ma rimane limitato — non ha la struttura multi-layer (momentum, mood field, appraisal vettoriale) né il retrieval multi-segnale con pesi adattivi.

---

### 1.7 Emotional RAG (Huang et al., 2024)

**Paper**: "Emotional RAG: Enhancing Role-Playing Agents through Emotional Retrieval" (arXiv:2410.23041)

**Fondamento teorico**: Bower's Mood-Dependent Memory Theory (1981)

**Due strategie di retrieval**:
1. **Combination**: peso congiunto similarità semantica + stato emotivo
2. **Sequential**: filtraggio per emozione poi ranking semantico (o viceversa)

Supera RAG standard su InCharacter, CharacterEval, Character-LLM datasets.

**Gap**: Testato solo in scenari di role-playing. Non ha memory management a lungo termine. L'emozione è un singolo segnale, non un campo multi-layer.

---

### 1.8 A-Mem / Agentic Memory (2025)

**Paper**: "A-MEM: Agentic Memory for LLM Agents" (NeurIPS 2025, arXiv:2502.12110)
**GitHub**: https://github.com/WujiangXu/A-mem

**Approccio**: L'agente LLM gestisce autonomamente le proprie operazioni di memoria (cosa ricordare, come strutturare, quando dimenticare). Memoria come sistema adattivo auto-organizzante.

**Gap emotivo**: Focus su auto-gestione, non su dimensioni affettive.

---

### 1.9 MemOS — Memory Operating System (2025)

**GitHub**: https://github.com/MemTensor/MemOS

**Approccio**: Sistema operativo per la memoria degli LLM con knowledge base, memory feedback, memoria multi-modale, e memory tool per la pianificazione degli agenti.

**Gap emotivo**: Struttura sistemica senza componente affettiva.

---

## 2. MemEmo Benchmark: Il Gap Definitivo (Feb 2026)

**Paper**: "MemEmo: A Benchmark for Evaluating Emotional Memory in LLM Agents" (arXiv:2602.23944)

**Risultato chiave**: **Nessun sistema di memoria attuale gestisce correttamente l'emozione** attraverso tutte le fasi: estrazione, aggiornamento, e question-answering.

**Dataset**: HLME (Human-Like Memory with Emotion) — conversazioni con stati emotivi espliciti e impliciti, cambiamenti di stato, e query che richiedono reasoning emotivo sulla memoria.

**Task valutati**:
1. **Emotion Extraction**: Identificare lo stato emotivo in una conversazione
2. **Emotion Update**: Aggiornare la memoria quando l'emozione cambia
3. **Emotion QA**: Rispondere a domande che richiedono reasoning sulla memoria emotiva

**Sistemi testati**: Tutti i principali sistemi (Mem0, Zep, MemGPT/Letta, altri) falliscono sistematicamente, specialmente nell'aggiornamento e nel reasoning sulla memoria emotiva.

Questo paper fornisce una forte motivazione per una libreria che tratti
l'emozione come parte strutturale della memoria, ma non costituisce da solo una
validazione della nostra implementazione.

---

## 3. Affective Computing: Panorama Contemporaneo

### Fusione Multimodale

La tendenza dominante (2022-2026) combina:
- Espressione facciale
- Prosodia del parlato
- Testo
- Segnali fisiologici (EDA, heart rate, EEG)
- Eye tracking

Transformer cross-modali (es. Joint-Dimension-Aware Transformer) dominano. Accuracy > 90% in alcuni paradigmi.

**Rilevanza**: Il nostro Layer 4 (Appraisal) può potenzialmente accettare input multimodale oltre al testo.

### Emotional Support Conversation (ESC)

LLM-based systems per supporto emotivo: rilevano lo stato emotivo dell'utente e generano dialogo empatico con validazione, normalizzazione, reframing.

**Gap**: Questi sistemi rilevano e rispondono all'emozione nel momento, ma non la memorizzano strutturalmente per informare interazioni future.

### EmoLLMs (Liu et al., 2024)

**Paper**: arXiv:2401.08508
Prima serie di LLM open-source per analisi affettiva completa. Include AAID (234K samples) e benchmark AEB (14 task). Supera ChatGPT/GPT-4 sulla maggior parte dei task.

### EmotionPrompt & NegativePrompt

**EmotionPrompt** (Li et al., 2023, arXiv:2307.11760): Stimoli emotivi nei prompt producono +8% su Instruction Induction, +115% su BIG-Bench.

**NegativePrompt** (Wang et al., 2024, arXiv:2405.02814): Stimoli emotivi negativi: +12.89% e +46.25% rispettivamente.

**Implicazione**: Il contesto emotivo modifica significativamente il comportamento degli LLM. Una libreria che mantiene e inietta contesto emotivo nella memoria del sistema può migliorare sistematicamente la qualità delle risposte.

### Latent Emotional Representations in LLMs (Zhang & Zhong, 2025)

**Paper**: "Decoding Emotion in the Deep" (arXiv:2510.04064)

**Scoperte chiave**:
- Gli LLM sviluppano una **geometria interna ben definita** dell'emozione (rappresentazioni latenti strutturate)
- Il segnale emotivo **raggiunge il picco a metà rete**
- Il tono emotivo iniziale **persiste per centinaia di token**

**Implicazione rivoluzionaria**: Gli LLM già rappresentano internamente l'emozione in modo strutturato. Una libreria di memoria emotiva può agganciarsi a queste rappresentazioni latenti piuttosto che costruire tutto dall'esterno.

---

## 4. Gap Analysis e Posizionamento

### Gap identificati nel panorama attuale

| Gap | Impatto | Nostra risposta |
|-----|---------|-----------------|
| Emozione come settore separato, non dimensione pervasiva | Ogni sistema manca l'impatto affettivo su encoding/retrieval | Emotional tagging pervasivo su ogni memoria |
| Rappresentazione statica (punto nello spazio) | Non cattura la dinamica — dove stai andando | Layer 2: Affective Momentum (derivate) |
| Nessun mood field globale | Il mood corrente non pesa nel retrieval in modo strutturato | Layer 3: MoodField |
| Nessun appraisal integrato | L'emozione viene assunta come input, non generata | Layer 4: Appraisal Vector |
| Retrieval mono-segnale | Solo semantica, o semantica + un segnale emotivo statico | Retrieval multi-segnale con 6 componenti e pesi adattivi |
| Nessun riconsolidamento | Il retrieval non può aggiornare il tag emotivo | Finestra di riconsolidamento post-retrieval |
| Decay uniforme | Memorie emotivamente intense non sono privilegiate sistematicamente | Decay non-lineare modulato da arousal |

### Il nostro posizionamento

```
                    Ampiezza della copertura emotiva
                    (quanti aspetti dell'emozione coprono)
                              Low        High
                         ┌─────────────────────┐
                    High  │              ★ Nostra│
Profondità               │              libreria │
dell'integrazione        │  Memory Bear AI       │
(quanto è pervasiva      │  OpenMemory          │
l'emozione nel sistema)  │                      │
                    Low   │ MemGPT, Mem0, Zep   │
                         └─────────────────────┘
```

Nel panorama qui analizzato, AFT occupa un punto relativamente raro: **alta
profondita' di integrazione affettiva** (l'emozione pesa su encoding e retrieval)
e **copertura multi-layer** (core affect + momentum + mood field + appraisal +
resonance). Questa mappa e' interpretativa, non una prova di esclusivita': sistemi
come Memory Bear AI e OpenMemory coprono sottoinsiemi importanti dello stesso
spazio progettuale con astrazioni diverse.

---

## Note Bibliografiche

- Packer, C. et al. (2023). "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560.
- Mem0 paper. arXiv:2504.19413.
- Park, J.S. et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior." arXiv:2304.03442.
- OpenMemory / Cavira: https://github.com/CaviraOSS/OpenMemory
- Memory Bear AI: arXiv:2512.20651, arXiv:2603.22306.
- Huang et al. (2024). "Emotional RAG." arXiv:2410.23041.
- Xu, W. et al. (2025). "A-MEM: Agentic Memory for LLM Agents." NeurIPS 2025. arXiv:2502.12110.
- MemEmo (2026). arXiv:2602.23944.
- Liu, Z. et al. (2024). "EmoLLMs: A Series of Emotional Large Language Models." arXiv:2401.08508.
- Li, C. et al. (2023). "EmotionPrompt: Elevating LLM Performance Through Emotional Intelligence." arXiv:2307.11760.
- Wang et al. (2024). "NegativePrompt." arXiv:2405.02814.
- Zhang & Zhong (2025). "Decoding Emotion in the Deep." arXiv:2510.04064.
- Dynamic Affective Memory Management (2025). arXiv:2510.27418.
