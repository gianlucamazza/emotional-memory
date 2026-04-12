# Neuroscienze e Scienze Cognitive della Memoria Emotiva

> Questo documento raccoglie le basi neuroscientifiche e cognitive che informano l'architettura della libreria. Ogni sezione termina con le implicazioni operative per il sistema.

---

## 1. Il Circuito Neurale Fondamentale: Amigdala-Ippocampo

### La Doppia Via di LeDoux

Joseph LeDoux ha stabilito l'architettura neurale della memoria di paura attraverso il condizionamento Pavloviano.

**Due percorsi paralleli**:
- **Via rapida** (*low road*): Talamo sensoriale → Amigdala direttamente. ~150-200ms. Elaborazione grezza, sufficiente per risposte difensive immediate prima della coscienza.
- **Via lenta** (*high road*): Talamo sensoriale → Corteccia sensoriale → Amigdala. Più lenta ma consente valutazione dettagliata e nuancée. Riduce i falsi allarmi.

Il nucleo laterale dell'amigdala riceve input convergenti da entrambe le vie ed è critico per associare stimoli a minacce.

**Implicazione per l'encoding**: Il sistema deve supportare encoding a due velocità:
- **Fast path**: tag emotivo immediato (valence + arousal grezzo) basato su pattern superficiali
- **Slow path**: tag emotivo elaborato con appraisal completo post-elaborazione

---

### Modulazione dell'Ippocampo da parte dell'Amigdala (McGaugh)

James McGaugh ha dimostrato sperimentalmente che il nucleo basolaterale dell'amigdala (BLA) media la modulazione della consolidazione mnestica da parte degli ormoni dello stress.

**Meccanismo**: Dopo un apprendimento emotivamente rilevante, la norepinefrina extracellulare nell'amigdala sale a ~300% sopra la baseline e correla con le prestazioni di ritenzione. La BLA modula poi la consolidazione nelle regioni downstream (caudato, nucleus accumbens, corteccia).

**L'ipotesi dell'emotional tagging** (Richter-Levin & Akirav): L'amigdala, attivata dall'arousal emotivo, "taglia" una traccia mnestica ippocampale rinforzando la consolidazione sinaptica ai neuroni attivati.

**Arousale vs. valenza**: Le memorie emotivamente arousing dipendono dalla rete amigdala-ippocampo; le memorie emotivamente valenzate ma non arousing dipendono dalla rete corteccia prefrontale-ippocampo (Kensinger & Corkin, 2004).

**Implicazione**: La `consolidation_strength` di ogni memoria è funzione primaria dell'**arousal** al momento dell'encoding, non della valenza. Alta arousal → consolidamento preferenziale → decay più lento.

---

## 2. La Teoria dei Marcatori Somatici (Damasio)

Antonio Damasio ha proposto che le emozioni, esperite come stati corporei ("marcatori somatici"), svolgano un ruolo critico nel ragionamento decisionale in condizioni di complessità e incertezza.

**Core principles**:
- I marcatori somatici sono cambiamenti di stato corporeo (battito cardiaco accelerato, sensazione viscerale) associati agli esiti emotivi passati delle decisioni
- Sono rappresentati e regolati principalmente nella corteccia prefrontale ventromediale (vmPFC) e nell'amigdala
- Durante le decisioni, i marcatori somatici orientano le scelte riattivando il "tag" emotivo degli esiti precedenti, coscientemente o inconsciamente

**Iowa Gambling Task**: I pazienti con danno alla vmPFC non sviluppano marcatori somatici e fanno scelte sistematicamente svantaggiose nonostante capacità logiche intatte.

**Implicazione**: I **somatic markers** sono il precedente neuroscientifico dei nostri `EmotionalTag`. Un sistema di retrieval che usa i tag emotivi delle memorie passate come bias decisionale implementa computazionalmente l'ipotesi di Damasio.

---

## 3. I Sette Sistemi Affettivi di Panksepp

Jaak Panksepp ha mappato sette sistemi di comando emotivo primari attraverso stimolazione elettrica cerebrale, sfide farmacologiche e studi lesionali in cervelli mammiferi.

| Sistema | Valenza | Funzione |
|---------|---------|----------|
| **SEEKING** (Aspettativa) | Positiva | Esplorazione, curiosità, entusiasmo anticipatorio |
| **RAGE** (Rabbia) | Negativa | Frustrazione, aggressività quando i goal sono bloccati |
| **FEAR** (Paura) | Negativa | Rilevamento minacce, risposte freeze/fuga |
| **LUST** (Desiderio sessuale) | Positiva | Spinta riproduttiva |
| **CARE** (Cura) | Positiva | Attaccamento, comportamento materno |
| **PANIC/GRIEF** (Panico/Dolore) | Negativa | Distress da separazione, perdita del legame sociale |
| **PLAY** (Gioco) | Positiva | Legame sociale, gioco ruvido |

Ogni sistema è generato subcorticamente, conservato evolutivamente nelle specie, e valutativo (avvicinamento per positivo, evitamento per negativo).

**Implicazione**: Questi 7 sistemi offrono una tassonomia di **primitive emotive biologicamente fondate** che possono servire da base per la categorizzazione discreta, distinta dalle coordinate continue (valence, arousal). Il sistema SEEKING corrisponde funzionalmente al conatus spinoziano — la spinta esplorativa fondamentale.

---

## 4. Consolidamento e Modulazione Emotiva della Memoria

### Flashbulb Memory (Brown & Kulik, 1977)

Le *flashbulb memories* sono ricordi vividi, dettagliati, altamente confidenti delle circostanze in cui si è appresa una notizia significativa ed emotivamente intensa (es. l'11 settembre, un evento personale traumatico).

**Meccanismo**: L'attivazione dell'amigdala innesca il rilascio di ormoni dello stress che potenziano la consolidazione ippocampale — producendo tracce mnestiche inusualmente durature.

**Limitazione critica**: Nonostante la loro vivacità soggettiva, le flashbulb memories **non sono necessariamente più accurate** delle memorie ordinarie. Sono più consistenti e confidently held, ma soggette alla distorsione nel tempo (Talarico & Rubin, 2003).

**Implicazione per la libreria**: Alta arousal → alta `consolidation_strength` → decay lento → alta priorità di retrieval. Ma non necessariamente **accuracy** elevata. Il sistema non deve trattare le memorie ad alta arousal come ground truth.

---

### Ormoni dello Stress e Curva di Yerkes-Dodson

**Relazione a U invertita** tra livello di ormoni dello stress e prestazioni mnemoniche:
- Livelli moderati di cortisolo e norepinefrina **potenziano** encoding e consolidamento
- Stress estremo o cronico (cortisolo elevato) **compromette** la funzione ippocampale e il retrieval

**Il timing conta**:
- Ormoni rilasciati **durante o immediatamente dopo** l'encoding → potenziano la consolidazione
- Stress **prima del retrieval** → lo compromette

**Implicazione**: Il sistema deve modellare la relazione non-lineare tra arousal e consolidation_strength. Un valore di arousal molto elevato non produce necessariamente la massima consolidation_strength.

---

### Riconsolidamento (Nader, Schiller)

Karim Nader, Glenn Schaller e Daniela Schiller hanno dimostrato che le memorie riattivate diventano temporaneamente labili e devono essere "riconsolidate" per persistere.

**Scoperta fondamentale**: Quando una memoria consolidata viene riattivata (recuperata), entra in uno stato labile per una finestra temporale limitata. Durante questa finestra:
- La memoria può essere **indebolita** (agenti amnesici come propranolol o anisomicina)
- La memoria può essere **aggiornata** con nuove informazioni
- Il **tag emotivo** può essere modificato

**Prediction error come trigger**: L'errore di predizione (mismatch tra aspettativa ed esperienza) sembra necessario per innescare il riconsolidamento.

**Applicazione clinica**: I ricordi di paura possono essere modificati durante il riconsolidamento attraverso interventi comportamentali (estinzione durante la finestra di riconsolidamento) o farmacologici.

**Implicazione critica per la libreria**: Ogni operazione di **retrieval** deve aprire una finestra di riconsolidamento durante la quale il `EmotionalTag` della memoria può essere aggiornato. Questo significa che il retrieval non è un'operazione read-only — può modificare la memoria stessa.

---

## 5. Teorie dell'Appraisal Cognitivo

### Lazarus — Appraisal Primario e Secondario (1966, 1991)

Richard Lazarus ha proposto che le emozioni emergano non dagli eventi stessi ma dalla **valutazione cognitiva** di quegli eventi.

**Appraisal in due fasi**:
1. **Appraisal primario**: Questo evento è rilevante per me? È minaccioso, benefico, o irrilevante?
2. **Appraisal secondario**: Posso far fronte? Quali risorse ho?

La combinazione determina l'emozione specifica generata (es. minaccia + basso coping = paura/ansia; minaccia + alto coping = sfida).

---

### Scherer — Component Process Model (CPM)

Klaus Scherer ha espanso la teoria dell'appraisal in un modello di processo multi-componente sequenziale.

**Cinque Stimulus Evaluation Checks (SECs)**:
1. **Novità**: Sta succedendo qualcosa di nuovo o inaspettato?
2. **Piacevolezza intrinseca**: È intrinsecamente piacevole o spiacevole?
3. **Significanza per i goal**: Favorisce o ostacola i miei goal?
4. **Potenziale di coping**: Posso gestirlo?
5. **Significanza normativa**: È coerente con le mie norme e l'auto-concetto?

L'emozione è definita come la sincronizzazione di cinque componenti: appraisal cognitivo, arousal fisiologico, espressione motoria, tendenza all'azione, e sentimento soggettivo.

**Implicazione per il Layer 4**: Il nostro **Appraisal Vector** implementa una versione computazionale dei SECs di Scherer. Le cinque dimensioni dell'appraisal nel nostro modello (novelty, goal_relevance, coping_potential, norm_congruence, self_relevance) derivano direttamente da questo framework.

---

## 6. La Teoria dell'Emozione Costruita (Barrett)

Lisa Feldman Barrett sfida la visione classica delle emozioni di base con un account costruzionista.

**Core claims**:
- Le emozioni non sono categorie cablate, universali, innescate da circuiti cerebrali specifici. Invece, sono **costruite** nel momento dalla macchina predittiva del cervello.
- Il cervello genera continuamente predizioni sui segnali sensoriali in ingresso basandosi su **interocezione** (sensing corporeo interno), esperienza precedente, e concetti emotivi appresi culturalmente.
- Il **core affect** — uno stream continuo di valenza (piacevole/spiacevole) e arousal (attivato/deattivato) — è il mattone fondamentale. Le emozioni discrete (rabbia, paura, ecc.) emergono quando il cervello categorizza il core affect usando conoscenza concettuale.

**Implicazione**: Non esiste un'unica firma cerebrale per nessuna emozione; le emozioni emergono da reti cerebrali domain-general. Per la libreria: le etichette emotive discrete (joy, fear, anger) sono **pattern emergenti** nello spazio affettivo continuo, non primitive hardwired. Questo giustifica un approccio ibrido: coordinate continue + categorizzazione contestuale.

---

## 7. Memoria Congruente con il Mood e Apprendimento State-Dependent

### Mood-Congruent Memory (Bower, 1981)

Le persone codificano e recuperano preferenzialmente informazioni che corrispondono al loro stato d'umore corrente.

**Base teorica**: La **Teoria della Rete Associativa di Gordon Bower** — l'affetto esiste come nodo centrale in una rete di memoria associativa. Quando un umore è attivo, diffonde attivazione ai nodi mood-congruenti, facilitando encoding e retrieval del materiale corrispondente.

**Evidenza empirica**: Gli individui felici ricordano meglio il materiale positivo; gli individui tristi ricordano meglio il materiale negativo. Gli effetti MCM sono più robusti per gli umoris positivi che negativi.

**Implicazione**: Il segnale `emotional_congruence` nel retrieval multi-segnale deve misurare il match tra il Mood corrente del sistema e il `mood_snapshot` della memoria al momento dell'encoding.

---

### State-Dependent Learning (Eich, 1989)

Il retrieval è potenziato quando lo stato interno (umore, arousal fisiologico, stato farmacologico) al retrieval corrisponde allo stato durante l'encoding, indipendentemente dalla valenza emotiva del materiale.

Distinto dalla memoria mood-congruente: la memoria state-dependent riguarda il **match tra contesti di encoding e retrieval**, non il match tra umore e valenza del materiale.

**Implicazione**: Il Mood al momento dell'encoding deve essere catturato come parte del tag (`mood_snapshot`). Il retrieval con Mood molto diverso dal `mood_snapshot` può penalizzare il score di una memoria altrimenti rilevante.

---

## 8. Prediction Error e Apprendimento Affettivo

### Dopaminergic Reward Prediction Error (Schultz, 1997)

Il segnale di errore di predizione della ricompensa (RPE) dopaminergico, caratterizzato da Wolfram Schultz, è un meccanismo centrale che collega l'emozione all'apprendimento:
- Neuroni dopaminergici codificano la **differenza tra ricompensa attesa e ricevuta**
- RPE positivo (migliore del previsto) → aumento della dopamina → rinforza il comportamento
- RPE negativo (peggiore del previsto) → sopprime la dopamina → guida l'aggiustamento

### Segnali Neurali Separabili per RPE e Affective Prediction Error (2025)

Ricerca recente (Nature Communications, 2025) ha identificato segnali neurali **separabili** per RPE e **errori di predizione affettiva** — deviazioni dalle aspettative emotive che guidano l'apprendimento indipendentemente dal valore della ricompensa, attraverso interazioni striato-amigdala.

**Implicazione per il riconsolidamento**: Quando una memoria viene recuperata e l'evento corrente genera un affective prediction error significativo (la realtà differisce dall'attesa emotiva codificata nella memoria), si apre la finestra di riconsolidamento. Questo è il meccanismo computazionale che triggera l'aggiornamento del `EmotionalTag`.

---

## 9. Sviluppi Recenti (2020-2026)

### Firme Neurali Generalizzabili della Memoria Emotiva (2025)

Un preprint (bioRxiv, 2025) ha identificato pattern cross-individuali di attività cerebrale che predicono la formazione della memoria emotiva — potenzialmente aprendo la strada a biomarker per encoding emotivo di successo.

### Affective Computing: Stato dell'Arte 2026

La revisione sistematica di Fang et al. (2025) dimostra che la fusione multimodale (espressione facciale + prosodia del parlato + testo + segnali fisiologici) supera i modelli unimodali. I transformer cross-modali dominano.

**Il gap MemEmo (2026)**: Il benchmark MemEmo ha dimostrato che nessun sistema di memoria per LLM attuale gestisce correttamente l'emozione attraverso estrazione, aggiornamento e question-answering. Questo è il gap che la nostra libreria mira a colmare.

---

## Sintesi: Dal Neurone all'Architettura

| Fenomeno neuroscientifico | Parametro nella libreria |
|---------------------------|--------------------------|
| Attivazione amigdala → tagging ippocampale | `consolidation_strength = f(arousal_at_encoding)` |
| Doppia via LeDoux (fast/slow) | Fast encoding (grezzo) + Slow encoding (appraisal completo) |
| Curva Yerkes-Dodson | `consolidation_strength` non-lineare rispetto ad arousal |
| Riconsolidamento al retrieval | Finestra di aggiornamento del `EmotionalTag` post-retrieval |
| Somatic Marker Hypothesis | `EmotionalTag` come bias di navigazione decisionale |
| Mood-Congruent Memory | Peso `emotional_congruence` nel retrieval scoring |
| State-Dependent Learning | `mood_snapshot` nell'`EmotionalTag` + match al retrieval |
| Affective Prediction Error | Trigger per apertura finestra di riconsolidamento |
| Panksepp 7 sistemi | Possibile base per categorie discrete di emozione |
| Costruzionismo di Barrett | Etichette discrete come pattern emergenti nello spazio continuo |
| SECs di Scherer | Layer 4: Appraisal Vector (4 dimensioni) |

---

## Note Bibliografiche

- LeDoux, J.E. (1996). *The Emotional Brain*. Simon & Schuster.
- LeDoux, J.E. (2000). "Emotion circuits in the brain." *Annual Review of Neuroscience*, 23, 155-184.
- McGaugh, J.L. (2004). "The amygdala modulates the consolidation of memories of emotionally arousing experiences." *Annual Review of Neuroscience*.
- McGaugh, J.L. (2013). "Making lasting memories: Remembering the significant." *PNAS*.
- Kensinger, E.A. & Corkin, S. (2004). "Two routes to emotional memory: Distinct neural processes for valence and arousal." *PNAS*.
- Richter-Levin, G. (2004). "The Amygdala, the Hippocampus, and Emotional Modulation of Memory." *The Neuroscientist*.
- Damasio, A.R. (1994). *Descartes' Error: Emotion, Reason, and the Human Brain*. Putnam.
- Damasio, A.R. (1996). "The somatic marker hypothesis and the possible functions of the prefrontal cortex." *Phil. Trans. Royal Society B*.
- Panksepp, J. (1998). *Affective Neuroscience: The Foundations of Human and Animal Emotions*. Oxford University Press.
- Brown, R. & Kulik, J. (1977). "Flashbulb memories." *Cognition*, 5(1), 73-99.
- Talarico, J.M. & Rubin, D.C. (2003). "Confidence, not consistency, characterizes flashbulb memories." *Psychological Science*, 14(5), 455-461.
- Nader, K., Schafe, G.E. & LeDoux, J.E. (2000). "Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval." *Nature*, 406, 722-726.
- Schiller, D. et al. (2010). "Preventing the return of fear in humans using reconsolidation update mechanisms." *Nature*, 463, 49-53.
- Lee, J.L.C., Nader, K. & Schiller, D. (2017). "An update on memory reconsolidation updating." *Trends in Cognitive Sciences*, 21(7), 531-545.
- Lazarus, R.S. (1991). *Emotion and Adaptation*. Oxford University Press.
- Scherer, K.R. (2009). "The dynamic architecture of emotion: Evidence for the component process model." *Cognition and Emotion*.
- Barrett, L.F. (2017). *How Emotions Are Made: The Secret Life of the Brain*. Houghton Mifflin Harcourt.
- Barrett, L.F. (2017). "The theory of constructed emotion: an active inference account of interoception and categorization." *Social Cognitive and Affective Neuroscience*, 12(1), 1-23.
- Bower, G.H. (1981). "Mood and memory." *American Psychologist*, 36(2), 129-148.
- Eich, E. & Metcalfe, J. (1989). "Mood dependent memory for internal versus external events." *Journal of Experimental Psychology: Learning, Memory, and Cognition*.
- Schultz, W., Dayan, P. & Montague, P.R. (1997). "A neural substrate of prediction and reward." *Science*, 275, 1593-1599.
- Nature Communications (2025). "Separable neural signals for reward and emotion prediction errors."
