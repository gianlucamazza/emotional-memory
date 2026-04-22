# Limiti Noti di Affective Field Theory

> Questa sezione documenta i limiti riconosciuti dell'implementazione attuale. La trasparenza sui limiti è parte integrante della rivendicazione di rigore scientifico.

---

## 1. Limiti del Modello Affettivo

### 1.1 Dimensionalità dell'affetto

Il modello usa il circomplesso a 2 dimensioni di Russell (valenza–arousal). Questo è il modello più citato e computazionalmente trattabile, ma presenta limitazioni documentate:

- **Non cattura la dominanza (PAD completo)**: Il modello PAD di Mehrabian & Russell (1974) include una terza dimensione di *dominance* (controllo/sottomissione). Nella versione attuale, `CoreAffect` modella solo valenza e arousal; la dominanza è derivata opzionalmente via `AppraisalVector.dominance`. Non è ancora una dimensione primaria del campo.
- **Struttura circomplessa contestata**: Russell (2003) stesso ha rivisto la teoria verso il "core affect" come costruzione contestuale. Il modello attuale usa la forma classica del 1980, non le revisioni costruzioniste.
- **Granularità emozionale limitata**: La mappatura Plutchik a 8 emozioni primarie è una discretizzazione del continuo. Emozioni complesse (nostalgia, Schadenfreude, awe) non hanno rappresentazione diretta.

### 1.2 Dipendenza linguistica

- **Ottimizzato per inglese e italiano**: Il `KeywordAppraisalEngine` usa regole in inglese. Il `LLMAppraisalEngine` dipende dal modello LLM sottostante — funziona meglio con testi in lingue ad alta presenza nel pretraining.
- **Nessuna validazione cross-linguistica formale**: Non esistono benchmark che testino la coerenza psicologica delle predizioni affettive su lingue diverse dall'inglese.

---

## 2. Limiti della Validazione Empirica

### 2.1 Nessuna validazione con utenti umani

I 126 test di fedeltà psicologica validano che il sistema si comporta *coerentemente con le teorie* che implementa (ad esempio, che il recupero è mood-congruente, che il decadimento segue una power-law). Non validano che il comportamento del sistema corrisponda a come *gli esseri umani reali* formano e recuperano memorie emotive.

Questa è la distinzione critica tra **validazione intra-teorica** (testata) e **validazione ecologica** (non testata). Un sistema che implementa fedelmente una teoria sbagliata supera tutti i test di fedeltà.

### 2.2 Benchmark comparativi sintetici

Il repository include oggi un benchmark comparativo controllato
(`benchmarks/comparative/`) su `affect_reference_v1`, un dataset sintetico di 258
esempi affect-labeled con 4 query di retrieval mood-congruente. Questo benchmark
e' utile per misurare se AFT modifica il ranking nella direzione teoricamente
attesa, ma ha limiti importanti:

- **Setup piccolo e sintetico**: non rappresenta conversazioni multi-sessione o task agentici realistici.
- **Protocollo orientato al retrieval affect-aware**: misura recall@k per quadranti affettivi, non answer quality downstream.
- **Affect esplicito per adapter affect-aware**: AFT riceve il contesto affettivo della query tramite `valence/arousal`; i baseline generalisti possono ignorarlo del tutto.
- **Latenze non omogenee**: i numeri mescolano store locali, dipendenze opzionali e sistemi progettati per scopi diversi.

Di conseguenza, i risultati comparativi attuali vanno letti come **evidenza
controllata precoce**, non come prova di superiorita' generale su sistemi di
memoria production-grade.

### 2.3 Dataset pubblici ancora piccoli

Il repository include ora due famiglie di dataset pubblici:

- `affect_reference_v1.jsonl`: benchmark sintetico e mirato al retrieval mood-congruente
- `realistic_recall_v1.json`: benchmark replay multi-sessione, ancora piccolo e scripted

Questo migliora la riproducibilita' rispetto alle versioni precedenti. Il
benchmark realistico ora rifiuta query triviali (`candidate_count <= top_k`) e
usa `top1_accuracy` come metrica primaria invece di affidarsi solo a `hit@k`.
Tuttavia non equivale ancora a un dataset standardizzato, ampio e
realisticamente multi-turno capace di sostenere confronti forti tra sistemi su
memoria emotiva.

Le comparazioni con `Mem0` e `LangMem` non sono piu' ipotetiche: il repository
include adapter e risultati riproducibili nel benchmark controllato. Tuttavia
queste comparazioni restano limitate dal protocollo sintetico, dalla diversa
superficie funzionale dei sistemi, e dall'assenza di human eval o task
downstream realistici. `Letta` resta non valutato senza accesso al servizio
cloud/API key.

Nel benchmark replay multi-sessione attuale, AFT si separa in modo chiaro da un
baseline puramente `recency` e mantiene un vantaggio aggregato rispetto a
`naive_cosine` nel piccolo dataset replay corrente. Tuttavia questo vantaggio e'
ancora concentrato soprattutto nei query subset di tipo `affective_arc`; il
subset `semantic_confound`, ora ampliato, resta un'area debole e non mostra
ancora un vantaggio AFT rispetto al baseline semantic-only. Quindi non sostiene
ancora claim forti di superiorita' generale su retrieval semantic-only in
scenari realistici piu' ampi.

### 2.4 Pipeline human-eval ancora non eseguita con rating reali

Il repository include ora una pipeline eseguibile in `benchmarks/human_eval/`
per generare packet, template di rating e summary aggregati, con un pilot v1
bloccato su `aft` vs `naive_cosine` e 10 scenari. Questo risolve il gap
procedurale, ma non il gap empirico:

- **Nessun rater reale incluso nel repository**: il pilot non e' ancora stato
  eseguito con partecipanti umani.
- **I template vuoti non contano come evidenza**: la pipeline ora rifiuta
  `ratings.jsonl` lasciati nello stato placeholder.
- **Nessun summary checked-in conta come risultato**: `summary.json` e
  `summary.md` non fanno parte della surface di evidenza finche' il pilot non
  viene eseguito con rating completati.
- **Nessuna misura di accordo inter-rater**: finche' non esistono rating
  completati, non e' possibile stimare affidabilita' o stabilita' del protocollo.

---

## 3. Limiti Architetturali

### 3.1 Appraisal dipendente da LLM esterno

`LLMAppraisalEngine` richiede una chiamata a un LLM esterno (OpenAI-compatible) per produrre `AppraisalVector`. Questo introduce:

- **Latenza**: 200–2000ms per encoding in modalità slow-path.
- **Costo**: dipendente dal provider e dal modello.
- **Non-determinismo**: Lo stesso testo può produrre appraisal diversi in chiamate successive.
- **Dipendenza di rete**: Encoding non funziona offline senza `KeywordAppraisalEngine` come fallback.

Il `KeywordAppraisalEngine` è un fallback rule-based con copertura limitata (~50 keyword classi). Non generalizza a dominio libero.

### 3.2 Stato affettivo: persistenza locale si', condivisione distribuita ancora limitata

`AffectiveState` non e' piu' soltanto uno stato in-process. Il repository
supporta oggi `AffectiveStateStore` dedicati, con backend locali come
`InMemoryAffectiveStateStore` e `SQLiteAffectiveStateStore`, piu' un backend
`RedisAffectiveStateStore` opzionale per stato condiviso.

Restano pero' limiti architetturali importanti:

- **Nessuna validazione production-grade del backend condiviso**: Redis e'
  disponibile come backend opzionale, ma manca ancora una validazione estesa su
  deployment multi-worker reali.
- **Nessuna transazione congiunta memoria+stato**: `MemoryStore` e
  `AffectiveStateStore` restano separati per design; non esiste ancora una
  semantica atomica cross-store.
- **Sincronizzazione distribuita limitata**: il backend Redis copre la
  persistenza dello stato affettivo, non la memoria vettoriale distribuita.

### 3.3 Thread-safety limitata a connessione singola

`SQLiteStore` usa un singolo oggetto `sqlite3.Connection` con `threading.RLock`. Questo serializza gli accessi ma non consente parallelismo tra lettori. Per throughput elevato su store SQLite, una architettura connection-pool sarebbe più scalabile.

### 3.4 Nessun store vettoriale "enterprise"

Gli adapter disponibili sono `InMemoryStore` (RAM, non persistente) e `SQLiteStore` (file locale, scalabile a ~10^6 memorie). Non esistono adapter per sistemi distribuiti come Qdrant, Chroma, Weaviate, Pinecone. Il `MemoryStore` Protocol è duck-typed e i contributi sono benvenuti.

---

## 4. Limiti Teorici

### 4.1 Operazionalizzazione parziale di Heidegger

La `MoodField` operazionalizza *Stimmung* (tonalità emotiva di fondo) come EMA su valenza-arousal con parametri di decadimento. Questa è una necessaria semplificazione computazionale di un concetto fenomenologico che in Heidegger è pre-cognitivo e strutturalmente legato all'essere-nel-mondo. La mappatura è *ispirata*, non *fedele*.

### 4.2 Credenziali della novelty AFT non peer-reviewed

Al momento della release v0.5.x, Affective Field Theory non è stata formalmente pubblicata su una venue peer-reviewed. Le rivendicazioni di originalità (in particolare l'integrazione unificata di Russell + Scherer + Heidegger + Hebb + Collins & Loftus in un unico modello computazionale per LLM memory) sono plausibili ma non ancora validate dalla comunità scientifica.

---

## 5. Lavori Futuri

I seguenti limiti sono esplicitamente pianificati, ma non sono vincolati a una
versione precisa:

| Limite | Orizzonte indicativo |
|---|---|
| Benchmark affect-aware realistico piu' ampio e comparativo | post-0.6 |
| Adapter Qdrant / Chroma | track architettura |
| Esecuzione del pilot human-eval con rating reali | track ricerca |
| Dimensione di dominanza come dimensione primaria | track ricerca |
| Memory-store distribuito / enterprise (Qdrant, Chroma, ...) | track prodotto |

---

*Documento aggiunto in v0.5.1. Aggiornare ad ogni release significativa.*
