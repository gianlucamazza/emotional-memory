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

I benchmark di performance usano embedder hash-based e store in-memory. I numeri di latenza (es. retrieve top-5 su 10.000 memorie) non riflettono performance su embedder reali (sentence-transformers, OpenAI) né su hardware eterogeneo.

### 2.3 Assenza di dataset di riferimento pubblico

Non esiste un dataset affect-labeled standardizzato incluso nel repository per riprodurre i confronti cross-system. Ogni comparazione con Mem0, Letta, Zep è attualmente ipotetica.

---

## 3. Limiti Architetturali

### 3.1 Appraisal dipendente da LLM esterno

`LLMAppraisalEngine` richiede una chiamata a un LLM esterno (OpenAI-compatible) per produrre `AppraisalVector`. Questo introduce:

- **Latenza**: 200–2000ms per encoding in modalità slow-path.
- **Costo**: dipendente dal provider e dal modello.
- **Non-determinismo**: Lo stesso testo può produrre appraisal diversi in chiamate successive.
- **Dipendenza di rete**: Encoding non funziona offline senza `KeywordAppraisalEngine` come fallback.

Il `KeywordAppraisalEngine` è un fallback rule-based con copertura limitata (~50 keyword classi). Non generalizza a dominio libero.

### 3.2 Stato affettivo in-process

`AffectiveState` è un oggetto Python in-memory, non persistito implicitamente tra sessioni. Deve essere serializzato/ripristinato manualmente via `save_state()` / `load_state()`. In deployment multi-istanza (più agenti che condividono lo stesso "utente emotivo"), non c'è sincronizzazione dello stato affettivo.

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

I seguenti limiti sono esplicitamente pianificati per versioni future:

| Limite | Versione target |
|---|---|
| Dataset affect-labeled pubblico per benchmark cross-system | v0.6 |
| Adapter Qdrant / Chroma | v0.7 |
| Validazione ecologica leggera (human eval) | v0.8 |
| Dimensione di dominanza come dimensione primaria | v0.8 |
| Stato affettivo distribuito (Redis backend) | v0.9 |

---

*Documento aggiunto in v0.5.1. Aggiornare ad ogni release significativa.*
