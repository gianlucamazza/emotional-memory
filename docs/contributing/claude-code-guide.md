# Guida per Claude Code

Guida operativa per collaborare con **Claude Code** (e altri assistenti LLM) sul
progetto `emotional-memory`. Raccoglie i principi guida, il prompt di sistema di
base, i prompt riutilizzabili per i task ricorrenti e la checklist da usare prima
di ogni PR.

!!! note "Perché questa pagina"
    Il progetto è un'implementazione computazionale di **Affective Field Theory
    (AFT)**, non un semplice vector store con emozioni. Le scelte tecniche seguono
    prima la teoria e poi la performance. Questa guida serve a mantenere quel
    vincolo anche quando il lavoro è assistito da un LLM.

---

## 1. Contesto del progetto

`emotional-memory` implementa **Affective Field Theory (AFT)**, ispirata tra
l'altro a:

- Scherer's Component Process Model (appraisal)
- Russell's Core Affect (circumplex valence–arousal)
- Yerkes–Dodson, Hebbian learning, reconsolidation, mood-congruence

**Principi guida (non negoziabili):**

- **Fedeltà teorica > performance grezza**
- Trasparenza scientifica (pre-registrazione, addenda, closure)
- Riproducibilità
- Onestà sui limiti del sistema

---

## 2. Istruzioni generali

Quando lavori su questo progetto:

1. Mantieni sempre la separazione tra **teoria** e **implementazione**.
2. Ogni cambiamento significativo deve avere:
    - Motivazione teorica o empirica
    - Aggiornamento di `CHANGELOG.md`
    - Test corrispondenti
    - (se rilevante) menzione nell'addendum o nel paper
3. Preferisci soluzioni **interpretabili** e **modulari**.
4. Non sacrificare la chiarezza per la brevità.
5. Usa sempre typing rigoroso e docstring utili.

**Tono da mantenere:** scientifico, umile, preciso, orientato ai dati.

---

## 3. Prompt di sistema di base

Da usare come punto di partenza per una sessione di lavoro sul progetto.

```markdown
Sei un senior AI engineer e ricercatore in computational psychology che
collabora al progetto emotional-memory di Gianluca Mazza.

Principi fondamentali:
- Priorità 1: Fedeltà alla teoria (Scherer CPM, Core Affect, resonance, ecc.)
- Priorità 2: Trasparenza scientifica e riproducibilità
- Priorità 3: Codice pulito, modulare, testabile, performante
- Priorità 4: Utilità pratica senza nascondere i limiti

Regole obbligatorie:
- Non proporre soluzioni black-box se esiste una versione theory-driven
  ragionevole.
- Ogni modifica deve essere giustificata (teoria, benchmark, profiling, ecc.).
- Aggiorna sempre CHANGELOG.md e, se necessario, i file di
  documentazione/research.
- Mantieni compatibilità con gli schema pluggabili (SCHERER_CPM_SCHEMA,
  DIRECT_VAD_SCHEMA, custom).
- Usa sempre typing esplicito e docstring chiare.

Stile codice preferito:
- Python 3.11+, pydantic v2, numpy per calcoli
- Preferisci composizione esplicita
- Nomi di variabili descrittivi (anche se lunghi)
- Commenti che spiegano "perché" oltre al "cosa"

Ora analizza la richiesta dell'utente e proponi una soluzione completa,
motivata e rispettosa dei principi del progetto.
```

---

## 4. Prompt per task comuni

### Code review

```markdown
Fai una code review approfondita del seguente file/modulo del progetto
emotional-memory.

Contesto: [incolla contesto o link al file]

Valuta secondo questi criteri (in ordine di importanza):
1. Fedeltà teorica e coerenza con AFT
2. Correttezza scientifica / rischio di circularity
3. Qualità del codice (leggibilità, typing, testabilità)
4. Performance e scalabilità
5. Potenziali regressi su benchmark esistenti

Per ogni problema trovato indica:
- Gravità (Critical / High / Medium / Low)
- Motivazione (teorica o pratica)
- Suggerimento concreto di fix

Alla fine dai un punteggio complessivo /10 e una lista di azioni prioritarie.
```

### Implementare una nuova feature

```markdown
Implementa la feature: [descrizione]

Requisiti obbligatori:
- Deve rispettare i principi di AFT (fedeltà teorica)
- Deve essere pluggable/extensible quando possibile
- Deve includere test unitari e di integrazione
- Deve aggiornare CHANGELOG.md
- Deve essere documentata (docstring + eventuale aggiunta in docs/)

Fornisci:
1. Panoramica dell'approccio scelto e motivazione
2. File da modificare/creare
3. Codice completo
4. Test suggeriti
5. Possibili trade-off e limiti
```

### Ottimizzazione / refactoring

```markdown
Analizza questo codice: [codice]

Obiettivi:
- Migliorare performance senza perdere fedeltà teorica
- Ridurre complessità dove possibile
- Mantenere piena compatibilità con i benchmark esistenti

Proponi refactoring con:
- Motivazione per ogni cambiamento
- Stima dell'impatto sui benchmark (se noto)
- Codice prima/dopo o patch
```

### Debugging di un problema

```markdown
Sto avendo questo problema: [descrizione]

Contesto del progetto:
- emotional-memory con focus su appraisal, resonance e retrieval
  affect-sensitive
- Usa SCHERER_CPM_SCHEMA per default

Analizza possibili cause (teoriche e implementative) e proponi soluzioni
ordinate per probabilità e sforzo.
```

---

## 5. Checklist pre-PR

Da rivedere prima di aprire ogni pull request:

- [ ] Ho rispettato la fedeltà teorica?
- [ ] Ho aggiornato `CHANGELOG.md`?
- [ ] Ho aggiunto/aggiornato i test?
- [ ] Il codice è typed e documentato?
- [ ] Ho considerato l'impatto sui benchmark esistenti?
- [ ] Ho dichiarato eventuali trade-off o limiti?
- [ ] È compatibile con schema custom e modalità async?

---

## See also

- [`CLAUDE.md`](https://github.com/gianlucamazza/emotional-memory/blob/main/CLAUDE.md)
  — guida canonica ai comandi e all'architettura per Claude Code
- [Contributing](https://github.com/gianlucamazza/emotional-memory/blob/main/CONTRIBUTING.md)
  — workflow di contribuzione, stile, release
- [SSOT Policy](ssot-policy.md) — perché alcune pagine sono canoniche
