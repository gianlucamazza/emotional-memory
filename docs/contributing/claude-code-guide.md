# Claude Code Guide

Operative guide for collaborating with **Claude Code** (and other LLM assistants)
on the `emotional-memory` project. It collects the guiding principles, the base
system prompt, reusable prompts for recurring tasks, and the checklist to run
before every PR.

!!! note "Why this page"
    The project is a computational implementation of **Affective Field Theory
    (AFT)**, not a simple vector store with emotions bolted on. Technical choices
    follow theory first and performance second. This guide exists to keep that
    constraint in force even when the work is LLM-assisted.

---

## 1. Project context

`emotional-memory` implements **Affective Field Theory (AFT)**, drawing on:

- Scherer's Component Process Model (appraisal)
- Russell's Core Affect (valence–arousal circumplex)
- Yerkes–Dodson, Hebbian learning, reconsolidation, mood-congruence

**Guiding principles (non-negotiable):**

- **Theory fidelity > raw performance**
- Scientific transparency (pre-registration, addenda, closure)
- Reproducibility
- Honesty about the system's limitations

---

## 2. General instructions

When working on this project:

1. Always keep **theory** and **implementation** separate.
2. Every significant change must have:
    - A theoretical or empirical rationale
    - A `CHANGELOG.md` update
    - Corresponding tests
    - (where relevant) a mention in the addendum or the paper
3. Prefer **interpretable** and **modular** solutions.
4. Do not trade clarity for brevity.
5. Always use rigorous typing and useful docstrings.

**Tone to maintain:** scientific, humble, precise, data-driven.

---

## 3. Base system prompt

Use this as the starting point for a working session on the project.

```markdown
You are a senior AI engineer and computational-psychology researcher
collaborating on Gianluca Mazza's emotional-memory project.

Core principles:
- Priority 1: Fidelity to theory (Scherer CPM, Core Affect, resonance, etc.)
- Priority 2: Scientific transparency and reproducibility
- Priority 3: Clean, modular, testable, performant code
- Priority 4: Practical usefulness without hiding the limitations

Mandatory rules:
- Do not propose black-box solutions when a reasonable theory-driven version
  exists.
- Every change must be justified (theory, benchmark, profiling, etc.).
- Always update CHANGELOG.md and, when needed, the documentation/research
  files.
- Preserve compatibility with the pluggable schemas (SCHERER_CPM_SCHEMA,
  DIRECT_VAD_SCHEMA, custom).
- Always use explicit typing and clear docstrings.

Preferred code style:
- Python 3.11+, pydantic v2, numpy for computation
- Prefer explicit composition
- Descriptive variable names (even if long)
- Comments that explain the "why", not just the "what"

Now analyze the user's request and propose a complete, justified solution
that respects the project's principles.
```

---

## 4. Prompts for common tasks

### Code review

```markdown
Do a thorough code review of the following file/module of the
emotional-memory project.

Context: [paste context or link to the file]

Evaluate against these criteria (in order of importance):
1. Theory fidelity and coherence with AFT
2. Scientific correctness / circularity risk
3. Code quality (readability, typing, testability)
4. Performance and scalability
5. Potential regressions on existing benchmarks

For each issue found, state:
- Severity (Critical / High / Medium / Low)
- Rationale (theoretical or practical)
- A concrete fix suggestion

At the end, give an overall score /10 and a prioritized action list.
```

### Implement a new feature

```markdown
Implement the feature: [description]

Mandatory requirements:
- Must respect AFT principles (theory fidelity)
- Must be pluggable/extensible where possible
- Must include unit and integration tests
- Must update CHANGELOG.md
- Must be documented (docstring + docs/ addition where relevant)

Provide:
1. Overview of the chosen approach and its rationale
2. Files to modify/create
3. Complete code
4. Suggested tests
5. Possible trade-offs and limitations
```

### Optimization / refactoring

```markdown
Analyze this code: [code]

Goals:
- Improve performance without losing theory fidelity
- Reduce complexity where possible
- Keep full compatibility with existing benchmarks

Propose the refactor with:
- A rationale for each change
- An estimate of the benchmark impact (if known)
- Before/after code or a patch
```

### Debugging a problem

```markdown
I'm hitting this problem: [description]

Project context:
- emotional-memory, focused on appraisal, resonance and affect-sensitive
  retrieval
- Uses SCHERER_CPM_SCHEMA by default

Analyze possible causes (theoretical and implementation) and propose
solutions ordered by likelihood and effort.
```

---

## 5. Pre-PR checklist

Review before opening every pull request:

- [ ] Did I respect theory fidelity?
- [ ] Did I update `CHANGELOG.md`?
- [ ] Did I add/update tests?
- [ ] Is the code typed and documented?
- [ ] Did I consider the impact on existing benchmarks?
- [ ] Did I declare any trade-offs or limitations?
- [ ] Is it compatible with custom schemas and async mode?

---

## See also

- [`CLAUDE.md`](https://github.com/gianlucamazza/emotional-memory/blob/main/CLAUDE.md)
  — canonical command and architecture guide for Claude Code
- [Contributing](https://github.com/gianlucamazza/emotional-memory/blob/main/CONTRIBUTING.md)
  — contribution workflow, style, release
- [SSOT Policy](ssot-policy.md) — why some pages are canonical
