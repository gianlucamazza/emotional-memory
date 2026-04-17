"""Generate affect_reference_v1.jsonl — synthetic affect-labeled dataset.

Produces 240 examples (60 per Russell circumplex quadrant):
  Q1: High Valence / High Arousal  (joy, excitement, enthusiasm)
  Q2: Low Valence  / High Arousal  (fear, anger, anxiety, panic)
  Q3: Low Valence  / Low Arousal   (sadness, depression, boredom)
  Q4: High Valence / Low Arousal   (calm, contentment, serenity)

Each example has: id, text, valence [-1,1], arousal [0,1], dominance [-1,1],
expected_label (Plutchik primary), source="synthetic-v1".

Usage::

    python benchmarks/datasets/generate_dataset.py
    # writes benchmarks/datasets/affect_reference_v1.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).parent / "affect_reference_v1.jsonl"

# ---------------------------------------------------------------------------
# Seed texts by quadrant and intensity (strong / moderate / mild)
# ---------------------------------------------------------------------------

_Q1_STRONG = [
    "I just got the promotion I've been working toward for years!",
    "Finally got the research paper accepted after three rejections!",
    "My child took their first steps today — pure joy!",
    "We beat the deadline and the client loved every detail!",
    "The crowd erupted when we scored — I've never felt so alive!",
    "I'm vibrating with excitement — just got into my dream university!",
    "This raise is life-changing — I'm overjoyed and energised!",
    "My proposal was approved! I'm jumping for joy!",
    "The show was a standing ovation — I'm still buzzing!",
    "We launched today and the response is overwhelming — sheer elation!",
    "We won the championship! This is the best day of my life!",
    "I'm bursting with excitement — our startup just closed a Series A!",
    "I feel absolutely unstoppable right now.",
    "Dancing with joy — everything is going perfectly!",
    "Pure euphoria — I can't believe how amazing this moment is.",
    "My heart is racing with delight, I want to shout from the rooftops!",
    "This is thrilling — every cell in my body is alive with energy!",
    "I'm ecstatic beyond words. Nothing could bring me down today.",
    "We did it! Absolute triumph. I'm overwhelmed with happiness.",
]
_Q1_MOD = [
    "I got a really good grade on my thesis — feeling proud and energized.",
    "A surprise birthday party was thrown for me. I'm genuinely touched.",
    "The meeting went better than expected and now I feel motivated.",
    "Just heard some great news from a friend — smiling widely.",
    "Finished a big project and feel a surge of satisfied energy.",
    "That concert was incredible — still buzzing with excitement.",
    "I feel inspired and ready to take on the world today.",
    "Got a compliment from my mentor — feeling warm and motivated.",
    "The collaboration worked beautifully — happy and energized.",
    "I love this challenge. Feeling fired up and confident.",
]
_Q1_MILD = [
    "Today was a pleasant day — things went smoothly.",
    "I'm in a good mood after a nice lunch with colleagues.",
    "Feeling cheerful — the weather is lovely.",
    "A small win at work put a smile on my face.",
    "Content and lightly optimistic about the week ahead.",
    "The weekend trip was fun, nothing extraordinary but enjoyable.",
    "Feeling a bit upbeat — got some encouraging feedback.",
    "Light-hearted and curious about what the afternoon will bring.",
    "Had a good conversation; feeling warmly satisfied.",
    "Nothing dramatic, just a generally pleasant, energized day.",
]

_Q2_STRONG = [
    "The test results came back and the news is devastating. I'm terrified.",
    "I can't stop shaking — the accident was horrifying.",
    "Rage is overwhelming me. This injustice is completely unacceptable!",
    "I'm paralysed with fear. I don't know what to do.",
    "My whole body is tense with fury — I have never felt this angry.",
    "Absolute panic — the presentation starts in five minutes and I'm blank.",
    "I'm livid. This betrayal is unforgivable and I'm shaking with anger.",
    "Terror grips me — I feel completely trapped with no way out.",
    "The news sent shockwaves through me. I'm horrified and furious.",
    "I'm desperate and frantic — something has gone very, very wrong.",
]
_Q2_MOD = [
    "Worried about the upcoming interview — my stomach is in knots.",
    "I'm frustrated with the situation. It keeps getting worse.",
    "There's a creeping anxiety I can't shake before the deadline.",
    "I feel annoyed and agitated — nothing is working out today.",
    "The uncertainty is making me anxious and on edge.",
    "I dread the confrontation I know I have to have.",
    "Tense and irritable — too many things going wrong at once.",
    "Nervous about the exam results coming out tomorrow.",
    "Something feels off and I'm unsettled and alert.",
    "Frustrated by the repeated miscommunication — it feels endless.",
]
_Q2_MILD = [
    "A little uneasy about the decision I made — second-guessing myself.",
    "Slightly nervous before the presentation, nothing too serious.",
    "I feel mildly irritated by the delay.",
    "There's a low hum of worry in the back of my mind.",
    "Somewhat anxious but trying to stay calm.",
    "A small knot of concern about tomorrow's meeting.",
    "Mildly stressed — too many tasks queued up.",
    "Feeling a bit edgy after reading the news.",
    "Slight unease about the feedback I might receive.",
    "Low-level tension that I can manage but haven't fully resolved.",
]

_Q3_STRONG = [
    "Everything feels hopeless. I can't see a way forward anymore.",
    "Overwhelming grief — I miss them so much it's physically painful.",
    "I am utterly exhausted and deeply depressed. Nothing matters.",
    "Profound sadness weighs on me. I can barely get out of bed.",
    "I feel completely hollow and disconnected from the world.",
    "Despair has settled in. I no longer believe things will get better.",
    "Deep sorrow consumes me — I feel utterly alone.",
    "The loss is unbearable. I am numb and broken.",
    "Hopelessness and fatigue — I'm running on empty, emotionally.",
    "Dark thoughts swirl. I feel crushed under the weight of everything.",
]
_Q3_MOD = [
    "I'm feeling down today — disheartened by how things turned out.",
    "A sense of melancholy has been following me all week.",
    "Tired and a bit sad — the project I cared about got cancelled.",
    "I feel low and unmotivated. Not sure where my energy went.",
    "Disappointment settles in after the rejection letter arrived.",
    "Feeling lonely and somewhat empty after the goodbye.",
    "The grey weather matches my mood — subdued and a bit blue.",
    "I'm emotionally drained and feeling pessimistic about the future.",
    "A quiet sadness has been with me since the news broke.",
    "Discouraged — I've been trying so hard but nothing seems to work.",
]
_Q3_MILD = [
    "A little bored and restless — nothing particularly interesting today.",
    "Feeling slightly flat, not sad exactly, just low energy.",
    "Mild disappointment that the event was cancelled.",
    "I'm a bit disengaged this afternoon — hard to focus.",
    "Slightly glum without any clear reason.",
    "Listless — going through the motions but not really present.",
    "A minor sense of emptiness after a long, uneventful week.",
    "A touch of melancholy, nothing overwhelming.",
    "Feeling somewhat indifferent and slow today.",
    "Mildly blue — nothing dramatic, just a quiet low.",
]

_Q4_STRONG = [
    "Lying in a meadow watching clouds — absolute peace and serenity.",
    "Deep contentment washes over me. I am exactly where I need to be.",
    "Profound tranquility after the meditation retreat. Fully at ease.",
    "A sense of complete gratitude and peace — nothing more is needed.",
    "Total inner stillness. The world is good and I am part of it.",
    "Blissful quiet after a long walk in nature. No worries, just calm.",
    "Serene and fully present — this moment is enough.",
    "A deep, settled happiness with no need for anything more.",
    "At peace with everything. Gentle warmth in my chest.",
    "Pure restful contentment — I feel whole and unhurried.",
]
_Q4_MOD = [
    "Relaxed after a good night's sleep — ready for the day, no rush.",
    "Content with how the week went. Looking forward to a quiet evening.",
    "Pleasantly satisfied — dinner was great and the conversation easy.",
    "A comfortable sense of well-being, nothing dramatic, just good.",
    "Calm and grateful — appreciating the simple things today.",
    "Feeling settled and warmly positive about things.",
    "Gently optimistic — things are going well and I feel grounded.",
    "A quiet joy in the routine — morning coffee, good book, soft light.",
    "Mild happiness and ease. The day is unfolding nicely.",
    "Feeling balanced — no highs or lows, just steady contentment.",
]
_Q4_MILD = [
    "Mildly pleased and relaxed after finishing my to-do list.",
    "A gentle sense of okayness — nothing special, just fine.",
    "Somewhat calm today, nothing urgent pressing.",
    "Low-key satisfied with how the morning went.",
    "Quietly comfortable — no stress, no strong feelings either way.",
    "Faintly content, like a cup of warm tea on a cool afternoon.",
    "Mildly at ease. Not particularly excited but not worried either.",
    "A background hum of gentle satisfaction.",
    "Soft, unhurried calm this afternoon.",
    "Quietly happy in an understated way.",
]

# ---------------------------------------------------------------------------
# PAD values by quadrant x intensity
# ---------------------------------------------------------------------------

# (valence_mean, valence_std, arousal_mean, arousal_std, dominance_mean, dominance_std)
_PAD_PARAMS: dict[str, tuple[float, float, float, float, float, float]] = {
    "Q1_strong": (0.80, 0.10, 0.85, 0.08, 0.60, 0.10),
    "Q1_mod": (0.55, 0.10, 0.60, 0.10, 0.40, 0.10),
    "Q1_mild": (0.30, 0.08, 0.40, 0.10, 0.20, 0.10),
    "Q2_strong": (-0.75, 0.10, 0.85, 0.08, -0.60, 0.12),
    "Q2_mod": (-0.50, 0.10, 0.62, 0.10, -0.35, 0.12),
    "Q2_mild": (-0.25, 0.08, 0.42, 0.10, -0.15, 0.10),
    "Q3_strong": (-0.80, 0.08, 0.18, 0.08, -0.65, 0.10),
    "Q3_mod": (-0.55, 0.10, 0.28, 0.10, -0.40, 0.12),
    "Q3_mild": (-0.25, 0.08, 0.32, 0.08, -0.15, 0.10),
    "Q4_strong": (0.78, 0.08, 0.18, 0.08, 0.50, 0.12),
    "Q4_mod": (0.52, 0.10, 0.28, 0.10, 0.30, 0.10),
    "Q4_mild": (0.28, 0.08, 0.32, 0.08, 0.15, 0.10),
}

_LABELS: dict[str, str] = {
    "Q1_strong": "joy",
    "Q1_mod": "joy",
    "Q1_mild": "trust",
    "Q2_strong": "fear",
    "Q2_mod": "anger",
    "Q2_mild": "anticipation",
    "Q3_strong": "sadness",
    "Q3_mod": "sadness",
    "Q3_mild": "disgust",
    "Q4_strong": "trust",
    "Q4_mod": "trust",
    "Q4_mild": "trust",
}

_TEXTS: dict[str, list[str]] = {
    "Q1_strong": _Q1_STRONG,
    "Q1_mod": _Q1_MOD,
    "Q1_mild": _Q1_MILD,
    "Q2_strong": _Q2_STRONG,
    "Q2_mod": _Q2_MOD,
    "Q2_mild": _Q2_MILD,
    "Q3_strong": _Q3_STRONG,
    "Q3_mod": _Q3_MOD,
    "Q3_mild": _Q3_MILD,
    "Q4_strong": _Q4_STRONG,
    "Q4_mod": _Q4_MOD,
    "Q4_mild": _Q4_MILD,
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _jitter(mean: float, std: float, lo: float, hi: float, rng: random.Random) -> float:
    return _clamp(rng.gauss(mean, std), lo, hi)


def generate(seed: int = 42, samples_per_text: int = 2) -> list[dict]:  # type: ignore[type-arg]
    """Generate *samples_per_text* PAD-jittered rows for every seed text.

    Default samples_per_text=2 → 240 examples from 120 seed texts.
    Each copy gets independent noise so PAD distributions remain realistic.
    """
    rng = random.Random(seed)
    examples = []
    idx = 0
    for bucket, texts in _TEXTS.items():
        vm, vs, am, as_, dm, ds = _PAD_PARAMS[bucket]
        label = _LABELS[bucket]
        for text in texts:
            for _ in range(samples_per_text):
                v = round(_jitter(vm, vs, -1.0, 1.0, rng), 3)
                a = round(_jitter(am, as_, 0.0, 1.0, rng), 3)
                d = round(_jitter(dm, ds, -1.0, 1.0, rng), 3)
                examples.append(
                    {
                        "id": f"sv1_{idx:04d}",
                        "text": text,
                        "valence": v,
                        "arousal": a,
                        "dominance": d,
                        "expected_label": label,
                        "source": "synthetic-v1",
                    }
                )
                idx += 1
    return examples


def main() -> None:
    examples = generate()
    OUT.write_text("\n".join(json.dumps(e) for e in examples) + "\n")
    print(f"Written {len(examples)} examples → {OUT.relative_to(OUT.parent.parent.parent)}")
    quads = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    for e in examples:
        if e["valence"] > 0 and e["arousal"] > 0.5:
            quads["Q1"] += 1
        elif e["valence"] < 0 and e["arousal"] > 0.5:
            quads["Q2"] += 1
        elif e["valence"] < 0:
            quads["Q3"] += 1
        else:
            quads["Q4"] += 1
    for q, n in quads.items():
        print(f"  {q}: {n}")


if __name__ == "__main__":
    main()
