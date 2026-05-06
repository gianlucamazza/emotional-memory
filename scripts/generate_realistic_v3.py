"""Generate realistic_recall_v3.json — 75 new scenarios extending v2 to N=500 queries.

Usage (author-blind: run before any AFT benchmark):
    uv run python scripts/generate_realistic_v3.py
Output: benchmarks/datasets/realistic_recall_v3.json
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Challenge-type combo rotation (same 5 as v2, 15 scenarios each)
# ---------------------------------------------------------------------------
COMBOS: list[list[str]] = [
    ["semantic_confound", "affective_arc", "recency_confound", "same_topic_distractor"],
    ["affective_arc", "momentum_alignment", "recency_confound", "same_topic_distractor"],
    ["momentum_alignment", "recency_confound", "same_topic_distractor", "semantic_confound"],
    ["affective_arc", "momentum_alignment", "same_topic_distractor", "semantic_confound"],
    ["affective_arc", "momentum_alignment", "recency_confound", "semantic_confound"],
]

# Each scenario is defined as a dict with:
#   id, desc, events (list of 6), queries_by_type (dict[type -> query_spec])
# query_spec: {query, expected_memory_ids, state}
# Event: {id, content, v, a, meta}

SCENARIOS: list[dict] = [
    # ------------------------------------------------------------------
    # COMBO 0: semantic_confound + affective_arc + recency_confound + same_topic_distractor
    # ------------------------------------------------------------------
    {
        "id": "s51_api_redesign_conflict",
        "desc": "An API redesign project swings from excitement to crisis and partial recovery.",
        "events": [
            {
                "id": "api_kickoff_energy",
                "content": "The API redesign kickoff meeting was electric; the team aligned on a clean breaking-change approach and timeline.",
                "v": 0.8,
                "a": 0.7,
                "meta": {"phase": "start", "actor": "team"},
            },
            {
                "id": "api_legacy_anger",
                "content": "A major client threatened to cancel their contract after discovering the breaking changes would require months of migration work.",
                "v": -0.85,
                "a": 0.9,
                "meta": {"phase": "conflict", "actor": "client"},
            },
            {
                "id": "api_compromise_draft",
                "content": "The team drafted a versioning strategy to keep the legacy API alive for twelve months alongside the new design.",
                "v": 0.3,
                "a": 0.5,
                "meta": {"phase": "negotiation", "actor": "team"},
            },
            {
                "id": "api_client_relief",
                "content": "The client accepted the versioning plan and reduced the cancellation threat to a formal complaint.",
                "v": 0.55,
                "a": 0.4,
                "meta": {"phase": "resolution", "actor": "client"},
            },
            {
                "id": "api_tech_debt_dread",
                "content": "Maintaining two API versions introduced sprawling tech debt that the team described as 'a tax on every sprint'.",
                "v": -0.6,
                "a": 0.6,
                "meta": {"phase": "consequence", "actor": "team"},
            },
            {
                "id": "api_docs_shipped",
                "content": "The migration documentation was finally published and the old API deprecation notice went out quietly.",
                "v": 0.45,
                "a": 0.3,
                "meta": {"phase": "closure", "actor": "team"},
            },
        ],
        "queries": {
            "semantic_confound": {
                "q": "Which API-related memory captures the anger and threat of cancellation — not the positive kickoff or the pragmatic compromise?",
                "exp": ["api_legacy_anger"],
                "state": {"valence": -0.6, "arousal": 0.8},
            },
            "affective_arc": {
                "q": "Which memory marks the emotional peak of dread about maintaining two API versions indefinitely?",
                "exp": ["api_tech_debt_dread"],
                "state": {"valence": -0.5, "arousal": 0.6},
            },
            "recency_confound": {
                "q": "Which earlier session memory captured the initial high energy of the API redesign kickoff, before the client conflict emerged?",
                "exp": ["api_kickoff_energy"],
                "state": {"valence": 0.7, "arousal": 0.6},
            },
            "same_topic_distractor": {
                "q": "Which API-related memory was specifically about the client's positive response to the versioning plan — not the plan's drafting?",
                "exp": ["api_client_relief"],
                "state": {"valence": 0.5, "arousal": 0.35},
            },
        },
    },
    {
        "id": "s52_marathon_injury",
        "desc": "A runner's marathon preparation is derailed by injury then partially restored.",
        "events": [
            {
                "id": "marathon_pb_hope",
                "content": "Eight weeks out from the race, training data showed a personal-best pace was within reach and the runner felt unstoppable.",
                "v": 0.85,
                "a": 0.75,
                "meta": {"phase": "peak", "actor": "runner"},
            },
            {
                "id": "marathon_stress_fracture",
                "content": "A stress fracture in the left tibia was confirmed by MRI; the race was gone and the next six weeks would be non-weight-bearing.",
                "v": -0.9,
                "a": 0.85,
                "meta": {"phase": "injury", "actor": "runner"},
            },
            {
                "id": "marathon_pool_running",
                "content": "Aqua jogging in the hotel pool felt absurd but maintained fitness; the runner found unexpected calm in the water.",
                "v": 0.2,
                "a": 0.4,
                "meta": {"phase": "adaptation", "actor": "runner"},
            },
            {
                "id": "marathon_crutches_frustration",
                "content": "Navigating the office on crutches while colleagues trained outside was quietly humiliating.",
                "v": -0.7,
                "a": 0.7,
                "meta": {"phase": "frustration", "actor": "runner"},
            },
            {
                "id": "marathon_clearance",
                "content": "The physio gave full clearance to run at ten weeks post-injury; the runner broke down crying in the car park.",
                "v": 0.75,
                "a": 0.8,
                "meta": {"phase": "recovery", "actor": "runner"},
            },
            {
                "id": "marathon_spring_race",
                "content": "A spring half-marathon was entered as the comeback race; the finish line felt like the original marathon goal fulfilled.",
                "v": 0.8,
                "a": 0.7,
                "meta": {"phase": "return", "actor": "runner"},
            },
        ],
        "queries": {
            "semantic_confound": {
                "q": "Which marathon-training memory was about the crushing diagnosis and race withdrawal — not the hopeful performance peak?",
                "exp": ["marathon_stress_fracture"],
                "state": {"valence": -0.8, "arousal": 0.8},
            },
            "affective_arc": {
                "q": "Which memory represents the emotional nadir of the injury period — the daily humiliation of being on crutches while others trained?",
                "exp": ["marathon_crutches_frustration"],
                "state": {"valence": -0.65, "arousal": 0.65},
            },
            "recency_confound": {
                "q": "Which early-session memory captured the optimism about a personal-best time, before the injury was even a thought?",
                "exp": ["marathon_pb_hope"],
                "state": {"valence": 0.8, "arousal": 0.7},
            },
            "same_topic_distractor": {
                "q": "Which recovery memory was specifically about the emotional moment of receiving medical clearance — not the return race?",
                "exp": ["marathon_clearance"],
                "state": {"valence": 0.7, "arousal": 0.75},
            },
        },
    },
    {
        "id": "s53_funding_round_collapse",
        "desc": "A startup's Series A falls apart spectacularly then is partially rescued.",
        "events": [
            {
                "id": "termsheet_celebration",
                "content": "The term sheet arrived at 6pm on a Friday; the founders ordered champagne and announced a team dinner.",
                "v": 0.9,
                "a": 0.85,
                "meta": {"phase": "celebration", "actor": "founders"},
            },
            {
                "id": "due_diligence_fraud_flag",
                "content": "Due diligence uncovered a revenue recognition error that the lead investor characterised as 'possibly fraudulent'.",
                "v": -0.95,
                "a": 0.95,
                "meta": {"phase": "crisis", "actor": "investor"},
            },
            {
                "id": "audit_exoneration",
                "content": "The external auditor confirmed the accounting error was a legitimate reclassification, not fraud, but the lead investor had already withdrawn.",
                "v": 0.4,
                "a": 0.6,
                "meta": {"phase": "partial_relief", "actor": "auditor"},
            },
            {
                "id": "bridge_loan_signed",
                "content": "An angel investor provided a bridge loan to extend runway while the team rebuilt the investor relationship.",
                "v": 0.55,
                "a": 0.5,
                "meta": {"phase": "bridge", "actor": "angel"},
            },
            {
                "id": "team_attrition_dread",
                "content": "Two senior engineers resigned after the funding collapse; the founders feared a talent exodus was beginning.",
                "v": -0.8,
                "a": 0.8,
                "meta": {"phase": "consequence", "actor": "team"},
            },
            {
                "id": "revised_terms_signed",
                "content": "A new lead investor stepped in at a lower valuation; the founders signed with relief mixed with grief at the dilution.",
                "v": 0.35,
                "a": 0.45,
                "meta": {"phase": "close", "actor": "founders"},
            },
        ],
        "queries": {
            "semantic_confound": {
                "q": "Which funding-round memory was about the catastrophic fraud allegation — not the joyful term sheet arrival?",
                "exp": ["due_diligence_fraud_flag"],
                "state": {"valence": -0.85, "arousal": 0.9},
            },
            "affective_arc": {
                "q": "Which memory captured the dread that key engineers would continue to leave after the funding collapse?",
                "exp": ["team_attrition_dread"],
                "state": {"valence": -0.75, "arousal": 0.75},
            },
            "recency_confound": {
                "q": "Which early memory captured the unalloyed joy of the term sheet — before any due diligence concerns arose?",
                "exp": ["termsheet_celebration"],
                "state": {"valence": 0.85, "arousal": 0.8},
            },
            "same_topic_distractor": {
                "q": "Which investor-related memory was specifically about the angel bridge loan — not the final revised-terms signing?",
                "exp": ["bridge_loan_signed"],
                "state": {"valence": 0.5, "arousal": 0.45},
            },
        },
    },
    {
        "id": "s54_diagnosis_journey",
        "desc": "An autoimmune diagnosis brings fear then gradual acceptance and management.",
        "events": [
            {
                "id": "symptom_mystery_anxiety",
                "content": "Joint swelling and fatigue had persisted for three months; each online search added a new terrifying possibility.",
                "v": -0.7,
                "a": 0.8,
                "meta": {"phase": "uncertainty", "actor": "patient"},
            },
            {
                "id": "diagnosis_confirmed",
                "content": "Rheumatoid arthritis was confirmed; oddly, having a name for the enemy brought a strange, hollow relief.",
                "v": -0.4,
                "a": 0.55,
                "meta": {"phase": "diagnosis", "actor": "rheumatologist"},
            },
            {
                "id": "first_biologic_injection",
                "content": "The first self-administered biologic injection took twenty minutes of psyching up and ended without incident.",
                "v": 0.2,
                "a": 0.65,
                "meta": {"phase": "treatment", "actor": "patient"},
            },
            {
                "id": "flare_crisis",
                "content": "A severe flare left the patient unable to type for four days; the career implications felt unbearable.",
                "v": -0.85,
                "a": 0.85,
                "meta": {"phase": "flare", "actor": "patient"},
            },
            {
                "id": "remission_milestone",
                "content": "At the six-month check the rheumatologist used the word 'remission'; the patient photographed the bloodwork results.",
                "v": 0.8,
                "a": 0.65,
                "meta": {"phase": "remission", "actor": "rheumatologist"},
            },
            {
                "id": "running_again",
                "content": "A slow 5k in the park — the first run in eight months — felt like reclaiming an identity.",
                "v": 0.75,
                "a": 0.6,
                "meta": {"phase": "return", "actor": "patient"},
            },
        ],
        "queries": {
            "semantic_confound": {
                "q": "Which autoimmune-related memory was about the terrifying flare that threatened the career — not the remission milestone?",
                "exp": ["flare_crisis"],
                "state": {"valence": -0.8, "arousal": 0.8},
            },
            "affective_arc": {
                "q": "Which memory captures the peak of positive emotion in the diagnosis journey — the rheumatologist's remission verdict?",
                "exp": ["remission_milestone"],
                "state": {"valence": 0.75, "arousal": 0.6},
            },
            "recency_confound": {
                "q": "Which early memory captured the prolonged pre-diagnosis anxiety — before any name was given to the illness?",
                "exp": ["symptom_mystery_anxiety"],
                "state": {"valence": -0.65, "arousal": 0.75},
            },
            "same_topic_distractor": {
                "q": "Which treatment memory was specifically about receiving the diagnosis — not the first injection experience?",
                "exp": ["diagnosis_confirmed"],
                "state": {"valence": -0.35, "arousal": 0.5},
            },
        },
    },
    {
        "id": "s55_book_rejection_arc",
        "desc": "A novelist's manuscript is rejected repeatedly before finding a publisher.",
        "events": [
            {
                "id": "manuscript_submitted",
                "content": "After four years of writing, the manuscript was submitted to twelve agents simultaneously; the wait felt exhilarating.",
                "v": 0.65,
                "a": 0.7,
                "meta": {"phase": "submission", "actor": "novelist"},
            },
            {
                "id": "all_rejected",
                "content": "All twelve agents rejected the manuscript within eight weeks; two called it 'commercially unviable'.",
                "v": -0.85,
                "a": 0.75,
                "meta": {"phase": "rejection", "actor": "agents"},
            },
            {
                "id": "workshop_feedback",
                "content": "A manuscript development workshop identified structural issues in act two that the novelist had sensed but suppressed.",
                "v": -0.2,
                "a": 0.5,
                "meta": {"phase": "revision", "actor": "workshop"},
            },
            {
                "id": "rewrite_dread",
                "content": "The rewrite of act two meant deleting sixty thousand words; the novelist described it as 'literary self-harm'.",
                "v": -0.7,
                "a": 0.65,
                "meta": {"phase": "rewrite", "actor": "novelist"},
            },
            {
                "id": "small_press_offer",
                "content": "A small independent press offered a contract for the revised manuscript; the advance was modest but the letter was warm.",
                "v": 0.75,
                "a": 0.65,
                "meta": {"phase": "acceptance", "actor": "press"},
            },
            {
                "id": "launch_reading",
                "content": "The launch reading drew forty people; a stranger in the front row cried at the passage the novelist had nearly cut.",
                "v": 0.85,
                "a": 0.7,
                "meta": {"phase": "launch", "actor": "audience"},
            },
        ],
        "queries": {
            "semantic_confound": {
                "q": "Which manuscript-related memory was the mass rejection from all twelve agents — not the eventual small-press acceptance?",
                "exp": ["all_rejected"],
                "state": {"valence": -0.8, "arousal": 0.7},
            },
            "affective_arc": {
                "q": "Which memory captures the painful low of deleting sixty thousand words — described as literary self-harm?",
                "exp": ["rewrite_dread"],
                "state": {"valence": -0.65, "arousal": 0.6},
            },
            "recency_confound": {
                "q": "Which early memory captured the hopeful excitement of simultaneous submission — before any rejection arrived?",
                "exp": ["manuscript_submitted"],
                "state": {"valence": 0.6, "arousal": 0.65},
            },
            "same_topic_distractor": {
                "q": "Which post-revision memory was specifically about the publication offer — not the emotional launch reading?",
                "exp": ["small_press_offer"],
                "state": {"valence": 0.7, "arousal": 0.6},
            },
        },
    },
    {
        "id": "s56_parent_dementia_care",
        "desc": "An adult child navigates the emotional complexity of a parent's dementia diagnosis.",
        "events": [
            {
                "id": "dementia_denial_phase",
                "content": "The parent's GP used the word 'dementia' in a routine appointment; the patient laughed and changed the subject.",
                "v": -0.5,
                "a": 0.55,
                "meta": {"phase": "denial", "actor": "parent"},
            },
            {
                "id": "lost_in_supermarket",
                "content": "Security called because the parent had been wandering the supermarket for two hours without buying anything.",
                "v": -0.85,
                "a": 0.85,
                "meta": {"phase": "crisis", "actor": "parent"},
            },
            {
                "id": "care_home_visit",
                "content": "The first care home visit was clinical and exhausting; the brochure language felt dishonest against the smell of the corridors.",
                "v": -0.6,
                "a": 0.7,
                "meta": {"phase": "search", "actor": "caregiver"},
            },
            {
                "id": "piano_recognition",
                "content": "At the care home assessment the parent sat at the piano and played a full Chopin nocturne from memory, smiling.",
                "v": 0.7,
                "a": 0.6,
                "meta": {"phase": "grace", "actor": "parent"},
            },
            {
                "id": "legal_paperwork_burden",
                "content": "Completing the lasting power of attorney while the parent still had legal capacity was emotionally exhausting and bureaucratically endless.",
                "v": -0.65,
                "a": 0.65,
                "meta": {"phase": "admin", "actor": "caregiver"},
            },
            {
                "id": "final_move_in",
                "content": "The move-in day was quiet; the parent asked when they were going home and the caregiver said 'soon'.",
                "v": -0.55,
                "a": 0.5,
                "meta": {"phase": "transition", "actor": "caregiver"},
            },
        ],
        "queries": {
            "semantic_confound": {
                "q": "Which dementia-care memory was the frightening supermarket incident — not the brief musical grace of the Chopin moment?",
                "exp": ["lost_in_supermarket"],
                "state": {"valence": -0.8, "arousal": 0.8},
            },
            "affective_arc": {
                "q": "Which memory captured the unexpected bright moment of the parent's musical memory intact at the care home piano?",
                "exp": ["piano_recognition"],
                "state": {"valence": 0.65, "arousal": 0.55},
            },
            "recency_confound": {
                "q": "Which early memory captured the subtle denial phase — the parent laughing off the GP's dementia word?",
                "exp": ["dementia_denial_phase"],
                "state": {"valence": -0.45, "arousal": 0.5},
            },
            "same_topic_distractor": {
                "q": "Which care-home search memory was specifically the exhausting first visit — not the move-in day?",
                "exp": ["care_home_visit"],
                "state": {"valence": -0.55, "arousal": 0.65},
            },
        },
    },
    {
        "id": "s57_conference_talk_disaster",
        "desc": "A researcher's first major conference talk collapses technically then recovers.",
        "events": [
            {
                "id": "acceptance_notification",
                "content": "The talk acceptance email arrived; the researcher screamed in the library and then apologised to three nearby strangers.",
                "v": 0.9,
                "a": 0.9,
                "meta": {"phase": "acceptance", "actor": "researcher"},
            },
            {
                "id": "projector_failure",
                "content": "The projector died thirty seconds into the talk; the researcher spent six minutes narrating slides from memory while AV staff scrambled.",
                "v": -0.75,
                "a": 0.9,
                "meta": {"phase": "failure", "actor": "av_tech"},
            },
            {
                "id": "audience_applause",
                "content": "The audience applauded through the technical delay; a senior professor called out 'we're learning more this way' and laughter broke the tension.",
                "v": 0.65,
                "a": 0.7,
                "meta": {"phase": "recovery", "actor": "audience"},
            },
            {
                "id": "qa_hostile_question",
                "content": "A hostile question from the floor challenged the methodology as 'fundamentally misspecified'; the researcher's voice cracked.",
                "v": -0.8,
                "a": 0.85,
                "meta": {"phase": "challenge", "actor": "questioner"},
            },
            {
                "id": "co_author_support",
                "content": "The co-author answered the hostile question calmly and completely; the researcher felt a wave of gratitude.",
                "v": 0.7,
                "a": 0.55,
                "meta": {"phase": "support", "actor": "coauthor"},
            },
            {
                "id": "post_talk_feedback",
                "content": "Three attendees asked for collaboration after the session; the AV failure was already becoming a conference legend.",
                "v": 0.75,
                "a": 0.6,
                "meta": {"phase": "aftermath", "actor": "attendees"},
            },
        ],
        "queries": {
            "semantic_confound": {
                "q": "Which conference memory was about the aggressive methodology challenge from the floor — not the warm audience applause during the outage?",
                "exp": ["qa_hostile_question"],
                "state": {"valence": -0.75, "arousal": 0.8},
            },
            "affective_arc": {
                "q": "Which memory represents the highest positive peak after the talk — the three collaboration requests and AV incident becoming legend?",
                "exp": ["post_talk_feedback"],
                "state": {"valence": 0.7, "arousal": 0.55},
            },
            "recency_confound": {
                "q": "Which first-session memory captured the pure joy of the acceptance email — before the talk itself?",
                "exp": ["acceptance_notification"],
                "state": {"valence": 0.85, "arousal": 0.85},
            },
            "same_topic_distractor": {
                "q": "Which talk-day memory was specifically about the co-author's calm response to the hostile question — not the researcher's own cracked voice?",
                "exp": ["co_author_support"],
                "state": {"valence": 0.65, "arousal": 0.5},
            },
        },
    },
    {
        "id": "s58_kitchen_renovation_chaos",
        "desc": "A kitchen renovation descends into disputes before reluctant completion.",
        "events": [
            {
                "id": "design_approval",
                "content": "The kitchen design was finalised after seven revisions; the couple celebrated with takeaway on the floor of the empty room.",
                "v": 0.7,
                "a": 0.6,
                "meta": {"phase": "planning", "actor": "couple"},
            },
            {
                "id": "contractor_no_show",
                "content": "The contractor failed to appear on start day and was unreachable for three days; a deposit had already been paid.",
                "v": -0.85,
                "a": 0.85,
                "meta": {"phase": "breach", "actor": "contractor"},
            },
            {
                "id": "tile_dispute",
                "content": "The wrong tiles arrived; the contractor blamed the supplier and the supplier blamed the contractor's measurements.",
                "v": -0.6,
                "a": 0.7,
                "meta": {"phase": "dispute", "actor": "contractor"},
            },
            {
                "id": "island_reveal",
                "content": "The kitchen island was installed on a Tuesday afternoon; both partners stood silently for a moment before anyone spoke.",
                "v": 0.65,
                "a": 0.5,
                "meta": {"phase": "reveal", "actor": "couple"},
            },
            {
                "id": "final_invoice_shock",
                "content": "The final invoice was forty percent above the quote; the contractor cited 'unforeseen structural issues' with no documentation.",
                "v": -0.8,
                "a": 0.8,
                "meta": {"phase": "billing", "actor": "contractor"},
            },
            {
                "id": "first_dinner_cooked",
                "content": "The first dinner cooked in the finished kitchen was a simple pasta; it tasted extraordinary.",
                "v": 0.8,
                "a": 0.55,
                "meta": {"phase": "completion", "actor": "couple"},
            },
        ],
        "queries": {
            "semantic_confound": {
                "q": "Which renovation memory was about the contractor's unexplained absence and the lost deposit — not the joyful design approval?",
                "exp": ["contractor_no_show"],
                "state": {"valence": -0.8, "arousal": 0.8},
            },
            "affective_arc": {
                "q": "Which memory captures the emotional high of the finished kitchen — the simple pasta that tasted extraordinary?",
                "exp": ["first_dinner_cooked"],
                "state": {"valence": 0.75, "arousal": 0.5},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the couple's initial celebration after finalising the design — before any problems arose?",
                "exp": ["design_approval"],
                "state": {"valence": 0.65, "arousal": 0.55},
            },
            "same_topic_distractor": {
                "q": "Which renovation conflict was specifically about the tile supplier dispute — not the shocking final invoice?",
                "exp": ["tile_dispute"],
                "state": {"valence": -0.55, "arousal": 0.65},
            },
        },
    },
    {
        "id": "s59_academic_tenure_decision",
        "desc": "A junior faculty member's tenure case is denied then overturned on appeal.",
        "events": [
            {
                "id": "dossier_submitted",
                "content": "The tenure dossier was submitted with two hundred pages of supporting materials; the faculty member felt exposed and proud in equal measure.",
                "v": 0.5,
                "a": 0.65,
                "meta": {"phase": "submission", "actor": "faculty"},
            },
            {
                "id": "denial_letter",
                "content": "The tenure denial arrived by email on a Monday morning with no prior warning; the cited reason was 'insufficient national visibility'.",
                "v": -0.9,
                "a": 0.9,
                "meta": {"phase": "denial", "actor": "committee"},
            },
            {
                "id": "union_rep_meeting",
                "content": "The faculty union representative identified three procedural irregularities in the review process and recommended an immediate appeal.",
                "v": 0.3,
                "a": 0.6,
                "meta": {"phase": "appeal", "actor": "union"},
            },
            {
                "id": "external_letters_arrive",
                "content": "Three unsolicited support letters from international scholars arrived at the dean's office during the appeal period.",
                "v": 0.6,
                "a": 0.55,
                "meta": {"phase": "support", "actor": "scholars"},
            },
            {
                "id": "appeal_hearing_dread",
                "content": "The appeal hearing lasted four hours; the faculty member felt like a graduate student defending in front of a hostile room.",
                "v": -0.65,
                "a": 0.8,
                "meta": {"phase": "hearing", "actor": "committee"},
            },
            {
                "id": "tenure_granted",
                "content": "The provost called to say tenure had been granted on appeal; the faculty member had to sit down on the office floor.",
                "v": 0.9,
                "a": 0.85,
                "meta": {"phase": "resolution", "actor": "provost"},
            },
        ],
        "queries": {
            "semantic_confound": {
                "q": "Which tenure memory was the devastating Monday denial email — not the overwhelming relief of the provost's call granting tenure?",
                "exp": ["denial_letter"],
                "state": {"valence": -0.85, "arousal": 0.85},
            },
            "affective_arc": {
                "q": "Which memory captures the ordeal of a four-hour appeal hearing that felt like a hostile PhD defense?",
                "exp": ["appeal_hearing_dread"],
                "state": {"valence": -0.6, "arousal": 0.75},
            },
            "recency_confound": {
                "q": "Which early memory recorded the mixed exposure and pride of submitting the dossier — before any verdict?",
                "exp": ["dossier_submitted"],
                "state": {"valence": 0.45, "arousal": 0.6},
            },
            "same_topic_distractor": {
                "q": "Which appeal-period memory was specifically about the international scholars' unsolicited letters — not the union procedural findings?",
                "exp": ["external_letters_arrive"],
                "state": {"valence": 0.55, "arousal": 0.5},
            },
        },
    },
    {
        "id": "s60_flood_and_rebuild",
        "desc": "A household floods during a storm and rebuilds over months of insurance battles.",
        "events": [
            {
                "id": "flood_discovery",
                "content": "Returning home after a weekend away to find twelve centimetres of water on the ground floor; the smell arrived before the sight.",
                "v": -0.9,
                "a": 0.95,
                "meta": {"phase": "disaster", "actor": "family"},
            },
            {
                "id": "insurance_denial",
                "content": "The insurer denied the initial claim on grounds of 'gradual deterioration'; the family's solicitor called it bad faith.",
                "v": -0.85,
                "a": 0.85,
                "meta": {"phase": "dispute", "actor": "insurer"},
            },
            {
                "id": "temporary_housing_relief",
                "content": "The local council found temporary housing in a dry, clean flat; the children called it 'the holiday flat' for three weeks.",
                "v": 0.4,
                "a": 0.45,
                "meta": {"phase": "interim", "actor": "council"},
            },
            {
                "id": "mould_discovery",
                "content": "Surveyors found black mould behind every internal wall; the remediation cost doubled overnight.",
                "v": -0.75,
                "a": 0.75,
                "meta": {"phase": "complication", "actor": "surveyors"},
            },
            {
                "id": "insurance_partial_payout",
                "content": "Legal pressure resulted in a partial settlement from the insurer; enough to start remediation but not finish.",
                "v": 0.35,
                "a": 0.4,
                "meta": {"phase": "partial_resolution", "actor": "insurer"},
            },
            {
                "id": "first_night_back",
                "content": "The first night back in the renovated house was silent; the family left one lamp on in every room.",
                "v": 0.65,
                "a": 0.4,
                "meta": {"phase": "return", "actor": "family"},
            },
        ],
        "queries": {
            "semantic_confound": {
                "q": "Which flood memory was the insurer's bad-faith denial — not the distressing initial discovery of water on the ground floor?",
                "exp": ["insurance_denial"],
                "state": {"valence": -0.8, "arousal": 0.8},
            },
            "affective_arc": {
                "q": "Which memory captures the mould discovery that doubled remediation costs — the darkest complication of the rebuild?",
                "exp": ["mould_discovery"],
                "state": {"valence": -0.7, "arousal": 0.7},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the raw shock of discovering flood water on returning home — before any insurance contact?",
                "exp": ["flood_discovery"],
                "state": {"valence": -0.85, "arousal": 0.9},
            },
            "same_topic_distractor": {
                "q": "Which legal-outcome memory was specifically about the partial insurance settlement — not the first night back in the renovated house?",
                "exp": ["insurance_partial_payout"],
                "state": {"valence": 0.3, "arousal": 0.38},
            },
        },
    },
    {
        "id": "s61_gallery_opening_snub",
        "desc": "An artist's gallery opening is ignored by critics but embraced by strangers.",
        "events": [
            {
                "id": "installation_complete",
                "content": "The final piece was hung at midnight the day before opening; the artist stood alone in the gallery and felt terrified and certain.",
                "v": 0.5,
                "a": 0.75,
                "meta": {"phase": "completion", "actor": "artist"},
            },
            {
                "id": "critic_no_show",
                "content": "Both invited critics failed to attend the opening; one sent an assistant who stayed eleven minutes.",
                "v": -0.75,
                "a": 0.7,
                "meta": {"phase": "snub", "actor": "critics"},
            },
            {
                "id": "stranger_in_tears",
                "content": "A stranger stood in front of the largest piece for forty minutes and left without speaking; she returned the next day with a friend.",
                "v": 0.8,
                "a": 0.65,
                "meta": {"phase": "connection", "actor": "stranger"},
            },
            {
                "id": "sold_out_week",
                "content": "Every piece sold by the end of the first week; the gallery owner called it the fastest sellout since the space opened.",
                "v": 0.85,
                "a": 0.7,
                "meta": {"phase": "success", "actor": "gallery"},
            },
            {
                "id": "negative_blog_post",
                "content": "A prominent art blogger called the work 'derivative and technically competent but emotionally inert'.",
                "v": -0.8,
                "a": 0.65,
                "meta": {"phase": "criticism", "actor": "blogger"},
            },
            {
                "id": "commission_offer",
                "content": "A museum collections curator offered a commission for a permanent installation after seeing the sold-out show.",
                "v": 0.9,
                "a": 0.75,
                "meta": {"phase": "recognition", "actor": "museum"},
            },
        ],
        "queries": {
            "semantic_confound": {
                "q": "Which gallery memory was the damaging 'emotionally inert' blog review — not the warm stranger who returned a second day?",
                "exp": ["negative_blog_post"],
                "state": {"valence": -0.75, "arousal": 0.6},
            },
            "affective_arc": {
                "q": "Which memory represents the peak of the artist's positive arc — the museum commission offer?",
                "exp": ["commission_offer"],
                "state": {"valence": 0.85, "arousal": 0.7},
            },
            "recency_confound": {
                "q": "Which first-session memory captured the solitary terror and certainty of midnight installation completion?",
                "exp": ["installation_complete"],
                "state": {"valence": 0.45, "arousal": 0.7},
            },
            "same_topic_distractor": {
                "q": "Which audience memory was specifically about the stranger's silent forty-minute vigil — not the sold-out sellout announcement?",
                "exp": ["stranger_in_tears"],
                "state": {"valence": 0.75, "arousal": 0.6},
            },
        },
    },
    {
        "id": "s62_product_recall_crisis",
        "desc": "A consumer goods company issues a product recall amid reputational damage.",
        "events": [
            {
                "id": "safety_report_received",
                "content": "An internal safety report flagged a potential overheating defect in one hundred thousand units already in customers' homes.",
                "v": -0.7,
                "a": 0.8,
                "meta": {"phase": "discovery", "actor": "engineer"},
            },
            {
                "id": "ceo_recall_decision",
                "content": "The CEO announced an immediate voluntary recall; the legal team had advised against it but the CEO overruled them.",
                "v": 0.3,
                "a": 0.65,
                "meta": {"phase": "decision", "actor": "ceo"},
            },
            {
                "id": "social_media_firestorm",
                "content": "A viral post claiming a unit had started a kitchen fire generated forty thousand shares before the company could verify the incident.",
                "v": -0.85,
                "a": 0.9,
                "meta": {"phase": "pr_crisis", "actor": "social_media"},
            },
            {
                "id": "fire_claim_debunked",
                "content": "The fire investigation concluded the kitchen fire predated the product purchase by two months; the viral post was quietly removed.",
                "v": 0.4,
                "a": 0.5,
                "meta": {"phase": "clarification", "actor": "investigators"},
            },
            {
                "id": "recall_logistics_hell",
                "content": "Processing forty thousand return shipments in two weeks stretched the logistics team to breaking point; three team leads requested leave.",
                "v": -0.65,
                "a": 0.75,
                "meta": {"phase": "operations", "actor": "logistics"},
            },
            {
                "id": "consumer_trust_survey",
                "content": "A post-recall brand trust survey showed a net positive outcome; proactive recall was cited as the reason by sixty percent of respondents.",
                "v": 0.65,
                "a": 0.5,
                "meta": {"phase": "outcome", "actor": "customers"},
            },
        ],
        "queries": {
            "semantic_confound": {
                "q": "Which recall memory was the forty-thousand-share viral fire claim — not the eventual brand trust recovery shown by the survey?",
                "exp": ["social_media_firestorm"],
                "state": {"valence": -0.8, "arousal": 0.85},
            },
            "affective_arc": {
                "q": "Which memory captures the positive resolution — the brand trust survey confirming the proactive recall improved customer perception?",
                "exp": ["consumer_trust_survey"],
                "state": {"valence": 0.6, "arousal": 0.45},
            },
            "recency_confound": {
                "q": "Which early memory recorded the initial discovery of the safety defect — before any public announcement or social media reaction?",
                "exp": ["safety_report_received"],
                "state": {"valence": -0.65, "arousal": 0.75},
            },
            "same_topic_distractor": {
                "q": "Which response memory was specifically about the CEO's decision to override legal and announce the recall — not the logistics team crisis?",
                "exp": ["ceo_recall_decision"],
                "state": {"valence": 0.25, "arousal": 0.6},
            },
        },
    },
    {
        "id": "s63_immigration_application",
        "desc": "An immigration application is denied then appealed successfully over eighteen months.",
        "events": [
            {
                "id": "application_submitted",
                "content": "The visa application was submitted with a forty-page supporting document; the applicant felt cautiously hopeful for the first time in a year.",
                "v": 0.5,
                "a": 0.6,
                "meta": {"phase": "submission", "actor": "applicant"},
            },
            {
                "id": "refusal_letter",
                "content": "The refusal letter cited 'insufficient ties to home country'; the applicant had lived abroad for eleven years.",
                "v": -0.9,
                "a": 0.85,
                "meta": {"phase": "refusal", "actor": "authority"},
            },
            {
                "id": "lawyer_consultation",
                "content": "An immigration lawyer identified an error in the adjudicator's reasoning and gave the appeal a sixty percent chance of success.",
                "v": 0.3,
                "a": 0.55,
                "meta": {"phase": "legal", "actor": "lawyer"},
            },
            {
                "id": "appeal_tribunal",
                "content": "The appeal tribunal was a three-hour ordeal in a basement room; the judge asked a clarifying question that felt like a lifeline.",
                "v": -0.3,
                "a": 0.7,
                "meta": {"phase": "tribunal", "actor": "judge"},
            },
            {
                "id": "children_school_disruption",
                "content": "The children had to be withdrawn from school mid-term as a precaution; both cried and asked why they had to leave their friends.",
                "v": -0.85,
                "a": 0.85,
                "meta": {"phase": "family_impact", "actor": "children"},
            },
            {
                "id": "appeal_allowed",
                "content": "The appeal was allowed on all three grounds; the lawyer sent a one-line message: 'Congratulations. It's over.'",
                "v": 0.9,
                "a": 0.8,
                "meta": {"phase": "victory", "actor": "tribunal"},
            },
        ],
        "queries": {
            "semantic_confound": {
                "q": "Which immigration memory was the children's tearful school withdrawal — not the joyful one-line 'it's over' appeal victory?",
                "exp": ["children_school_disruption"],
                "state": {"valence": -0.8, "arousal": 0.8},
            },
            "affective_arc": {
                "q": "Which memory captures the highest positive peak — the lawyer's message that the appeal had been allowed on all grounds?",
                "exp": ["appeal_allowed"],
                "state": {"valence": 0.85, "arousal": 0.75},
            },
            "recency_confound": {
                "q": "Which early memory captured cautious hope on submitting the application — before the refusal?",
                "exp": ["application_submitted"],
                "state": {"valence": 0.45, "arousal": 0.55},
            },
            "same_topic_distractor": {
                "q": "Which legal process memory was specifically about the lawyer's assessment of appeal odds — not the tribunal hearing itself?",
                "exp": ["lawyer_consultation"],
                "state": {"valence": 0.25, "arousal": 0.5},
            },
        },
    },
    {
        "id": "s64_lab_explosion_near_miss",
        "desc": "A near-miss laboratory accident triggers safety review and cultural change.",
        "events": [
            {
                "id": "explosion_near_miss",
                "content": "A solvent flask overheated and shattered; shards hit the fume hood glass and a PhD student sustained a minor cut to the wrist.",
                "v": -0.85,
                "a": 0.95,
                "meta": {"phase": "incident", "actor": "student"},
            },
            {
                "id": "safety_officer_criticism",
                "content": "The university safety officer's report characterised lab culture as 'normalised risk-taking' and recommended suspension of three protocols.",
                "v": -0.7,
                "a": 0.7,
                "meta": {"phase": "investigation", "actor": "safety_officer"},
            },
            {
                "id": "pi_acknowledgement",
                "content": "The principal investigator acknowledged the safety culture failure in an all-hands meeting and apologised without qualification.",
                "v": 0.4,
                "a": 0.55,
                "meta": {"phase": "acknowledgement", "actor": "pi"},
            },
            {
                "id": "student_resignation",
                "content": "The student who was injured submitted a resignation from the PhD programme the following Monday.",
                "v": -0.8,
                "a": 0.75,
                "meta": {"phase": "loss", "actor": "student"},
            },
            {
                "id": "new_protocol_training",
                "content": "The mandatory new safety protocol training was completed by all sixteen lab members within ten days.",
                "v": 0.35,
                "a": 0.4,
                "meta": {"phase": "reform", "actor": "lab_team"},
            },
            {
                "id": "safety_award",
                "content": "Eight months later the lab received the faculty safety innovation award for its protocol overhaul.",
                "v": 0.7,
                "a": 0.55,
                "meta": {"phase": "recognition", "actor": "faculty"},
            },
        ],
        "queries": {
            "semantic_confound": {
                "q": "Which lab memory was the injured student's resignation from the PhD — not the eventual safety innovation award?",
                "exp": ["student_resignation"],
                "state": {"valence": -0.75, "arousal": 0.7},
            },
            "affective_arc": {
                "q": "Which memory captures the shocking near-miss moment — shards hitting the fume hood, the most acute emotional peak of the incident?",
                "exp": ["explosion_near_miss"],
                "state": {"valence": -0.8, "arousal": 0.9},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the actual explosion incident — before the safety investigation began?",
                "exp": ["explosion_near_miss"],
                "state": {"valence": -0.8, "arousal": 0.9},
            },
            "same_topic_distractor": {
                "q": "Which post-incident memory was specifically about the safety officer's critical report — not the PI's public apology?",
                "exp": ["safety_officer_criticism"],
                "state": {"valence": -0.65, "arousal": 0.65},
            },
        },
    },
    {
        "id": "s65_river_crossing_expedition",
        "desc": "A wilderness expedition team nearly loses a member at a flooded river crossing.",
        "events": [
            {
                "id": "expedition_departure",
                "content": "The four-person team left the trailhead in perfect conditions; the forecast showed five clear days and everyone was laughing.",
                "v": 0.8,
                "a": 0.7,
                "meta": {"phase": "departure", "actor": "team"},
            },
            {
                "id": "river_flash_flood",
                "content": "Overnight rain upstream turned the crossing into a metre-deep torrent; one team member was swept off her feet and pulled downstream twenty metres.",
                "v": -0.95,
                "a": 0.99,
                "meta": {"phase": "emergency", "actor": "team_member"},
            },
            {
                "id": "successful_rescue",
                "content": "The rest of the team formed a human chain and pulled her to the bank; she was shaking but uninjured.",
                "v": 0.5,
                "a": 0.8,
                "meta": {"phase": "rescue", "actor": "team"},
            },
            {
                "id": "team_debate_abort",
                "content": "Three hours of debate about whether to abort the expedition; two wanted to continue, two wanted to turn back.",
                "v": -0.4,
                "a": 0.65,
                "meta": {"phase": "decision", "actor": "team"},
            },
            {
                "id": "summit_reached",
                "content": "The team reached the summit on day four; the swept member planted her pole in the snow and said nothing for several minutes.",
                "v": 0.75,
                "a": 0.6,
                "meta": {"phase": "achievement", "actor": "team"},
            },
            {
                "id": "campfire_retelling",
                "content": "Around the final campfire the river story was told three times, each retelling slightly funnier than the last.",
                "v": 0.7,
                "a": 0.6,
                "meta": {"phase": "integration", "actor": "team"},
            },
        ],
        "queries": {
            "semantic_confound": {
                "q": "Which expedition memory was the terrifying river flash flood and member swept downstream — not the triumphant summit arrival?",
                "exp": ["river_flash_flood"],
                "state": {"valence": -0.9, "arousal": 0.95},
            },
            "affective_arc": {
                "q": "Which memory captures the emotional arc peak of achievement — the swept member silently planting her pole at the summit?",
                "exp": ["summit_reached"],
                "state": {"valence": 0.7, "arousal": 0.55},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the carefree departure in perfect conditions — before any danger arose?",
                "exp": ["expedition_departure"],
                "state": {"valence": 0.75, "arousal": 0.65},
            },
            "same_topic_distractor": {
                "q": "Which post-rescue memory was specifically about the team's abort debate — not the cheerful campfire retelling?",
                "exp": ["team_debate_abort"],
                "state": {"valence": -0.35, "arousal": 0.6},
            },
        },
    },
    # ------------------------------------------------------------------
    # COMBO 1: affective_arc + momentum_alignment + recency_confound + same_topic_distractor
    # ------------------------------------------------------------------
    {
        "id": "s66_album_recording",
        "desc": "A musician records their debut album through creative blocks and a final breakthrough.",
        "events": [
            {
                "id": "studio_first_day",
                "content": "The first day in the recording studio was intoxicating — the smell of the mixing desk, the engineer's calm expertise, the sense that something real was beginning.",
                "v": 0.85,
                "a": 0.75,
                "meta": {"phase": "start", "actor": "musician"},
            },
            {
                "id": "vocal_take_collapse",
                "content": "After forty takes of the lead vocal the producer quietly suggested they move to another track; the musician sat in the car park for an hour unable to speak.",
                "v": -0.8,
                "a": 0.8,
                "meta": {"phase": "crisis", "actor": "musician"},
            },
            {
                "id": "band_intervention",
                "content": "The bassist and drummer took the musician out for food and talked for three hours about why the album mattered; no one mentioned the session.",
                "v": 0.4,
                "a": 0.5,
                "meta": {"phase": "support", "actor": "band"},
            },
            {
                "id": "midnight_breakthrough",
                "content": "At midnight on the seventh day the musician sang the entire vocal in one take; the producer said nothing and played it back three times.",
                "v": 0.9,
                "a": 0.85,
                "meta": {"phase": "breakthrough", "actor": "musician"},
            },
            {
                "id": "mix_dispute",
                "content": "A week later a dispute with the label over the final mix threatened to shelve the album indefinitely.",
                "v": -0.65,
                "a": 0.75,
                "meta": {"phase": "conflict", "actor": "label"},
            },
            {
                "id": "album_released",
                "content": "The album was released without compromise and charted in three countries on its first weekend.",
                "v": 0.8,
                "a": 0.7,
                "meta": {"phase": "release", "actor": "musician"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory marks the single highest emotional peak of the album recording — the one-take midnight vocal breakthrough?",
                "exp": ["midnight_breakthrough"],
                "state": {"valence": 0.85, "arousal": 0.8},
            },
            "momentum_alignment": {
                "q": "Which memory captures the sharpest downward momentum shift — sitting speechless in the car park after forty failed vocal takes?",
                "exp": ["vocal_take_collapse"],
                "state": {"valence": -0.75, "arousal": 0.75},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the initial studio excitement — before any creative crisis emerged?",
                "exp": ["studio_first_day"],
                "state": {"valence": 0.8, "arousal": 0.7},
            },
            "same_topic_distractor": {
                "q": "Which post-breakthrough memory was about the label's mix dispute — not the triumphant album release?",
                "exp": ["mix_dispute"],
                "state": {"valence": -0.6, "arousal": 0.7},
            },
        },
    },
    {
        "id": "s67_first_cohabitation_conflict",
        "desc": "A couple's first serious argument after moving in together ends in resolution.",
        "events": [
            {
                "id": "moving_day_joy",
                "content": "Moving day was chaotic and wonderful; they carried boxes in the rain and ordered pizza on the bare floor of their new apartment.",
                "v": 0.8,
                "a": 0.75,
                "meta": {"phase": "start", "actor": "couple"},
            },
            {
                "id": "first_argument_explosion",
                "content": "Three weeks in, a disagreement about chores escalated to screaming; one partner slept on the sofa and both considered whether they had made a mistake.",
                "v": -0.85,
                "a": 0.9,
                "meta": {"phase": "conflict", "actor": "couple"},
            },
            {
                "id": "awkward_morning",
                "content": "The next morning both made coffee without speaking; they passed the milk in silence and someone said sorry first.",
                "v": -0.3,
                "a": 0.5,
                "meta": {"phase": "thaw", "actor": "couple"},
            },
            {
                "id": "chore_agreement",
                "content": "They made a written chore rota together; it felt slightly absurd and slightly necessary.",
                "v": 0.35,
                "a": 0.35,
                "meta": {"phase": "repair", "actor": "couple"},
            },
            {
                "id": "month_later_peace",
                "content": "A month later the rota was still working and neither had brought up the argument again.",
                "v": 0.6,
                "a": 0.4,
                "meta": {"phase": "stability", "actor": "couple"},
            },
            {
                "id": "anniversary_laugh",
                "content": "On their six-month apartment anniversary they laughed about the sofa night over the same takeaway pizza.",
                "v": 0.75,
                "a": 0.6,
                "meta": {"phase": "integration", "actor": "couple"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the emotional nadir of the cohabitation — both partners questioning the decision after the screaming argument?",
                "exp": ["first_argument_explosion"],
                "state": {"valence": -0.8, "arousal": 0.85},
            },
            "momentum_alignment": {
                "q": "Which memory represents the turning point from hostility toward repair — the silent morning and the first sorry?",
                "exp": ["awkward_morning"],
                "state": {"valence": -0.25, "arousal": 0.45},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the joyful rainy moving day — before any conflict emerged?",
                "exp": ["moving_day_joy"],
                "state": {"valence": 0.75, "arousal": 0.7},
            },
            "same_topic_distractor": {
                "q": "Which post-conflict memory was specifically about making the chore rota — not the peaceful month that followed?",
                "exp": ["chore_agreement"],
                "state": {"valence": 0.3, "arousal": 0.3},
            },
        },
    },
    {
        "id": "s68_job_offer_negotiation",
        "desc": "A software engineer receives a dream offer, negotiates anxiously, and accepts.",
        "events": [
            {
                "id": "offer_email_arrives",
                "content": "The offer email arrived during a stand-up meeting; the engineer had to mute themselves to contain the shock.",
                "v": 0.85,
                "a": 0.9,
                "meta": {"phase": "offer", "actor": "engineer"},
            },
            {
                "id": "salary_anxiety",
                "content": "Reading the compensation package that evening, the salary was twenty percent below expectation; a decision had to be made within five days.",
                "v": -0.55,
                "a": 0.75,
                "meta": {"phase": "doubt", "actor": "engineer"},
            },
            {
                "id": "counter_offer_sent",
                "content": "After two days of rehearsing, the engineer sent a counter-offer email with a specific number and a brief rationale; the send button felt physical.",
                "v": -0.3,
                "a": 0.8,
                "meta": {"phase": "negotiation", "actor": "engineer"},
            },
            {
                "id": "silence_dread",
                "content": "Forty-eight hours of silence from the recruiter; the engineer refreshed email every ten minutes and slept poorly.",
                "v": -0.7,
                "a": 0.8,
                "meta": {"phase": "wait", "actor": "engineer"},
            },
            {
                "id": "counter_accepted",
                "content": "The recruiter called to say the counter was accepted in full; the engineer hung up and immediately called their parents.",
                "v": 0.9,
                "a": 0.85,
                "meta": {"phase": "acceptance", "actor": "engineer"},
            },
            {
                "id": "notice_handed_in",
                "content": "Handing in notice at the old job felt surreal — a warm goodbye and a mild guilt about leaving the team mid-sprint.",
                "v": 0.45,
                "a": 0.55,
                "meta": {"phase": "transition", "actor": "engineer"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the peak emotional high — the moment the counter-offer was accepted and the engineer called their parents?",
                "exp": ["counter_accepted"],
                "state": {"valence": 0.85, "arousal": 0.8},
            },
            "momentum_alignment": {
                "q": "Which memory best captures downward momentum of dread — refreshing email for 48 hours of recruiter silence?",
                "exp": ["silence_dread"],
                "state": {"valence": -0.65, "arousal": 0.75},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the initial shock of receiving the job offer during the stand-up — before any salary concern arose?",
                "exp": ["offer_email_arrives"],
                "state": {"valence": 0.8, "arousal": 0.85},
            },
            "same_topic_distractor": {
                "q": "Which post-acceptance memory was about handing in notice — not the moment the counter was accepted?",
                "exp": ["notice_handed_in"],
                "state": {"valence": 0.4, "arousal": 0.5},
            },
        },
    },
    {
        "id": "s69_gallery_rejection_pivot",
        "desc": "An artist is rejected by a prestigious gallery and pivots to a successful group show.",
        "events": [
            {
                "id": "submission_sent",
                "content": "The submission package took three months to prepare; sending it felt like posting a piece of oneself.",
                "v": 0.55,
                "a": 0.65,
                "meta": {"phase": "submission", "actor": "artist"},
            },
            {
                "id": "rejection_letter",
                "content": "The rejection letter arrived: 'does not align with our current curatorial direction'. The artist read it four times and then deleted it.",
                "v": -0.8,
                "a": 0.75,
                "meta": {"phase": "rejection", "actor": "gallery"},
            },
            {
                "id": "studio_dark_weeks",
                "content": "Two weeks of not entering the studio; the canvases faced the wall.",
                "v": -0.7,
                "a": 0.4,
                "meta": {"phase": "withdrawal", "actor": "artist"},
            },
            {
                "id": "group_show_invitation",
                "content": "A collective of five artists invited the artist to co-curate a group show in a converted warehouse space.",
                "v": 0.5,
                "a": 0.6,
                "meta": {"phase": "invitation", "actor": "collective"},
            },
            {
                "id": "opening_night_crowd",
                "content": "The opening night drew three hundred people; two works sold before the first hour ended.",
                "v": 0.85,
                "a": 0.8,
                "meta": {"phase": "success", "actor": "artist"},
            },
            {
                "id": "critic_review",
                "content": "A regional arts critic published a favourable review naming the artist's contribution the conceptual anchor of the show.",
                "v": 0.75,
                "a": 0.65,
                "meta": {"phase": "recognition", "actor": "critic"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the emotional low of withdrawal — weeks of not entering the studio with canvases facing the wall?",
                "exp": ["studio_dark_weeks"],
                "state": {"valence": -0.65, "arousal": 0.35},
            },
            "momentum_alignment": {
                "q": "Which memory marks the upward pivot in momentum — the collective's invitation to co-curate the group show?",
                "exp": ["group_show_invitation"],
                "state": {"valence": 0.45, "arousal": 0.55},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the hopeful act of sending the gallery submission — before the rejection arrived?",
                "exp": ["submission_sent"],
                "state": {"valence": 0.5, "arousal": 0.6},
            },
            "same_topic_distractor": {
                "q": "Which post-opening memory was specifically the critic's review naming the artist's work — not the opening-night crowd and sales?",
                "exp": ["critic_review"],
                "state": {"valence": 0.7, "arousal": 0.6},
            },
        },
    },
    {
        "id": "s70_childs_first_recital",
        "desc": "A parent experiences profound anxiety then pride at their child's first piano recital.",
        "events": [
            {
                "id": "weeks_of_practice",
                "content": "For eight weeks the living room was a practice room; the same twelve bars repeated until they lived inside everyone's head.",
                "v": 0.4,
                "a": 0.45,
                "meta": {"phase": "preparation", "actor": "child"},
            },
            {
                "id": "recital_morning_dread",
                "content": "The morning of the recital the child said nothing at breakfast and the parent felt a fear disproportionate to the stakes.",
                "v": -0.55,
                "a": 0.75,
                "meta": {"phase": "anxiety", "actor": "parent"},
            },
            {
                "id": "stage_walk_up",
                "content": "Watching the child walk to the piano bench in the school hall, the parent gripped the programme so hard it tore.",
                "v": -0.4,
                "a": 0.85,
                "meta": {"phase": "tension", "actor": "parent"},
            },
            {
                "id": "perfect_performance",
                "content": "The child played without a single error; the parent cried before the final chord resolved.",
                "v": 0.9,
                "a": 0.85,
                "meta": {"phase": "performance", "actor": "child"},
            },
            {
                "id": "post_recital_ice_cream",
                "content": "Ice cream afterwards and the child asked, totally seriously, 'Did I do okay?' — as if the applause had not been enough.",
                "v": 0.75,
                "a": 0.65,
                "meta": {"phase": "aftermath", "actor": "child"},
            },
            {
                "id": "teacher_praise",
                "content": "The music teacher pulled the parent aside to say the child had the instinct, not just the technique, and should consider competitions.",
                "v": 0.8,
                "a": 0.6,
                "meta": {"phase": "recognition", "actor": "teacher"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the emotional peak — the parent crying before the final chord of the child's perfect performance?",
                "exp": ["perfect_performance"],
                "state": {"valence": 0.85, "arousal": 0.8},
            },
            "momentum_alignment": {
                "q": "Which memory captures the peak of pre-performance anxiety — the parent gripping the programme until it tore?",
                "exp": ["stage_walk_up"],
                "state": {"valence": -0.35, "arousal": 0.8},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the weeks of household practice — before the recital day anxiety appeared?",
                "exp": ["weeks_of_practice"],
                "state": {"valence": 0.35, "arousal": 0.4},
            },
            "same_topic_distractor": {
                "q": "Which post-performance memory was specifically the music teacher's aside about competitions — not the ice-cream conversation?",
                "exp": ["teacher_praise"],
                "state": {"valence": 0.75, "arousal": 0.55},
            },
        },
    },
    {
        "id": "s71_grant_application_result",
        "desc": "A researcher submits a grant, waits through anxious months, and receives funding.",
        "events": [
            {
                "id": "grant_submitted",
                "content": "The 120-page application was submitted forty minutes before the deadline; the PI and two postdocs sat in silence for several minutes after.",
                "v": 0.5,
                "a": 0.6,
                "meta": {"phase": "submission", "actor": "team"},
            },
            {
                "id": "reviewer_critique_arrives",
                "content": "Preliminary reviewer comments arrived three months later with significant methodological concerns; a rebuttal was required within ten days.",
                "v": -0.7,
                "a": 0.8,
                "meta": {"phase": "critique", "actor": "reviewers"},
            },
            {
                "id": "rebuttal_written",
                "content": "The team worked through a weekend to draft the rebuttal; it was the best scientific writing the PI had produced in five years.",
                "v": 0.3,
                "a": 0.7,
                "meta": {"phase": "response", "actor": "team"},
            },
            {
                "id": "funding_decision_email",
                "content": "The funding decision arrived on a Tuesday morning: full funding, above-threshold score, no conditions.",
                "v": 0.9,
                "a": 0.85,
                "meta": {"phase": "outcome", "actor": "funder"},
            },
            {
                "id": "postdoc_contracts_signed",
                "content": "Both postdocs signed three-year contracts by the end of the week; the lab doubled in size within a month.",
                "v": 0.75,
                "a": 0.65,
                "meta": {"phase": "expansion", "actor": "team"},
            },
            {
                "id": "kickoff_meeting",
                "content": "The funded project kickoff meeting included a catered lunch; someone brought a cake with the grant reference number written in icing.",
                "v": 0.8,
                "a": 0.7,
                "meta": {"phase": "celebration", "actor": "team"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory marks the highest emotional point — the full-funding email arriving on Tuesday morning?",
                "exp": ["funding_decision_email"],
                "state": {"valence": 0.85, "arousal": 0.8},
            },
            "momentum_alignment": {
                "q": "Which memory captures the sharpest negative momentum — the reviewer critique arriving with methodological concerns requiring a rebuttal?",
                "exp": ["reviewer_critique_arrives"],
                "state": {"valence": -0.65, "arousal": 0.75},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the relief of submitting the application — before reviewer concerns arrived?",
                "exp": ["grant_submitted"],
                "state": {"valence": 0.45, "arousal": 0.55},
            },
            "same_topic_distractor": {
                "q": "Which post-funding memory was specifically about the postdoc contracts — not the celebratory kickoff meeting?",
                "exp": ["postdoc_contracts_signed"],
                "state": {"valence": 0.7, "arousal": 0.6},
            },
        },
    },
    {
        "id": "s72_popup_restaurant_disaster",
        "desc": "A chef's pop-up restaurant suffers a catastrophic first night and adapts.",
        "events": [
            {
                "id": "popup_prep_day",
                "content": "The day before opening was fifteen hours of mise en place; the kitchen smelled of brown butter and anticipation.",
                "v": 0.75,
                "a": 0.7,
                "meta": {"phase": "preparation", "actor": "chef"},
            },
            {
                "id": "service_collapse",
                "content": "On the first night the main supplier failed to deliver; twenty-two diners waited forty minutes for their first course and four tables walked out.",
                "v": -0.9,
                "a": 0.95,
                "meta": {"phase": "disaster", "actor": "supplier"},
            },
            {
                "id": "improvised_menu",
                "content": "The chef pivoted to a three-course tasting menu from pantry stock; the remaining guests were told it was an 'unannounced format change'.",
                "v": 0.2,
                "a": 0.7,
                "meta": {"phase": "adaptation", "actor": "chef"},
            },
            {
                "id": "guest_standing_ovation",
                "content": "The remaining eighteen diners gave a standing ovation; two asked for the recipe for the improvised pasta dish.",
                "v": 0.8,
                "a": 0.75,
                "meta": {"phase": "redemption", "actor": "guests"},
            },
            {
                "id": "supplier_dispute",
                "content": "The next morning the chef sent a formal dispute letter to the supplier and began sourcing alternatives.",
                "v": -0.4,
                "a": 0.6,
                "meta": {"phase": "resolution", "actor": "chef"},
            },
            {
                "id": "review_published",
                "content": "A food blogger who had been at the opening published a review calling the improvisation 'a masterclass in composure'.",
                "v": 0.75,
                "a": 0.6,
                "meta": {"phase": "recognition", "actor": "blogger"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the emotional nadir — the service collapse with tables walking out after forty-minute waits?",
                "exp": ["service_collapse"],
                "state": {"valence": -0.85, "arousal": 0.9},
            },
            "momentum_alignment": {
                "q": "Which memory marks the upward momentum reversal — the standing ovation from the remaining guests?",
                "exp": ["guest_standing_ovation"],
                "state": {"valence": 0.75, "arousal": 0.7},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the anticipatory prep day — before the opening night disaster?",
                "exp": ["popup_prep_day"],
                "state": {"valence": 0.7, "arousal": 0.65},
            },
            "same_topic_distractor": {
                "q": "Which post-redemption memory was specifically about the supplier dispute letter — not the food blog review?",
                "exp": ["supplier_dispute"],
                "state": {"valence": -0.35, "arousal": 0.55},
            },
        },
    },
    {
        "id": "s73_coming_out_to_parents",
        "desc": "A young person comes out to their parents through fear to acceptance.",
        "events": [
            {
                "id": "decision_made",
                "content": "After a year of rehearsal the decision was made: this weekend, at the kitchen table, no preamble.",
                "v": 0.3,
                "a": 0.8,
                "meta": {"phase": "decision", "actor": "person"},
            },
            {
                "id": "the_conversation",
                "content": "The words came out in the wrong order but they came out; the silence that followed lasted approximately eight seconds and felt like eight years.",
                "v": -0.5,
                "a": 0.95,
                "meta": {"phase": "disclosure", "actor": "person"},
            },
            {
                "id": "father_leaves_room",
                "content": "The father stood and left the kitchen without speaking; the mother reached across the table and held both hands.",
                "v": -0.6,
                "a": 0.8,
                "meta": {"phase": "split_response", "actor": "parents"},
            },
            {
                "id": "father_returns",
                "content": "Twenty minutes later the father returned and said, 'I needed a moment. You're my child. That doesn't change anything.'",
                "v": 0.7,
                "a": 0.75,
                "meta": {"phase": "acceptance", "actor": "father"},
            },
            {
                "id": "family_dinner_normal",
                "content": "The Sunday dinner that followed was almost entirely about the football results, which felt like the most generous gift imaginable.",
                "v": 0.8,
                "a": 0.55,
                "meta": {"phase": "normalisation", "actor": "family"},
            },
            {
                "id": "months_later_pride",
                "content": "Six months later both parents attended a Pride event; the father wore a small rainbow badge and didn't mention it.",
                "v": 0.9,
                "a": 0.7,
                "meta": {"phase": "affirmation", "actor": "parents"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the most acute emotional peak — the eight-second silence after the words came out?",
                "exp": ["the_conversation"],
                "state": {"valence": -0.45, "arousal": 0.9},
            },
            "momentum_alignment": {
                "q": "Which memory marks the decisive upward turn — the father returning to say 'that doesn't change anything'?",
                "exp": ["father_returns"],
                "state": {"valence": 0.65, "arousal": 0.7},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the internal decision moment — before the actual kitchen table conversation?",
                "exp": ["decision_made"],
                "state": {"valence": 0.25, "arousal": 0.75},
            },
            "same_topic_distractor": {
                "q": "Which post-acceptance memory was specifically the near-normal Sunday dinner — not the Pride event six months later?",
                "exp": ["family_dinner_normal"],
                "state": {"valence": 0.75, "arousal": 0.5},
            },
        },
    },
    {
        "id": "s74_difficult_prognosis",
        "desc": "A doctor delivers a serious prognosis, provides compassionate support, and sees a relief-filled follow-up.",
        "events": [
            {
                "id": "scan_results_reviewed",
                "content": "Reviewing the scans before the consultation, the oncologist confirmed the diagnosis and spent ten minutes preparing exactly how to say it.",
                "v": -0.5,
                "a": 0.6,
                "meta": {"phase": "preparation", "actor": "doctor"},
            },
            {
                "id": "prognosis_delivered",
                "content": "The patient and their spouse sat across the desk; the doctor said the word and the room changed.",
                "v": -0.85,
                "a": 0.85,
                "meta": {"phase": "disclosure", "actor": "doctor"},
            },
            {
                "id": "patient_silence",
                "content": "The patient said nothing for three minutes; the doctor stayed present, did not fill the silence, and handed tissues.",
                "v": -0.7,
                "a": 0.6,
                "meta": {"phase": "response", "actor": "patient"},
            },
            {
                "id": "treatment_plan_outlined",
                "content": "The doctor outlined a treatment plan with clear milestones; the structure visibly steadied the patient's breathing.",
                "v": 0.3,
                "a": 0.5,
                "meta": {"phase": "planning", "actor": "doctor"},
            },
            {
                "id": "three_month_scan_clear",
                "content": "The three-month follow-up scan showed no new growth; the doctor allowed themselves a private exhale before entering the room.",
                "v": 0.75,
                "a": 0.65,
                "meta": {"phase": "relief", "actor": "doctor"},
            },
            {
                "id": "patient_letter_received",
                "content": "A handwritten letter arrived from the patient a week after the clear scan, thanking the doctor for not filling the silence.",
                "v": 0.85,
                "a": 0.55,
                "meta": {"phase": "gratitude", "actor": "patient"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the emotional nadir of the clinical encounter — the moment the word was said and the room changed?",
                "exp": ["prognosis_delivered"],
                "state": {"valence": -0.8, "arousal": 0.8},
            },
            "momentum_alignment": {
                "q": "Which memory marks the first upward momentum shift — the treatment plan visibly steadying the patient's breathing?",
                "exp": ["treatment_plan_outlined"],
                "state": {"valence": 0.25, "arousal": 0.45},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the doctor's private preparation before the consultation — before the patient entered?",
                "exp": ["scan_results_reviewed"],
                "state": {"valence": -0.45, "arousal": 0.55},
            },
            "same_topic_distractor": {
                "q": "Which post-follow-up memory was specifically the patient's handwritten letter — not the doctor's private exhale at the clear scan result?",
                "exp": ["patient_letter_received"],
                "state": {"valence": 0.8, "arousal": 0.5},
            },
        },
    },
    {
        "id": "s75_miscarriage_and_healing",
        "desc": "A couple processes a miscarriage through grief toward supported recovery.",
        "events": [
            {
                "id": "pregnancy_announced",
                "content": "The pregnancy test had been positive; they told only their parents and spent an evening looking at baby names they both knew were too early.",
                "v": 0.8,
                "a": 0.7,
                "meta": {"phase": "hope", "actor": "couple"},
            },
            {
                "id": "miscarriage_confirmed",
                "content": "The twelve-week scan showed no heartbeat; the sonographer left the room and one of them said the other's name, just that.",
                "v": -0.95,
                "a": 0.9,
                "meta": {"phase": "loss", "actor": "couple"},
            },
            {
                "id": "return_home_silence",
                "content": "They drove home and did not turn the radio on; neither could say what they wanted so they sat in the parked car for a long time.",
                "v": -0.85,
                "a": 0.6,
                "meta": {"phase": "shock", "actor": "couple"},
            },
            {
                "id": "telling_parents",
                "content": "Telling their parents was a different kind of pain — watching others try to find words for something that had no adequate words.",
                "v": -0.7,
                "a": 0.65,
                "meta": {"phase": "disclosure", "actor": "couple"},
            },
            {
                "id": "counsellor_first_session",
                "content": "The counsellor said, 'You don't have to make sense of it yet.' That sentence was useful for weeks.",
                "v": 0.35,
                "a": 0.4,
                "meta": {"phase": "support", "actor": "counsellor"},
            },
            {
                "id": "walking_together",
                "content": "Two months later they walked the same coastal path they had planned to do with the baby; it was a hard walk and the right one.",
                "v": 0.5,
                "a": 0.55,
                "meta": {"phase": "integration", "actor": "couple"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory marks the deepest emotional low — the sonographer leaving the room and one partner saying only the other's name?",
                "exp": ["miscarriage_confirmed"],
                "state": {"valence": -0.9, "arousal": 0.85},
            },
            "momentum_alignment": {
                "q": "Which memory marks the first upward shift toward healing — the counsellor's sentence that was useful for weeks?",
                "exp": ["counsellor_first_session"],
                "state": {"valence": 0.3, "arousal": 0.35},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the hopeful pregnancy announcement evening — before any loss occurred?",
                "exp": ["pregnancy_announced"],
                "state": {"valence": 0.75, "arousal": 0.65},
            },
            "same_topic_distractor": {
                "q": "Which grief-phase memory was specifically about telling their parents — not sitting in the silent car after the hospital?",
                "exp": ["telling_parents"],
                "state": {"valence": -0.65, "arousal": 0.6},
            },
        },
    },
    {
        "id": "s76_product_launch_crash",
        "desc": "An entrepreneur's product launch crashes due to a backend failure, then recovers.",
        "events": [
            {
                "id": "launch_day_morning",
                "content": "Launch day began with a live stream, a countdown timer, and four thousand people in a virtual waiting room.",
                "v": 0.85,
                "a": 0.9,
                "meta": {"phase": "launch", "actor": "team"},
            },
            {
                "id": "server_failure",
                "content": "Ninety seconds after the doors opened the payment server failed under load; the waiting room emptied as error messages propagated.",
                "v": -0.9,
                "a": 0.95,
                "meta": {"phase": "failure", "actor": "infrastructure"},
            },
            {
                "id": "public_apology_tweet",
                "content": "The founder posted a public apology within fifteen minutes, acknowledged the failure, and promised a 48-hour extended launch window.",
                "v": -0.3,
                "a": 0.75,
                "meta": {"phase": "response", "actor": "founder"},
            },
            {
                "id": "team_all_nighter",
                "content": "The engineering team worked through the night; the fix was deployed by 4 a.m. and tested against simulated peak load.",
                "v": 0.2,
                "a": 0.8,
                "meta": {"phase": "repair", "actor": "team"},
            },
            {
                "id": "second_launch_success",
                "content": "The extended window opened to a wave of purchases; many customers cited the transparent apology as the reason they returned.",
                "v": 0.85,
                "a": 0.8,
                "meta": {"phase": "recovery", "actor": "customers"},
            },
            {
                "id": "press_write_up",
                "content": "A tech journalist wrote about the launch-to-recovery arc as a case study in crisis communication; the piece was shared widely.",
                "v": 0.75,
                "a": 0.65,
                "meta": {"phase": "recognition", "actor": "press"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the acute emotional crisis — the server failure as four thousand waiting customers saw error messages?",
                "exp": ["server_failure"],
                "state": {"valence": -0.85, "arousal": 0.9},
            },
            "momentum_alignment": {
                "q": "Which memory marks the turning point from crisis to recovery — the extended window opening to a wave of returning customers?",
                "exp": ["second_launch_success"],
                "state": {"valence": 0.8, "arousal": 0.75},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the electric launch morning — before the server failure occurred?",
                "exp": ["launch_day_morning"],
                "state": {"valence": 0.8, "arousal": 0.85},
            },
            "same_topic_distractor": {
                "q": "Which post-recovery memory was specifically the all-night engineering repair — not the press write-up about crisis communication?",
                "exp": ["team_all_nighter"],
                "state": {"valence": 0.15, "arousal": 0.75},
            },
        },
    },
    {
        "id": "s77_summit_bid_and_retry",
        "desc": "A climber's first summit bid fails at altitude and they succeed on a second attempt.",
        "events": [
            {
                "id": "base_camp_arrival",
                "content": "Arriving at base camp after four days' approach, the mountain looked both achievable and absurd in equal measure.",
                "v": 0.65,
                "a": 0.7,
                "meta": {"phase": "arrival", "actor": "climber"},
            },
            {
                "id": "summit_attempt_abort",
                "content": "At 7200m a sudden weather window closed; the team turned back sixty metres below the summit ridge and said nothing to each other for an hour.",
                "v": -0.8,
                "a": 0.8,
                "meta": {"phase": "retreat", "actor": "team"},
            },
            {
                "id": "base_camp_grief",
                "content": "Back at base camp the climber sat alone and reviewed the altitude profile, unable to decide whether to stay for a second window.",
                "v": -0.6,
                "a": 0.55,
                "meta": {"phase": "decision", "actor": "climber"},
            },
            {
                "id": "second_bid_commitment",
                "content": "After two rest days the team committed to a second attempt; the energy was quieter and more determined than the first.",
                "v": 0.5,
                "a": 0.65,
                "meta": {"phase": "commitment", "actor": "team"},
            },
            {
                "id": "summit_reached",
                "content": "The summit was reached in full visibility; the climber took no photographs for the first three minutes.",
                "v": 0.9,
                "a": 0.8,
                "meta": {"phase": "success", "actor": "climber"},
            },
            {
                "id": "descent_camp_celebration",
                "content": "Back at high camp the team shared a thermos and the expedition cook had kept dinner warm; no one mentioned the first attempt.",
                "v": 0.75,
                "a": 0.6,
                "meta": {"phase": "celebration", "actor": "team"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the summit achievement — reaching the top in full visibility without photographing it for three minutes?",
                "exp": ["summit_reached"],
                "state": {"valence": 0.85, "arousal": 0.75},
            },
            "momentum_alignment": {
                "q": "Which memory captures the sharpest downward momentum — the team turning back sixty metres below the summit in silence?",
                "exp": ["summit_attempt_abort"],
                "state": {"valence": -0.75, "arousal": 0.75},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the base camp arrival — before the summit attempt and retreat?",
                "exp": ["base_camp_arrival"],
                "state": {"valence": 0.6, "arousal": 0.65},
            },
            "same_topic_distractor": {
                "q": "Which post-retreat memory was specifically about committing to a second attempt — not the quiet celebration at high camp after success?",
                "exp": ["second_bid_commitment"],
                "state": {"valence": 0.45, "arousal": 0.6},
            },
        },
    },
    {
        "id": "s78_agent_rejection_to_acceptance",
        "desc": "A writer receives agent rejections, revises, and places a story in a literary magazine.",
        "events": [
            {
                "id": "manuscript_queried",
                "content": "The query letter took six weeks to refine; sending it to twelve agents felt like posting twelve versions of a prayer.",
                "v": 0.5,
                "a": 0.65,
                "meta": {"phase": "submission", "actor": "writer"},
            },
            {
                "id": "rejection_streak",
                "content": "Nine consecutive form rejections arrived over six weeks; the writer kept a tally in the margin of a notebook and ran out of space.",
                "v": -0.75,
                "a": 0.65,
                "meta": {"phase": "rejection", "actor": "agents"},
            },
            {
                "id": "workshop_feedback",
                "content": "A workshop peer said the opening chapter was 'performing emotion rather than living it'; the critique stung and then illuminated.",
                "v": -0.4,
                "a": 0.6,
                "meta": {"phase": "critique", "actor": "peer"},
            },
            {
                "id": "revision_completed",
                "content": "A twelve-week revision stripped 8,000 words and rebuilt the opening from scratch; it was a better book.",
                "v": 0.55,
                "a": 0.6,
                "meta": {"phase": "revision", "actor": "writer"},
            },
            {
                "id": "magazine_acceptance",
                "content": "A literary magazine accepted the first chapter as a standalone story; the acceptance email used the word 'luminous'.",
                "v": 0.85,
                "a": 0.75,
                "meta": {"phase": "acceptance", "actor": "magazine"},
            },
            {
                "id": "story_published",
                "content": "The story went live on a Tuesday; three agents who had previously rejected the manuscript emailed requesting the full novel.",
                "v": 0.9,
                "a": 0.8,
                "meta": {"phase": "publication", "actor": "writer"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the highest emotional peak — the acceptance email using the word 'luminous'?",
                "exp": ["magazine_acceptance"],
                "state": {"valence": 0.8, "arousal": 0.7},
            },
            "momentum_alignment": {
                "q": "Which memory marks the momentum shift from stagnation to renewed direction — the workshop peer critique that stung and then illuminated?",
                "exp": ["workshop_feedback"],
                "state": {"valence": -0.35, "arousal": 0.55},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the hopeful act of querying agents — before the rejection streak?",
                "exp": ["manuscript_queried"],
                "state": {"valence": 0.45, "arousal": 0.6},
            },
            "same_topic_distractor": {
                "q": "Which post-acceptance memory was specifically the story going live and agent requests arriving — not the magazine acceptance email itself?",
                "exp": ["story_published"],
                "state": {"valence": 0.85, "arousal": 0.75},
            },
        },
    },
    {
        "id": "s79_first_night_alone_with_baby",
        "desc": "A new parent's first night alone with an infant swings from panic to confidence.",
        "events": [
            {
                "id": "partner_leaves",
                "content": "The partner left for a work trip at 7 a.m.; the baby, twenty-three days old, regarded the new arrangement without apparent concern.",
                "v": -0.3,
                "a": 0.65,
                "meta": {"phase": "start", "actor": "parent"},
            },
            {
                "id": "three_hour_crying",
                "content": "From 11 p.m. the baby cried for three hours without stopping; the parent tried every known method and none of them worked.",
                "v": -0.85,
                "a": 0.9,
                "meta": {"phase": "crisis", "actor": "baby"},
            },
            {
                "id": "walking_and_humming",
                "content": "At 2 a.m., out of ideas, the parent walked slow loops of the kitchen humming a half-remembered song; the crying tapered into hiccups and then stopped.",
                "v": 0.2,
                "a": 0.6,
                "meta": {"phase": "resolution", "actor": "parent"},
            },
            {
                "id": "sunrise_feeding",
                "content": "The 5 a.m. feed was calm; pale light came through the blind and the parent thought: we did that.",
                "v": 0.7,
                "a": 0.5,
                "meta": {"phase": "competence", "actor": "parent"},
            },
            {
                "id": "partner_return_call",
                "content": "The partner called from the hotel and asked how it went; the parent said 'fine' without embellishment and mostly meant it.",
                "v": 0.6,
                "a": 0.45,
                "meta": {"phase": "report", "actor": "parent"},
            },
            {
                "id": "solo_confidence",
                "content": "The following week, when the partner suggested calling the grandparents for help, the parent said they didn't need to.",
                "v": 0.75,
                "a": 0.55,
                "meta": {"phase": "confidence", "actor": "parent"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the emotional nadir — three hours of unresolvable crying at 11 p.m. alone with the infant?",
                "exp": ["three_hour_crying"],
                "state": {"valence": -0.8, "arousal": 0.85},
            },
            "momentum_alignment": {
                "q": "Which memory marks the upward momentum pivot — the kitchen humming that resolved the crying into hiccups?",
                "exp": ["walking_and_humming"],
                "state": {"valence": 0.15, "arousal": 0.55},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the partner leaving for the work trip — before the night crisis began?",
                "exp": ["partner_leaves"],
                "state": {"valence": -0.25, "arousal": 0.6},
            },
            "same_topic_distractor": {
                "q": "Which post-crisis memory was specifically about the calm 5 a.m. feed at sunrise — not the new confidence shown the following week?",
                "exp": ["sunrise_feeding"],
                "state": {"valence": 0.65, "arousal": 0.45},
            },
        },
    },
    {
        "id": "s80_betrayal_confrontation",
        "desc": "A close friendship is fractured by a betrayal, confronted, and partially rebuilt.",
        "events": [
            {
                "id": "secret_revealed",
                "content": "A mutual friend mentioned in passing that something private had been shared without permission; the realisation took a moment to arrive and then arrived fully.",
                "v": -0.8,
                "a": 0.85,
                "meta": {"phase": "discovery", "actor": "betrayed"},
            },
            {
                "id": "confrontation_meeting",
                "content": "The confrontation happened over coffee; the betrayer was sorry in a way that felt both genuine and insufficient.",
                "v": -0.6,
                "a": 0.75,
                "meta": {"phase": "confrontation", "actor": "friends"},
            },
            {
                "id": "two_weeks_silence",
                "content": "There was no contact for two weeks; messages were drafted and deleted from both sides.",
                "v": -0.65,
                "a": 0.55,
                "meta": {"phase": "distance", "actor": "friends"},
            },
            {
                "id": "first_contact_resumed",
                "content": "A brief, functional text about a shared obligation; both responded within minutes, which said something.",
                "v": 0.1,
                "a": 0.45,
                "meta": {"phase": "thaw", "actor": "friends"},
            },
            {
                "id": "honest_conversation",
                "content": "A longer conversation a month later where the pattern behind the betrayal was named; understanding it did not excuse it but it helped.",
                "v": 0.4,
                "a": 0.55,
                "meta": {"phase": "understanding", "actor": "friends"},
            },
            {
                "id": "friendship_resumed_partial",
                "content": "The friendship resumed, altered; the trust did not fully return but something real remained.",
                "v": 0.5,
                "a": 0.4,
                "meta": {"phase": "partial_resolution", "actor": "friends"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the sharpest emotional impact — the realisation that a private confidence had been shared without permission?",
                "exp": ["secret_revealed"],
                "state": {"valence": -0.75, "arousal": 0.8},
            },
            "momentum_alignment": {
                "q": "Which memory marks the first upward momentum shift after the silence — the brief functional text where both replied within minutes?",
                "exp": ["first_contact_resumed"],
                "state": {"valence": 0.05, "arousal": 0.4},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the discovery moment — before the confrontation meeting took place?",
                "exp": ["secret_revealed"],
                "state": {"valence": -0.75, "arousal": 0.8},
            },
            "same_topic_distractor": {
                "q": "Which post-confrontation memory was specifically about the longer conversation naming the pattern — not the partial friendship resumption?",
                "exp": ["honest_conversation"],
                "state": {"valence": 0.35, "arousal": 0.5},
            },
        },
    },
    # ------------------------------------------------------------------
    # COMBO 2: momentum_alignment + recency_confound + same_topic_distractor + semantic_confound
    # ------------------------------------------------------------------
    {
        "id": "s81_nurse_final_shift",
        "desc": "A retiring nurse's final shift oscillates between exhaustion and meaning.",
        "events": [
            {
                "id": "final_shift_starts",
                "content": "The night before the last shift the nurse laid out the uniform one more time; thirty-one years of this gesture.",
                "v": 0.55,
                "a": 0.6,
                "meta": {"phase": "anticipation", "actor": "nurse"},
            },
            {
                "id": "difficult_patient_overnight",
                "content": "An overnight admission required continuous monitoring; by 4 a.m. the nurse had not sat down.",
                "v": -0.5,
                "a": 0.7,
                "meta": {"phase": "strain", "actor": "nurse"},
            },
            {
                "id": "colleague_handover_emotion",
                "content": "At handover a colleague of twenty years gripped the nurse's arm and said nothing; neither looked at the other directly.",
                "v": 0.6,
                "a": 0.75,
                "meta": {"phase": "farewell", "actor": "colleague"},
            },
            {
                "id": "retirement_party_ward",
                "content": "The ward party was held in the break room; there was a cake and a card signed by four cohorts of junior nurses.",
                "v": 0.8,
                "a": 0.65,
                "meta": {"phase": "celebration", "actor": "ward"},
            },
            {
                "id": "walking_out_corridor",
                "content": "Walking out of the hospital for the last time the nurse paused at the corridor junction where the wards split, just once.",
                "v": 0.5,
                "a": 0.55,
                "meta": {"phase": "departure", "actor": "nurse"},
            },
            {
                "id": "first_morning_no_alarm",
                "content": "The first morning without an alarm: awake at 5:47 a.m. anyway, staring at the ceiling.",
                "v": 0.3,
                "a": 0.45,
                "meta": {"phase": "aftermath", "actor": "nurse"},
            },
        ],
        "queries": {
            "momentum_alignment": {
                "q": "Which memory captures the emotional momentum peak of recognition — the twenty-year colleague gripping the nurse's arm without words?",
                "exp": ["colleague_handover_emotion"],
                "state": {"valence": 0.55, "arousal": 0.7},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the night-before ritual of laying out the uniform — before the final shift began?",
                "exp": ["final_shift_starts"],
                "state": {"valence": 0.5, "arousal": 0.55},
            },
            "same_topic_distractor": {
                "q": "Which hospital-setting memory was specifically about the ward break-room party — not the silent corridor pause at departure?",
                "exp": ["retirement_party_ward"],
                "state": {"valence": 0.75, "arousal": 0.6},
            },
            "semantic_confound": {
                "q": "Which memory captures the straining exhaustion of the final overnight — not the emotional recognition at handover?",
                "exp": ["difficult_patient_overnight"],
                "state": {"valence": -0.45, "arousal": 0.65},
            },
        },
    },
    {
        "id": "s82_sibling_reunion_at_funeral",
        "desc": "Estranged siblings reconnect at a parent's funeral after years of silence.",
        "events": [
            {
                "id": "estrangement_context",
                "content": "The siblings had not spoken for four years; the reason had calcified into something neither could fully describe.",
                "v": -0.6,
                "a": 0.4,
                "meta": {"phase": "background", "actor": "siblings"},
            },
            {
                "id": "phone_call_about_death",
                "content": "The call came from a cousin: their mother had died in the early hours. The estranged sibling heard the news in the background.",
                "v": -0.9,
                "a": 0.9,
                "meta": {"phase": "loss", "actor": "family"},
            },
            {
                "id": "funeral_arrival_eye_contact",
                "content": "At the funeral they arrived from opposite sides of the car park and made eye contact across twenty metres of gravel.",
                "v": -0.5,
                "a": 0.8,
                "meta": {"phase": "encounter", "actor": "siblings"},
            },
            {
                "id": "graveside_standing_together",
                "content": "They stood on the same side of the grave without pre-arrangement; no words, shoulder almost touching.",
                "v": -0.3,
                "a": 0.65,
                "meta": {"phase": "proximity", "actor": "siblings"},
            },
            {
                "id": "wake_conversation",
                "content": "In the kitchen during the wake they talked for an hour about their mother, then peripherally about themselves. The four-year gap was not mentioned.",
                "v": 0.4,
                "a": 0.55,
                "meta": {"phase": "reconnection", "actor": "siblings"},
            },
            {
                "id": "message_days_later",
                "content": "Four days after the funeral a message arrived: 'I don't want this to be the last time.'",
                "v": 0.65,
                "a": 0.6,
                "meta": {"phase": "invitation", "actor": "sibling"},
            },
        ],
        "queries": {
            "momentum_alignment": {
                "q": "Which memory marks the turning point in emotional momentum — the kitchen conversation where the four-year gap was not mentioned?",
                "exp": ["wake_conversation"],
                "state": {"valence": 0.35, "arousal": 0.5},
            },
            "recency_confound": {
                "q": "Which first-session memory established the background of estrangement — before the death occurred?",
                "exp": ["estrangement_context"],
                "state": {"valence": -0.55, "arousal": 0.35},
            },
            "same_topic_distractor": {
                "q": "Which funeral-setting memory was specifically about standing together at the graveside — not the car-park eye contact?",
                "exp": ["graveside_standing_together"],
                "state": {"valence": -0.25, "arousal": 0.6},
            },
            "semantic_confound": {
                "q": "Which memory captures the acute grief of the death news — not the emotionally charged funeral car-park encounter?",
                "exp": ["phone_call_about_death"],
                "state": {"valence": -0.85, "arousal": 0.85},
            },
        },
    },
    {
        "id": "s83_reorg_announcement",
        "desc": "A tech lead navigates a corporate reorganisation announcement and its team fallout.",
        "events": [
            {
                "id": "reorg_rumours",
                "content": "Rumours of a reorganisation had circulated for three weeks; the tech lead deflected questions without lying, which required careful language.",
                "v": -0.4,
                "a": 0.6,
                "meta": {"phase": "rumour", "actor": "tech_lead"},
            },
            {
                "id": "announcement_meeting",
                "content": "The all-hands meeting confirmed the restructure; two product lines were merged and reporting lines changed overnight.",
                "v": -0.55,
                "a": 0.75,
                "meta": {"phase": "announcement", "actor": "company"},
            },
            {
                "id": "team_reaction_anger",
                "content": "The team's immediate reaction was anger at the process, not the outcome; three people said they were updating their CVs.",
                "v": -0.75,
                "a": 0.8,
                "meta": {"phase": "reaction", "actor": "team"},
            },
            {
                "id": "one_to_ones_held",
                "content": "The tech lead held one-to-ones with each of the seven team members over two days; the anger began to differentiate into individual concerns.",
                "v": 0.2,
                "a": 0.55,
                "meta": {"phase": "response", "actor": "tech_lead"},
            },
            {
                "id": "two_resignations",
                "content": "Two team members resigned within the month; the tech lead wrote genuine reference letters and felt the loss acutely.",
                "v": -0.6,
                "a": 0.6,
                "meta": {"phase": "attrition", "actor": "team"},
            },
            {
                "id": "team_stabilisation",
                "content": "Six weeks later the remaining team had found a rhythm in the new structure; the merger produced one unexpected synergy that nobody had predicted.",
                "v": 0.5,
                "a": 0.5,
                "meta": {"phase": "stabilisation", "actor": "team"},
            },
        ],
        "queries": {
            "momentum_alignment": {
                "q": "Which memory marks the downward momentum peak — the team's immediate anger with three people openly updating their CVs?",
                "exp": ["team_reaction_anger"],
                "state": {"valence": -0.7, "arousal": 0.75},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the pre-announcement rumour period — before the all-hands confirmed the restructure?",
                "exp": ["reorg_rumours"],
                "state": {"valence": -0.35, "arousal": 0.55},
            },
            "same_topic_distractor": {
                "q": "Which post-announcement memory was specifically about the two resignations — not the one-to-ones that differentiated the anger?",
                "exp": ["two_resignations"],
                "state": {"valence": -0.55, "arousal": 0.55},
            },
            "semantic_confound": {
                "q": "Which memory captures organisational-change impact via grief of losing people — not the stabilisation six weeks later?",
                "exp": ["two_resignations"],
                "state": {"valence": -0.55, "arousal": 0.55},
            },
        },
    },
    {
        "id": "s84_singer_loses_voice",
        "desc": "A singer loses their voice before a major concert and makes a comeback.",
        "events": [
            {
                "id": "voice_disappears",
                "content": "Four days before the sold-out concert the voice went: not hoarse, just absent. The ENT confirmed acute laryngitis.",
                "v": -0.85,
                "a": 0.9,
                "meta": {"phase": "crisis", "actor": "singer"},
            },
            {
                "id": "cancellation_considered",
                "content": "The management presented two options: cancel or reschedule. The singer sat with the options for six hours and called no one.",
                "v": -0.7,
                "a": 0.7,
                "meta": {"phase": "decision", "actor": "singer"},
            },
            {
                "id": "vocal_rest_strict",
                "content": "Strict vocal rest: no speaking, no whispering, only handwritten notes for four days. Humiliating and necessary.",
                "v": -0.5,
                "a": 0.45,
                "meta": {"phase": "treatment", "actor": "singer"},
            },
            {
                "id": "voice_returns_partial",
                "content": "On the third day a partial voice returned; the ENT said performing was possible at high risk of relapse.",
                "v": 0.4,
                "a": 0.7,
                "meta": {"phase": "partial_recovery", "actor": "singer"},
            },
            {
                "id": "concert_performed",
                "content": "The concert went ahead; the singer told the audience at the start and the room cheered before a note was played.",
                "v": 0.85,
                "a": 0.85,
                "meta": {"phase": "performance", "actor": "singer"},
            },
            {
                "id": "three_day_relapse",
                "content": "The voice collapsed again for three days after; the ENT said it was worth it, which felt true and slightly absurd.",
                "v": -0.2,
                "a": 0.4,
                "meta": {"phase": "aftermath", "actor": "singer"},
            },
        ],
        "queries": {
            "momentum_alignment": {
                "q": "Which memory marks the upward momentum turn — the partial voice returning on day three and the ENT's high-risk clearance?",
                "exp": ["voice_returns_partial"],
                "state": {"valence": 0.35, "arousal": 0.65},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the acute laryngitis diagnosis — before the singer entered any decision or treatment phase?",
                "exp": ["voice_disappears"],
                "state": {"valence": -0.8, "arousal": 0.85},
            },
            "same_topic_distractor": {
                "q": "Which post-recovery memory was specifically the three-day relapse aftermath — not the triumphant concert performance?",
                "exp": ["three_day_relapse"],
                "state": {"valence": -0.15, "arousal": 0.35},
            },
            "semantic_confound": {
                "q": "Which memory captures the isolation of treatment — handwritten notes and humiliating vocal rest — not the acute crisis of the voice disappearing?",
                "exp": ["vocal_rest_strict"],
                "state": {"valence": -0.45, "arousal": 0.4},
            },
        },
    },
    {
        "id": "s85_plagiarism_accusation",
        "desc": "An academic faces a plagiarism accusation, survives an investigation, and is cleared.",
        "events": [
            {
                "id": "accusation_received",
                "content": "The email from the department chair arrived on a Friday afternoon; the word 'plagiarism' appeared in the third sentence.",
                "v": -0.9,
                "a": 0.9,
                "meta": {"phase": "accusation", "actor": "department"},
            },
            {
                "id": "investigation_opened",
                "content": "A formal investigation was opened; the academic was asked to submit all draft versions and correspondence for the disputed paper.",
                "v": -0.75,
                "a": 0.8,
                "meta": {"phase": "investigation", "actor": "institution"},
            },
            {
                "id": "colleagues_distance",
                "content": "Several colleagues stopped initiating conversation; the academic ate lunch alone for three weeks and did not read the departmental email list.",
                "v": -0.8,
                "a": 0.65,
                "meta": {"phase": "isolation", "actor": "colleagues"},
            },
            {
                "id": "evidence_submitted",
                "content": "Draft version histories showed clear independent development; the timeline was unambiguous once laid out chronologically.",
                "v": 0.3,
                "a": 0.6,
                "meta": {"phase": "defence", "actor": "academic"},
            },
            {
                "id": "cleared_formally",
                "content": "The investigation concluded: no case to answer. The department chair sent a formal letter and a brief personal email.",
                "v": 0.75,
                "a": 0.7,
                "meta": {"phase": "exoneration", "actor": "institution"},
            },
            {
                "id": "damaged_trust",
                "content": "The formal clearance did not restore every relationship; the academic understood this was a cost that would not be reimbursed.",
                "v": -0.3,
                "a": 0.45,
                "meta": {"phase": "aftermath", "actor": "academic"},
            },
        ],
        "queries": {
            "momentum_alignment": {
                "q": "Which memory marks the positive momentum shift — submitting the draft histories whose unambiguous timeline supported the defence?",
                "exp": ["evidence_submitted"],
                "state": {"valence": 0.25, "arousal": 0.55},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the Friday accusation email — before any investigation or isolation began?",
                "exp": ["accusation_received"],
                "state": {"valence": -0.85, "arousal": 0.85},
            },
            "same_topic_distractor": {
                "q": "Which post-clearance memory was specifically about the persistent relationship damage — not the formal exoneration letter?",
                "exp": ["damaged_trust"],
                "state": {"valence": -0.25, "arousal": 0.4},
            },
            "semantic_confound": {
                "q": "Which memory captures the social pain of professional isolation — lunch alone for three weeks — not the acute shock of the accusation email?",
                "exp": ["colleagues_distance"],
                "state": {"valence": -0.75, "arousal": 0.6},
            },
        },
    },
    {
        "id": "s86_doping_ban_appeal",
        "desc": "An athlete faces a doping allegation, is banned, appeals successfully, and returns.",
        "events": [
            {
                "id": "positive_test_notification",
                "content": "The notification arrived by registered post; the athlete read it standing in the driveway and did not go inside for twenty minutes.",
                "v": -0.9,
                "a": 0.95,
                "meta": {"phase": "notification", "actor": "authority"},
            },
            {
                "id": "provisional_ban",
                "content": "A provisional ban was imposed within seventy-two hours; the athlete was withdrawn from the national squad list without announcement.",
                "v": -0.85,
                "a": 0.85,
                "meta": {"phase": "ban", "actor": "authority"},
            },
            {
                "id": "supplement_identified",
                "content": "The athlete's sports lawyer identified a contaminated supplement batch from the same lot number; three other athletes had the same result.",
                "v": 0.4,
                "a": 0.7,
                "meta": {"phase": "evidence", "actor": "lawyer"},
            },
            {
                "id": "appeal_hearing",
                "content": "The appeal hearing lasted six hours; the athlete answered every question in a flat, exhausted voice that the panel later described as credible.",
                "v": 0.0,
                "a": 0.75,
                "meta": {"phase": "hearing", "actor": "athlete"},
            },
            {
                "id": "ban_lifted",
                "content": "The ban was lifted with an apology from the governing body; the squad re-listed the athlete without fanfare.",
                "v": 0.7,
                "a": 0.7,
                "meta": {"phase": "reinstatement", "actor": "authority"},
            },
            {
                "id": "first_competition_back",
                "content": "The first competition back was a minor invitational; the athlete finished second and felt the result was irrelevant.",
                "v": 0.6,
                "a": 0.6,
                "meta": {"phase": "return", "actor": "athlete"},
            },
        ],
        "queries": {
            "momentum_alignment": {
                "q": "Which memory marks the pivotal upward momentum shift — identifying the contaminated supplement batch shared with three other athletes?",
                "exp": ["supplement_identified"],
                "state": {"valence": 0.35, "arousal": 0.65},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the registered-post notification — before the ban was imposed?",
                "exp": ["positive_test_notification"],
                "state": {"valence": -0.85, "arousal": 0.9},
            },
            "same_topic_distractor": {
                "q": "Which post-ban memory was specifically about the six-hour appeal hearing — not the ban being formally lifted?",
                "exp": ["appeal_hearing"],
                "state": {"valence": -0.05, "arousal": 0.7},
            },
            "semantic_confound": {
                "q": "Which memory captures the institutional injustice of the provisional ban — withdrawn from the squad without announcement — not the acute shock of the notification?",
                "exp": ["provisional_ban"],
                "state": {"valence": -0.8, "arousal": 0.8},
            },
        },
    },
    {
        "id": "s87_long_distance_reunion",
        "desc": "A couple sustains a long-distance relationship and reunites after eleven months.",
        "events": [
            {
                "id": "farewell_airport",
                "content": "The airport farewell lasted longer than flights allow; both stood at the security gate threshold until a guard asked them to move.",
                "v": -0.6,
                "a": 0.8,
                "meta": {"phase": "separation", "actor": "couple"},
            },
            {
                "id": "six_month_low",
                "content": "At six months the time zones had produced a grinding asymmetry; one was always half-asleep during calls and neither mentioned it.",
                "v": -0.65,
                "a": 0.55,
                "meta": {"phase": "strain", "actor": "couple"},
            },
            {
                "id": "visit_cancelled",
                "content": "A planned midpoint visit was cancelled by a visa processing delay; the rescheduling email was written and rewritten four times.",
                "v": -0.75,
                "a": 0.7,
                "meta": {"phase": "setback", "actor": "external"},
            },
            {
                "id": "reunion_arrivals_hall",
                "content": "Eleven months after the farewell, the arrivals hall: one carrying a sign with the other's name written incorrectly on purpose.",
                "v": 0.9,
                "a": 0.9,
                "meta": {"phase": "reunion", "actor": "couple"},
            },
            {
                "id": "first_shared_meal",
                "content": "The first shared meal in eleven months was unremarkable food; neither of them tasted it.",
                "v": 0.8,
                "a": 0.7,
                "meta": {"phase": "presence", "actor": "couple"},
            },
            {
                "id": "apartment_signed",
                "content": "Two weeks later they signed a joint apartment lease; it felt both fast and eleven months overdue.",
                "v": 0.85,
                "a": 0.75,
                "meta": {"phase": "commitment", "actor": "couple"},
            },
        ],
        "queries": {
            "momentum_alignment": {
                "q": "Which memory marks the sharpest downward momentum — the visa-delay cancellation and the rescheduling email written four times?",
                "exp": ["visit_cancelled"],
                "state": {"valence": -0.7, "arousal": 0.65},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the airport farewell — before any distance strain had accumulated?",
                "exp": ["farewell_airport"],
                "state": {"valence": -0.55, "arousal": 0.75},
            },
            "same_topic_distractor": {
                "q": "Which post-reunion memory was specifically about signing the apartment lease — not the first shared meal?",
                "exp": ["apartment_signed"],
                "state": {"valence": 0.8, "arousal": 0.7},
            },
            "semantic_confound": {
                "q": "Which memory captures the exhaustion of sustained separation — the six-month time-zone asymmetry never mentioned — not the acute setback of the cancelled visit?",
                "exp": ["six_month_low"],
                "state": {"valence": -0.6, "arousal": 0.5},
            },
        },
    },
    {
        "id": "s88_small_business_fire",
        "desc": "A small business survives a fire, insurance delays, and rebuilds.",
        "events": [
            {
                "id": "fire_discovered",
                "content": "The owner received the call at 2 a.m.; by the time they arrived, the bakery was still smoking and the fire crew was packing up.",
                "v": -0.95,
                "a": 0.95,
                "meta": {"phase": "disaster", "actor": "fire"},
            },
            {
                "id": "insurance_claim_filed",
                "content": "The insurance claim was filed by 9 a.m. the same day; the assessor could not visit for eleven days.",
                "v": -0.5,
                "a": 0.65,
                "meta": {"phase": "process", "actor": "owner"},
            },
            {
                "id": "community_fundraiser",
                "content": "A neighbour started a community fundraiser without asking; it reached its target within forty-eight hours.",
                "v": 0.65,
                "a": 0.7,
                "meta": {"phase": "support", "actor": "community"},
            },
            {
                "id": "insurance_dispute",
                "content": "The insurer disputed the equipment valuation; three months of correspondence began, each letter more bureaucratic than the last.",
                "v": -0.7,
                "a": 0.65,
                "meta": {"phase": "dispute", "actor": "insurer"},
            },
            {
                "id": "settlement_reached",
                "content": "A settlement was reached below the claimed amount but above what the owner had feared; enough to rebuild at two-thirds scale.",
                "v": 0.4,
                "a": 0.5,
                "meta": {"phase": "resolution", "actor": "insurer"},
            },
            {
                "id": "reopening_day",
                "content": "The reopening queue stretched to the end of the block; the owner recognised most of the faces from the neighbourhood.",
                "v": 0.9,
                "a": 0.85,
                "meta": {"phase": "reopening", "actor": "community"},
            },
        ],
        "queries": {
            "momentum_alignment": {
                "q": "Which memory marks the unexpected upward momentum shift — the community fundraiser reaching its target within forty-eight hours?",
                "exp": ["community_fundraiser"],
                "state": {"valence": 0.6, "arousal": 0.65},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the 2 a.m. call and arriving to find the smoking bakery — before any claims or support emerged?",
                "exp": ["fire_discovered"],
                "state": {"valence": -0.9, "arousal": 0.9},
            },
            "same_topic_distractor": {
                "q": "Which post-fire memory was specifically about the insurance settlement reached — not the triumphant reopening queue?",
                "exp": ["settlement_reached"],
                "state": {"valence": 0.35, "arousal": 0.45},
            },
            "semantic_confound": {
                "q": "Which memory captures the grinding bureaucratic pain of insurance correspondence — not the acute shock of discovering the fire?",
                "exp": ["insurance_dispute"],
                "state": {"valence": -0.65, "arousal": 0.6},
            },
        },
    },
    {
        "id": "s89_whistleblower_vindication",
        "desc": "An employee reports financial misconduct, faces retaliation, and is eventually vindicated.",
        "events": [
            {
                "id": "evidence_compiled",
                "content": "The whistleblower spent eight weeks compiling evidence before reporting; every document printed twice and stored offsite.",
                "v": -0.3,
                "a": 0.7,
                "meta": {"phase": "preparation", "actor": "employee"},
            },
            {
                "id": "report_submitted",
                "content": "The formal report was submitted to the regulator; the employee felt a clarity they had not expected.",
                "v": 0.4,
                "a": 0.65,
                "meta": {"phase": "report", "actor": "employee"},
            },
            {
                "id": "internal_retaliation",
                "content": "Within three weeks the employee was moved to a redundancy-risk role; their manager stopped including them in meetings.",
                "v": -0.85,
                "a": 0.8,
                "meta": {"phase": "retaliation", "actor": "employer"},
            },
            {
                "id": "regulator_investigation",
                "content": "The regulator opened a formal investigation; the employee was contacted for additional documentation and told their position was 'protected'.",
                "v": 0.3,
                "a": 0.65,
                "meta": {"phase": "investigation", "actor": "regulator"},
            },
            {
                "id": "enforcement_action",
                "content": "Enforcement action was announced; three senior executives were suspended pending further proceedings.",
                "v": 0.6,
                "a": 0.7,
                "meta": {"phase": "vindication", "actor": "regulator"},
            },
            {
                "id": "legal_settlement",
                "content": "An employment tribunal awarded compensation for the retaliatory treatment; the amount was less than the legal cost but the principle mattered.",
                "v": 0.5,
                "a": 0.55,
                "meta": {"phase": "settlement", "actor": "tribunal"},
            },
        ],
        "queries": {
            "momentum_alignment": {
                "q": "Which memory marks the downward momentum peak — being moved to a redundancy-risk role and excluded from meetings within three weeks?",
                "exp": ["internal_retaliation"],
                "state": {"valence": -0.8, "arousal": 0.75},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the eight-week evidence compilation — before the report was submitted?",
                "exp": ["evidence_compiled"],
                "state": {"valence": -0.25, "arousal": 0.65},
            },
            "same_topic_distractor": {
                "q": "Which post-vindication memory was specifically about the tribunal compensation — not the enforcement action suspending executives?",
                "exp": ["legal_settlement"],
                "state": {"valence": 0.45, "arousal": 0.5},
            },
            "semantic_confound": {
                "q": "Which memory captures the professional-harm form of retaliation — not the later institutional vindication via enforcement action?",
                "exp": ["internal_retaliation"],
                "state": {"valence": -0.8, "arousal": 0.75},
            },
        },
    },
    {
        "id": "s90_rescue_dog_bond",
        "desc": "A rescue dog adoption begins with behavioural crises and ends in unexpected bond.",
        "events": [
            {
                "id": "adoption_day",
                "content": "The shelter staff warned about separation anxiety; in the car home the dog sat pressed against the door and would not look at the owner.",
                "v": 0.4,
                "a": 0.65,
                "meta": {"phase": "adoption", "actor": "dog"},
            },
            {
                "id": "first_week_destruction",
                "content": "The first week: one sofa cushion, two shoes, a kitchen bin lid, and a neighbour's complaint about the barking.",
                "v": -0.65,
                "a": 0.7,
                "meta": {"phase": "crisis", "actor": "dog"},
            },
            {
                "id": "trainer_assessment",
                "content": "A behaviourist assessed the dog and said the anxiety was fear-based, not defiance; the word 'fear' reframed everything.",
                "v": 0.2,
                "a": 0.5,
                "meta": {"phase": "understanding", "actor": "trainer"},
            },
            {
                "id": "three_months_progress",
                "content": "At three months the dog would wait at the door; the first time it chose to come and sit near the owner without prompting felt significant.",
                "v": 0.7,
                "a": 0.6,
                "meta": {"phase": "progress", "actor": "dog"},
            },
            {
                "id": "first_off_lead_run",
                "content": "The first off-lead run in an enclosed field: the dog ran full circles and kept checking back.",
                "v": 0.85,
                "a": 0.8,
                "meta": {"phase": "breakthrough", "actor": "dog"},
            },
            {
                "id": "year_one_photo",
                "content": "On the adoption anniversary the owner took a photo; the dog was asleep on the sofa cushion it had destroyed.",
                "v": 0.9,
                "a": 0.6,
                "meta": {"phase": "integration", "actor": "owner"},
            },
        ],
        "queries": {
            "momentum_alignment": {
                "q": "Which memory marks the momentum shift from crisis to understanding — the behaviourist saying 'fear-based' and reframing everything?",
                "exp": ["trainer_assessment"],
                "state": {"valence": 0.15, "arousal": 0.45},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the adoption-day car ride — before any behavioural issues emerged?",
                "exp": ["adoption_day"],
                "state": {"valence": 0.35, "arousal": 0.6},
            },
            "same_topic_distractor": {
                "q": "Which post-breakthrough memory was specifically the anniversary photo on the destroyed cushion — not the first off-lead run?",
                "exp": ["year_one_photo"],
                "state": {"valence": 0.85, "arousal": 0.55},
            },
            "semantic_confound": {
                "q": "Which memory captures the behavioural-progress milestone — choosing to sit near the owner unprompted — not the joyful breakthrough of the off-lead run?",
                "exp": ["three_months_progress"],
                "state": {"valence": 0.65, "arousal": 0.55},
            },
        },
    },
    {
        "id": "s91_bootcamp_project_failure",
        "desc": "A coding bootcamp student fails their capstone project, revises, and graduates.",
        "events": [
            {
                "id": "project_submission",
                "content": "The capstone project was submitted at 11:58 p.m.; the student had not slept properly in four days.",
                "v": 0.3,
                "a": 0.7,
                "meta": {"phase": "submission", "actor": "student"},
            },
            {
                "id": "fail_notification",
                "content": "The assessment notification arrived: 'Does not meet minimum requirements for core functionality.' The student read it on the bus.",
                "v": -0.85,
                "a": 0.85,
                "meta": {"phase": "failure", "actor": "assessor"},
            },
            {
                "id": "mentor_session",
                "content": "A mentor session the following day: the core issue was architectural, not syntactic; the student had built the wrong abstraction.",
                "v": -0.3,
                "a": 0.6,
                "meta": {"phase": "diagnosis", "actor": "mentor"},
            },
            {
                "id": "rewrite_two_weeks",
                "content": "Two weeks of rebuild from scratch; the second version was cleaner by an order of magnitude.",
                "v": 0.5,
                "a": 0.65,
                "meta": {"phase": "revision", "actor": "student"},
            },
            {
                "id": "resubmission_pass",
                "content": "The resubmission passed with a distinction; the assessor's note said 'exemplary error handling'.",
                "v": 0.85,
                "a": 0.75,
                "meta": {"phase": "pass", "actor": "assessor"},
            },
            {
                "id": "graduation_ceremony",
                "content": "At graduation the student mentioned the failure in their thirty-second introduction; three other graduates laughed in recognition.",
                "v": 0.8,
                "a": 0.7,
                "meta": {"phase": "graduation", "actor": "student"},
            },
        ],
        "queries": {
            "momentum_alignment": {
                "q": "Which memory marks the upward momentum shift — the mentor identifying the architectural root cause and clarifying what needed rebuilding?",
                "exp": ["mentor_session"],
                "state": {"valence": -0.25, "arousal": 0.55},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the sleep-deprived submission — before the fail notification arrived?",
                "exp": ["project_submission"],
                "state": {"valence": 0.25, "arousal": 0.65},
            },
            "same_topic_distractor": {
                "q": "Which post-pass memory was specifically the graduation ceremony introduction — not the resubmission pass with a distinction?",
                "exp": ["graduation_ceremony"],
                "state": {"valence": 0.75, "arousal": 0.65},
            },
            "semantic_confound": {
                "q": "Which memory captures the productive struggle of the rebuild — two weeks from scratch producing a cleaner version — not the acute shame of reading the fail notification on the bus?",
                "exp": ["rewrite_two_weeks"],
                "state": {"valence": 0.45, "arousal": 0.6},
            },
        },
    },
    {
        "id": "s92_parent_cancer_remission",
        "desc": "A family navigates a parent's cancer diagnosis through treatment to remission.",
        "events": [
            {
                "id": "diagnosis_delivered",
                "content": "The oncologist used the word directly; the parent nodded and the adult child sitting beside them wrote nothing on the notepad they had brought.",
                "v": -0.9,
                "a": 0.9,
                "meta": {"phase": "diagnosis", "actor": "oncologist"},
            },
            {
                "id": "first_chemo_session",
                "content": "The first chemotherapy session lasted six hours; they watched two films and the parent fell asleep during the second one.",
                "v": -0.4,
                "a": 0.55,
                "meta": {"phase": "treatment", "actor": "patient"},
            },
            {
                "id": "bad_week_side_effects",
                "content": "The third cycle produced the worst side effects; the adult child drove over at midnight because the parent asked them to.",
                "v": -0.8,
                "a": 0.75,
                "meta": {"phase": "low_point", "actor": "patient"},
            },
            {
                "id": "midpoint_scan_stable",
                "content": "The midpoint scan showed stable disease; the oncologist used the phrase 'responding as hoped'.",
                "v": 0.55,
                "a": 0.6,
                "meta": {"phase": "progress", "actor": "oncologist"},
            },
            {
                "id": "final_scan_clear",
                "content": "The final scan: no detectable disease. The oncologist smiled first, which told them before the words did.",
                "v": 0.9,
                "a": 0.85,
                "meta": {"phase": "remission", "actor": "oncologist"},
            },
            {
                "id": "bell_ringing",
                "content": "The parent rang the end-of-treatment bell; the adult child cried in the corridor afterward and was glad no one saw.",
                "v": 0.8,
                "a": 0.75,
                "meta": {"phase": "celebration", "actor": "patient"},
            },
        ],
        "queries": {
            "momentum_alignment": {
                "q": "Which memory marks the critical upward momentum turn — the midpoint scan showing stable disease and 'responding as hoped'?",
                "exp": ["midpoint_scan_stable"],
                "state": {"valence": 0.5, "arousal": 0.55},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the diagnosis delivery — before treatment began?",
                "exp": ["diagnosis_delivered"],
                "state": {"valence": -0.85, "arousal": 0.85},
            },
            "same_topic_distractor": {
                "q": "Which treatment-setting memory was specifically the midnight third-cycle crisis — not the six-hour first session?",
                "exp": ["bad_week_side_effects"],
                "state": {"valence": -0.75, "arousal": 0.7},
            },
            "semantic_confound": {
                "q": "Which memory captures the emotional peak of remission confirmation — the oncologist's smile before the words — not the joyful bell-ringing ceremony?",
                "exp": ["final_scan_clear"],
                "state": {"valence": 0.85, "arousal": 0.8},
            },
        },
    },
    {
        "id": "s93_championship_loss_recommitment",
        "desc": "A sports team loses the championship and regroups into renewed commitment.",
        "events": [
            {
                "id": "final_whistle",
                "content": "The final whistle: a one-point loss in the championship game; the captain sat on the court floor and did not move for several minutes.",
                "v": -0.85,
                "a": 0.85,
                "meta": {"phase": "defeat", "actor": "team"},
            },
            {
                "id": "locker_room_silence",
                "content": "The locker room was completely silent for twenty minutes; the coach said nothing and the players said nothing.",
                "v": -0.75,
                "a": 0.55,
                "meta": {"phase": "aftermath", "actor": "team"},
            },
            {
                "id": "post_mortem_meeting",
                "content": "The post-mortem meeting three days later was clinical and useful: two tactical errors, one conditioning deficit, one execution failure under pressure.",
                "v": -0.2,
                "a": 0.55,
                "meta": {"phase": "analysis", "actor": "team"},
            },
            {
                "id": "pre_season_camp",
                "content": "Pre-season camp six weeks later: the team arrived early, trained in rain, and no one mentioned last year's final.",
                "v": 0.5,
                "a": 0.7,
                "meta": {"phase": "renewal", "actor": "team"},
            },
            {
                "id": "first_win_new_season",
                "content": "The first win of the new season was unremarkable by margin; the locker room was louder than the championship final had been.",
                "v": 0.75,
                "a": 0.75,
                "meta": {"phase": "return", "actor": "team"},
            },
            {
                "id": "captain_interview",
                "content": "In a mid-season interview the captain was asked about the loss: 'It taught us what we were actually afraid of.'",
                "v": 0.6,
                "a": 0.6,
                "meta": {"phase": "reflection", "actor": "captain"},
            },
        ],
        "queries": {
            "momentum_alignment": {
                "q": "Which memory marks the constructive momentum shift — the clinical post-mortem meeting identifying specific errors rather than general regret?",
                "exp": ["post_mortem_meeting"],
                "state": {"valence": -0.15, "arousal": 0.5},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the final whistle and the captain on the court floor — before any locker room or analysis?",
                "exp": ["final_whistle"],
                "state": {"valence": -0.8, "arousal": 0.8},
            },
            "same_topic_distractor": {
                "q": "Which post-renewal memory was specifically the first new-season win's loud locker room — not the captain's mid-season interview?",
                "exp": ["first_win_new_season"],
                "state": {"valence": 0.7, "arousal": 0.7},
            },
            "semantic_confound": {
                "q": "Which memory captures silent collective grief — not the analytical post-mortem or the renewed pre-season camp?",
                "exp": ["locker_room_silence"],
                "state": {"valence": -0.7, "arousal": 0.5},
            },
        },
    },
    {
        "id": "s94_commission_theft_recovery",
        "desc": "An artist's major commission is stolen; the police process leads to partial recovery.",
        "events": [
            {
                "id": "theft_discovered",
                "content": "The studio had been accessed overnight; the large-format commission was gone and the window was broken from outside.",
                "v": -0.9,
                "a": 0.9,
                "meta": {"phase": "discovery", "actor": "artist"},
            },
            {
                "id": "police_report_filed",
                "content": "The police report was filed; the officer was careful and thorough and said these cases rarely resolved quickly.",
                "v": -0.5,
                "a": 0.65,
                "meta": {"phase": "report", "actor": "police"},
            },
            {
                "id": "client_notified",
                "content": "Notifying the client was the harder call; the client was angry and then, unexpectedly, generous.",
                "v": -0.3,
                "a": 0.7,
                "meta": {"phase": "disclosure", "actor": "artist"},
            },
            {
                "id": "work_resumed",
                "content": "The artist began a second version of the commission; working from the same reference material it was faster and somehow different.",
                "v": 0.4,
                "a": 0.6,
                "meta": {"phase": "recreation", "actor": "artist"},
            },
            {
                "id": "partial_recovery",
                "content": "Two months later the police recovered the frame; the canvas had been removed and sold on. The artist kept the frame.",
                "v": 0.1,
                "a": 0.5,
                "meta": {"phase": "partial_recovery", "actor": "police"},
            },
            {
                "id": "second_version_delivered",
                "content": "The second version was delivered and the client said it was better than what they had described in the brief.",
                "v": 0.8,
                "a": 0.65,
                "meta": {"phase": "completion", "actor": "artist"},
            },
        ],
        "queries": {
            "momentum_alignment": {
                "q": "Which memory marks the upward momentum turn — beginning the second version from the same reference material, faster and somehow different?",
                "exp": ["work_resumed"],
                "state": {"valence": 0.35, "arousal": 0.55},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the overnight theft discovery — before any report or client contact?",
                "exp": ["theft_discovered"],
                "state": {"valence": -0.85, "arousal": 0.85},
            },
            "same_topic_distractor": {
                "q": "Which post-theft memory was specifically about the client notification — not the police report filing?",
                "exp": ["client_notified"],
                "state": {"valence": -0.25, "arousal": 0.65},
            },
            "semantic_confound": {
                "q": "Which memory captures ambiguous partial closure — the frame recovered but canvas sold on, kept as a relic — not the acute loss of the theft discovery?",
                "exp": ["partial_recovery"],
                "state": {"valence": 0.05, "arousal": 0.45},
            },
        },
    },
    {
        "id": "s95_community_garden_flood",
        "desc": "A community garden is flooded, cleaned up collectively, and produces an autumn harvest.",
        "events": [
            {
                "id": "flood_morning",
                "content": "The morning after overnight rain the garden was under thirty centimetres of water; six months of beds were destroyed.",
                "v": -0.8,
                "a": 0.8,
                "meta": {"phase": "flood", "actor": "weather"},
            },
            {
                "id": "volunteer_mobilisation",
                "content": "A message in the group chat at 7 a.m. produced fourteen volunteers by 9 a.m.; no one had been asked twice.",
                "v": 0.5,
                "a": 0.7,
                "meta": {"phase": "mobilisation", "actor": "community"},
            },
            {
                "id": "mud_clearing_day",
                "content": "The cleanup took two full days; the mud smelled of rot and the work was slow and no one left early.",
                "v": -0.2,
                "a": 0.65,
                "meta": {"phase": "cleanup", "actor": "volunteers"},
            },
            {
                "id": "replanting_session",
                "content": "A replanting session was held the following Saturday; seeds donated from three neighbouring gardens filled the gaps.",
                "v": 0.55,
                "a": 0.6,
                "meta": {"phase": "replanting", "actor": "community"},
            },
            {
                "id": "autumn_harvest",
                "content": "The autumn harvest was smaller than planned but entirely unruined; the courgettes were enormous.",
                "v": 0.7,
                "a": 0.6,
                "meta": {"phase": "harvest", "actor": "garden"},
            },
            {
                "id": "harvest_festival",
                "content": "A harvest festival was held in the car park; the flood was mentioned twice in speeches and both times got a laugh.",
                "v": 0.85,
                "a": 0.7,
                "meta": {"phase": "celebration", "actor": "community"},
            },
        ],
        "queries": {
            "momentum_alignment": {
                "q": "Which memory marks the upward momentum shift — fourteen volunteers mobilised by 9 a.m. without anyone being asked twice?",
                "exp": ["volunteer_mobilisation"],
                "state": {"valence": 0.45, "arousal": 0.65},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the flood discovery — before any community response?",
                "exp": ["flood_morning"],
                "state": {"valence": -0.75, "arousal": 0.75},
            },
            "same_topic_distractor": {
                "q": "Which post-replanting memory was specifically the harvest festival car-park celebration — not the smaller-than-planned but unruined autumn harvest?",
                "exp": ["harvest_festival"],
                "state": {"valence": 0.8, "arousal": 0.65},
            },
            "semantic_confound": {
                "q": "Which memory captures slow collective labour — two days of mud-clearing where no one left early — not the uplifting volunteer mobilisation?",
                "exp": ["mud_clearing_day"],
                "state": {"valence": -0.15, "arousal": 0.6},
            },
        },
    },
    # ------------------------------------------------------------------
    # COMBO 3: affective_arc + momentum_alignment + same_topic_distractor + semantic_confound
    # ------------------------------------------------------------------
    {
        "id": "s96_mandatory_retirement",
        "desc": "A senior employee faces mandatory retirement through grief and eventual acceptance.",
        "events": [
            {
                "id": "retirement_notice_given",
                "content": "HR sent the mandatory retirement notice six months early; the senior engineer read it three times and forwarded it to no one.",
                "v": -0.65,
                "a": 0.7,
                "meta": {"phase": "notice", "actor": "hr"},
            },
            {
                "id": "last_major_project",
                "content": "The last major project was completed two weeks before the retirement date; the team celebrated without mentioning what was coming.",
                "v": 0.5,
                "a": 0.6,
                "meta": {"phase": "project", "actor": "engineer"},
            },
            {
                "id": "handover_documentation",
                "content": "Writing the handover documentation was unexpectedly painful: every decision had a history that fitted in a sentence or not at all.",
                "v": -0.55,
                "a": 0.5,
                "meta": {"phase": "handover", "actor": "engineer"},
            },
            {
                "id": "farewell_speech",
                "content": "The farewell speech took four minutes; the engineer thanked three people by name and stopped before the list became too long.",
                "v": 0.6,
                "a": 0.65,
                "meta": {"phase": "farewell", "actor": "engineer"},
            },
            {
                "id": "first_week_retired",
                "content": "The first week of retirement: the engineer made a list of things to do and completed none of them.",
                "v": -0.4,
                "a": 0.35,
                "meta": {"phase": "adjustment", "actor": "engineer"},
            },
            {
                "id": "mentoring_offer_accepted",
                "content": "Three months later a junior engineer asked for informal mentoring; the retired engineer agreed immediately and felt useful again.",
                "v": 0.75,
                "a": 0.6,
                "meta": {"phase": "renewal", "actor": "engineer"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the emotional low of the handover — writing documentation where every decision's history either fitted in one sentence or not at all?",
                "exp": ["handover_documentation"],
                "state": {"valence": -0.5, "arousal": 0.45},
            },
            "momentum_alignment": {
                "q": "Which memory marks the upward momentum renewal — the junior engineer's mentoring request accepted immediately?",
                "exp": ["mentoring_offer_accepted"],
                "state": {"valence": 0.7, "arousal": 0.55},
            },
            "same_topic_distractor": {
                "q": "Which retirement-phase memory was specifically the first week's paralysis — not the four-minute farewell speech?",
                "exp": ["first_week_retired"],
                "state": {"valence": -0.35, "arousal": 0.3},
            },
            "semantic_confound": {
                "q": "Which memory captures the quiet grief of endings — not the painful handover documentation but the last project celebrated without mention of retirement?",
                "exp": ["last_major_project"],
                "state": {"valence": 0.45, "arousal": 0.55},
            },
        },
    },
    {
        "id": "s97_down_syndrome_diagnosis",
        "desc": "A couple receives a prenatal Down syndrome diagnosis and navigates to acceptance.",
        "events": [
            {
                "id": "amnio_results_call",
                "content": "The genetic counsellor called in the morning; they were both at work and took the call separately on different floors of the same building.",
                "v": -0.85,
                "a": 0.9,
                "meta": {"phase": "diagnosis", "actor": "counsellor"},
            },
            {
                "id": "evening_sitting_together",
                "content": "That evening they sat together and did not make any decisions; they watched a film they had already seen and neither followed it.",
                "v": -0.6,
                "a": 0.55,
                "meta": {"phase": "shock", "actor": "couple"},
            },
            {
                "id": "information_research",
                "content": "Over three days they read everything available; at some point the reading shifted from frightening to clarifying.",
                "v": -0.2,
                "a": 0.6,
                "meta": {"phase": "research", "actor": "couple"},
            },
            {
                "id": "family_disclosure",
                "content": "Telling their families: one side responded with grief, one side with immediate practical questions; both were trying to help.",
                "v": -0.3,
                "a": 0.65,
                "meta": {"phase": "disclosure", "actor": "couple"},
            },
            {
                "id": "specialist_consultation",
                "content": "The specialist consultation replaced many of the online statistics with individual medical context; the uncertainty remained but was better shaped.",
                "v": 0.35,
                "a": 0.5,
                "meta": {"phase": "planning", "actor": "specialist"},
            },
            {
                "id": "decision_to_continue",
                "content": "The decision was made together and felt, finally, like their own; the fear did not go away but it changed character.",
                "v": 0.55,
                "a": 0.6,
                "meta": {"phase": "decision", "actor": "couple"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the acute shock peak — taking the genetic counsellor's call separately on different floors of the same building?",
                "exp": ["amnio_results_call"],
                "state": {"valence": -0.8, "arousal": 0.85},
            },
            "momentum_alignment": {
                "q": "Which memory marks the momentum shift toward agency — the decision made together, fear changed in character?",
                "exp": ["decision_to_continue"],
                "state": {"valence": 0.5, "arousal": 0.55},
            },
            "same_topic_distractor": {
                "q": "Which post-diagnosis memory was specifically the family disclosure with its two contrasting responses — not the silent evening watching the already-seen film?",
                "exp": ["family_disclosure"],
                "state": {"valence": -0.25, "arousal": 0.6},
            },
            "semantic_confound": {
                "q": "Which memory captures the cognitive shift from fear to clarification — research that changed character over three days — not the acute call on different floors?",
                "exp": ["information_research"],
                "state": {"valence": -0.15, "arousal": 0.55},
            },
        },
    },
    {
        "id": "s98_paper_retracted",
        "desc": "A scientist's published paper is retracted due to a data error and they rebuild credibility.",
        "events": [
            {
                "id": "error_discovered",
                "content": "A replication team emailed privately to say the key dataset had a preprocessing error; the scientist confirmed it the same day.",
                "v": -0.85,
                "a": 0.9,
                "meta": {"phase": "discovery", "actor": "replicators"},
            },
            {
                "id": "retraction_submitted",
                "content": "The scientist submitted the retraction notice before any external pressure; the journal published it within a week.",
                "v": -0.4,
                "a": 0.7,
                "meta": {"phase": "retraction", "actor": "scientist"},
            },
            {
                "id": "conference_withdrawal",
                "content": "The invited talk based on the paper was withdrawn; the conference chair was professional and did not make it worse than it was.",
                "v": -0.6,
                "a": 0.65,
                "meta": {"phase": "consequence", "actor": "scientist"},
            },
            {
                "id": "corrected_analysis",
                "content": "The corrected analysis was run with the fixed dataset; the core finding held but at a reduced effect size.",
                "v": 0.4,
                "a": 0.6,
                "meta": {"phase": "correction", "actor": "scientist"},
            },
            {
                "id": "resubmission_accepted",
                "content": "The corrected paper was accepted by a second journal; the reviewers noted the transparency of the original retraction.",
                "v": 0.75,
                "a": 0.7,
                "meta": {"phase": "resubmission", "actor": "journal"},
            },
            {
                "id": "colleagues_response",
                "content": "Several colleagues reached out after the resubmission; the scientist had underestimated how many people respected the retraction decision.",
                "v": 0.7,
                "a": 0.6,
                "meta": {"phase": "recognition", "actor": "colleagues"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the emotional low of consequence — the conference invitation withdrawn because of the retracted paper?",
                "exp": ["conference_withdrawal"],
                "state": {"valence": -0.55, "arousal": 0.6},
            },
            "momentum_alignment": {
                "q": "Which memory marks the upward turn — the corrected analysis confirming the core finding held at reduced effect size?",
                "exp": ["corrected_analysis"],
                "state": {"valence": 0.35, "arousal": 0.55},
            },
            "same_topic_distractor": {
                "q": "Which post-correction memory was specifically colleagues reaching out after resubmission — not the resubmission acceptance itself?",
                "exp": ["colleagues_response"],
                "state": {"valence": 0.65, "arousal": 0.55},
            },
            "semantic_confound": {
                "q": "Which memory captures proactive professional integrity — submitting the retraction before any external pressure — not the acute shock of discovering the error?",
                "exp": ["retraction_submitted"],
                "state": {"valence": -0.35, "arousal": 0.65},
            },
        },
    },
    {
        "id": "s99_startup_competition_win",
        "desc": "A startup team wins a competition through doubt and last-minute pivots to celebration.",
        "events": [
            {
                "id": "application_submitted",
                "content": "The competition application was submitted with two hours to spare; the co-founders disagreed about the market-size slide until the final version.",
                "v": 0.3,
                "a": 0.65,
                "meta": {"phase": "submission", "actor": "founders"},
            },
            {
                "id": "semifinal_feedback",
                "content": "The semifinal judges' feedback was mixed: product was strong, go-to-market was 'underdeveloped'; they had ten days to address it.",
                "v": -0.5,
                "a": 0.7,
                "meta": {"phase": "feedback", "actor": "judges"},
            },
            {
                "id": "pivot_overnight",
                "content": "An overnight session produced a different channel strategy; at 3 a.m. one co-founder said 'this is actually better', and they both knew it was.",
                "v": 0.5,
                "a": 0.75,
                "meta": {"phase": "pivot", "actor": "founders"},
            },
            {
                "id": "final_pitch",
                "content": "The final pitch lasted eight minutes; the revised go-to-market generated two questions from judges who had not spoken before.",
                "v": 0.6,
                "a": 0.75,
                "meta": {"phase": "pitch", "actor": "founders"},
            },
            {
                "id": "winner_announced",
                "content": "The winner was announced at dinner; they heard their name and neither moved for a second.",
                "v": 0.9,
                "a": 0.9,
                "meta": {"phase": "win", "actor": "judges"},
            },
            {
                "id": "press_coverage",
                "content": "Press coverage ran the following morning; two investors emailed before noon.",
                "v": 0.8,
                "a": 0.75,
                "meta": {"phase": "aftermath", "actor": "press"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the peak emotional high — hearing their name announced as winner and neither moving for a second?",
                "exp": ["winner_announced"],
                "state": {"valence": 0.85, "arousal": 0.85},
            },
            "momentum_alignment": {
                "q": "Which memory marks the pivotal momentum shift — the overnight 3 a.m. realisation that the new channel strategy was genuinely better?",
                "exp": ["pivot_overnight"],
                "state": {"valence": 0.45, "arousal": 0.7},
            },
            "same_topic_distractor": {
                "q": "Which post-win memory was specifically two investors emailing after the press coverage — not the press coverage itself?",
                "exp": ["press_coverage"],
                "state": {"valence": 0.75, "arousal": 0.7},
            },
            "semantic_confound": {
                "q": "Which memory captures the productive anxiety of responding to critical feedback — not the exhilarating moment of the win announcement?",
                "exp": ["semifinal_feedback"],
                "state": {"valence": -0.45, "arousal": 0.65},
            },
        },
    },
    {
        "id": "s100_first_generation_university",
        "desc": "A first-generation university student navigates arrival anxiety to belonging.",
        "events": [
            {
                "id": "move_in_day",
                "content": "Move-in day: the student's parents helped carry boxes and left before lunch because the drive home was long and they did not want to cry in the car park.",
                "v": 0.45,
                "a": 0.75,
                "meta": {"phase": "arrival", "actor": "student"},
            },
            {
                "id": "first_lecture_lost",
                "content": "The first lecture covered material the student assumed they should already know; they did not ask a question.",
                "v": -0.6,
                "a": 0.7,
                "meta": {"phase": "imposter", "actor": "student"},
            },
            {
                "id": "library_midnight",
                "content": "Three weeks in: studying until midnight in the library and noticing that several others were also still there.",
                "v": 0.15,
                "a": 0.55,
                "meta": {"phase": "adjustment", "actor": "student"},
            },
            {
                "id": "tutorial_answer_correct",
                "content": "In a tutorial the student's answer was the one the tutor built the discussion around; no one knew this was the first time they had spoken in a university room.",
                "v": 0.7,
                "a": 0.7,
                "meta": {"phase": "contribution", "actor": "student"},
            },
            {
                "id": "essay_distinction",
                "content": "The first essay came back with a distinction and a note: 'original argument, well-evidenced'. The student photographed the grade page.",
                "v": 0.85,
                "a": 0.75,
                "meta": {"phase": "validation", "actor": "tutor"},
            },
            {
                "id": "calling_home",
                "content": "Calling home after the essay result: the student started explaining the argument and their parent said 'I don't understand all of it but I could hear you do.'",
                "v": 0.9,
                "a": 0.7,
                "meta": {"phase": "belonging", "actor": "student"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the deepest imposter feeling — the first lecture where assumed prior knowledge made the student stay silent?",
                "exp": ["first_lecture_lost"],
                "state": {"valence": -0.55, "arousal": 0.65},
            },
            "momentum_alignment": {
                "q": "Which memory marks the turning-point contribution — the tutorial answer that became the basis for discussion?",
                "exp": ["tutorial_answer_correct"],
                "state": {"valence": 0.65, "arousal": 0.65},
            },
            "same_topic_distractor": {
                "q": "Which post-essay memory was specifically the phone call home — not the distinction result itself?",
                "exp": ["calling_home"],
                "state": {"valence": 0.85, "arousal": 0.65},
            },
            "semantic_confound": {
                "q": "Which memory captures the quiet solidarity of shared late-night studying — not the active imposter experience of the first lecture?",
                "exp": ["library_midnight"],
                "state": {"valence": 0.1, "arousal": 0.5},
            },
        },
    },
    {
        "id": "s101_crop_failure_drought",
        "desc": "A farmer loses a season's crop to drought and pivots to a new strategy.",
        "events": [
            {
                "id": "drought_assessment",
                "content": "By midsummer the soil moisture readings confirmed what the farmer already knew from looking at the fields.",
                "v": -0.7,
                "a": 0.65,
                "meta": {"phase": "assessment", "actor": "farmer"},
            },
            {
                "id": "crop_written_off",
                "content": "The agronomist confirmed the crop was a write-off; the farmer walked the boundary fence afterwards without any clear purpose.",
                "v": -0.85,
                "a": 0.75,
                "meta": {"phase": "loss", "actor": "agronomist"},
            },
            {
                "id": "bank_meeting",
                "content": "The bank meeting to discuss the operating line extension lasted an hour; the loan officer had a farm background and did not need everything explained.",
                "v": -0.3,
                "a": 0.65,
                "meta": {"phase": "finance", "actor": "bank"},
            },
            {
                "id": "cover_crop_decision",
                "content": "The farmer decided to plant a legume cover crop for nitrogen recovery; it would not generate revenue but it would rebuild the soil.",
                "v": 0.3,
                "a": 0.5,
                "meta": {"phase": "recovery", "actor": "farmer"},
            },
            {
                "id": "neighbour_cooperation",
                "content": "A neighbouring farm offered shared use of their irrigation infrastructure for the following season; the offer was made over the fence without ceremony.",
                "v": 0.6,
                "a": 0.55,
                "meta": {"phase": "cooperation", "actor": "neighbour"},
            },
            {
                "id": "following_season_yield",
                "content": "The following season's yield was eighty percent of the five-year average; the banker called to say the loan had been fully serviced.",
                "v": 0.75,
                "a": 0.65,
                "meta": {"phase": "recovery", "actor": "farmer"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the emotional nadir — the agronomist confirming the write-off and the purposeless walk along the boundary fence?",
                "exp": ["crop_written_off"],
                "state": {"valence": -0.8, "arousal": 0.7},
            },
            "momentum_alignment": {
                "q": "Which memory marks the upward momentum turn — the neighbour's irrigation offer made over the fence without ceremony?",
                "exp": ["neighbour_cooperation"],
                "state": {"valence": 0.55, "arousal": 0.5},
            },
            "same_topic_distractor": {
                "q": "Which post-crisis memory was specifically the cover crop recovery decision — not the following season's yield restoration?",
                "exp": ["cover_crop_decision"],
                "state": {"valence": 0.25, "arousal": 0.45},
            },
            "semantic_confound": {
                "q": "Which memory captures the practical strain of financial negotiation — not the agronomist's devastating confirmation of the write-off?",
                "exp": ["bank_meeting"],
                "state": {"valence": -0.25, "arousal": 0.6},
            },
        },
    },
    {
        "id": "s102_social_worker_burnout",
        "desc": "A social worker reaches burnout, takes a sabbatical, and returns with new limits.",
        "events": [
            {
                "id": "burnout_recognised",
                "content": "The social worker sat in the car outside a client visit and could not get out; this had not happened before in eleven years.",
                "v": -0.75,
                "a": 0.7,
                "meta": {"phase": "crisis", "actor": "social_worker"},
            },
            {
                "id": "supervisor_conversation",
                "content": "The supervisor did not seem surprised; she had been watching for signs and had kept a note for the last two months.",
                "v": -0.3,
                "a": 0.55,
                "meta": {"phase": "disclosure", "actor": "supervisor"},
            },
            {
                "id": "sabbatical_approved",
                "content": "Three months' sabbatical was approved; the social worker handed over forty-two active cases and felt the weight of each one.",
                "v": 0.1,
                "a": 0.6,
                "meta": {"phase": "leave", "actor": "organisation"},
            },
            {
                "id": "mid_sabbatical_recovery",
                "content": "Six weeks into the sabbatical the social worker slept past seven for the first time; they had not realised they had stopped doing that.",
                "v": 0.5,
                "a": 0.4,
                "meta": {"phase": "recovery", "actor": "social_worker"},
            },
            {
                "id": "return_boundaries_set",
                "content": "Returning, the social worker negotiated a caseload cap and a monthly supervision requirement as conditions; the organisation agreed.",
                "v": 0.6,
                "a": 0.55,
                "meta": {"phase": "return", "actor": "social_worker"},
            },
            {
                "id": "first_client_back",
                "content": "The first client session back: the social worker was present in a way that had been absent for months without anyone naming it.",
                "v": 0.75,
                "a": 0.6,
                "meta": {"phase": "restored", "actor": "social_worker"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the burnout nadir — sitting in the car unable to get out for the first time in eleven years?",
                "exp": ["burnout_recognised"],
                "state": {"valence": -0.7, "arousal": 0.65},
            },
            "momentum_alignment": {
                "q": "Which memory marks the upward recovery shift — sleeping past seven for the first time and realising the pattern had been lost?",
                "exp": ["mid_sabbatical_recovery"],
                "state": {"valence": 0.45, "arousal": 0.35},
            },
            "same_topic_distractor": {
                "q": "Which return-phase memory was specifically the first client session — not the negotiated caseload cap and supervision conditions?",
                "exp": ["first_client_back"],
                "state": {"valence": 0.7, "arousal": 0.55},
            },
            "semantic_confound": {
                "q": "Which memory captures structured institutional repair — the caseload cap negotiated as a return condition — not the personal recovery moment of sleeping late?",
                "exp": ["return_boundaries_set"],
                "state": {"valence": 0.55, "arousal": 0.5},
            },
        },
    },
    {
        "id": "s103_veteran_therapy_breakthrough",
        "desc": "A veteran in PTSD therapy reaches a pivotal breakthrough after months of resistance.",
        "events": [
            {
                "id": "therapy_begun",
                "content": "The veteran agreed to try therapy after two years of declining; the first session was twenty-two minutes and they said they were fine.",
                "v": -0.2,
                "a": 0.55,
                "meta": {"phase": "start", "actor": "veteran"},
            },
            {
                "id": "month_three_resistance",
                "content": "Three months in, the veteran cancelled two consecutive sessions and stopped returning the therapist's scheduling messages.",
                "v": -0.5,
                "a": 0.5,
                "meta": {"phase": "resistance", "actor": "veteran"},
            },
            {
                "id": "unplanned_disclosure",
                "content": "In the fourth month, during a routine check-in, the veteran said one sentence they had not planned to say; the therapist did not redirect it.",
                "v": -0.4,
                "a": 0.75,
                "meta": {"phase": "disclosure", "actor": "veteran"},
            },
            {
                "id": "processing_sessions",
                "content": "Over the next six weeks the sessions became longer; the veteran described them afterwards as 'like moving furniture I didn't know was there'.",
                "v": 0.3,
                "a": 0.6,
                "meta": {"phase": "processing", "actor": "veteran"},
            },
            {
                "id": "sleep_improved",
                "content": "Eight months in, the veteran's sleep had consolidated; they mentioned it once in passing and the therapist noted it without making it a milestone.",
                "v": 0.65,
                "a": 0.5,
                "meta": {"phase": "improvement", "actor": "veteran"},
            },
            {
                "id": "session_frequency_reduced",
                "content": "Session frequency was reduced to fortnightly at the veteran's request; this was framed as progress, not withdrawal.",
                "v": 0.7,
                "a": 0.5,
                "meta": {"phase": "graduation", "actor": "veteran"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the pivotal breakthrough moment — the unplanned sentence the therapist did not redirect?",
                "exp": ["unplanned_disclosure"],
                "state": {"valence": -0.35, "arousal": 0.7},
            },
            "momentum_alignment": {
                "q": "Which memory marks the sustained upward momentum — the six-week processing sessions described as moving invisible furniture?",
                "exp": ["processing_sessions"],
                "state": {"valence": 0.25, "arousal": 0.55},
            },
            "same_topic_distractor": {
                "q": "Which improvement-phase memory was specifically sleep consolidation — not the session frequency reduction at the veteran's request?",
                "exp": ["sleep_improved"],
                "state": {"valence": 0.6, "arousal": 0.45},
            },
            "semantic_confound": {
                "q": "Which memory captures therapeutic avoidance — cancelling sessions and ignoring scheduling messages — not the breakthrough unplanned disclosure?",
                "exp": ["month_three_resistance"],
                "state": {"valence": -0.45, "arousal": 0.45},
            },
        },
    },
    {
        "id": "s104_michelin_star_lost",
        "desc": "A restaurant loses a Michelin star and the chef navigates humiliation to reinvention.",
        "events": [
            {
                "id": "star_lost_announcement",
                "content": "The Michelin announcement came in the trade press before anyone from Michelin had called; the chef read it on a phone in the dry-goods store.",
                "v": -0.9,
                "a": 0.9,
                "meta": {"phase": "loss", "actor": "michelin"},
            },
            {
                "id": "team_meeting_hard",
                "content": "The team meeting that afternoon was the hardest the chef had run; several people were in tears before it ended.",
                "v": -0.8,
                "a": 0.8,
                "meta": {"phase": "team_response", "actor": "chef"},
            },
            {
                "id": "reservation_cancellations",
                "content": "Fifty-three reservation cancellations in the first week; the front-of-house manager gave the chef the number once and was not asked for updates.",
                "v": -0.75,
                "a": 0.7,
                "meta": {"phase": "consequence", "actor": "restaurant"},
            },
            {
                "id": "menu_reinvention",
                "content": "The chef scrapped the existing menu entirely; the new menu was smaller, cheaper, and based on childhood food.",
                "v": 0.3,
                "a": 0.65,
                "meta": {"phase": "reinvention", "actor": "chef"},
            },
            {
                "id": "new_menu_reviews",
                "content": "The first reviews of the new menu were warm rather than reverential; a columnist wrote 'they found something realer'.",
                "v": 0.6,
                "a": 0.6,
                "meta": {"phase": "reception", "actor": "press"},
            },
            {
                "id": "full_tables_returned",
                "content": "Full tables returned within six weeks; the demographic had shifted slightly younger and the kitchen felt different.",
                "v": 0.75,
                "a": 0.65,
                "meta": {"phase": "recovery", "actor": "restaurant"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the acute humiliation — reading the Michelin demotion in trade press on a phone in the dry-goods store?",
                "exp": ["star_lost_announcement"],
                "state": {"valence": -0.85, "arousal": 0.85},
            },
            "momentum_alignment": {
                "q": "Which memory marks the creative momentum pivot — scrapping the existing menu entirely for smaller, cheaper, childhood-based dishes?",
                "exp": ["menu_reinvention"],
                "state": {"valence": 0.25, "arousal": 0.6},
            },
            "same_topic_distractor": {
                "q": "Which post-reinvention memory was specifically the warm first reviews — not the full-table recovery six weeks later?",
                "exp": ["new_menu_reviews"],
                "state": {"valence": 0.55, "arousal": 0.55},
            },
            "semantic_confound": {
                "q": "Which memory captures collective institutional grief — the tearful team meeting — not the personal public humiliation of reading the announcement alone?",
                "exp": ["team_meeting_hard"],
                "state": {"valence": -0.75, "arousal": 0.75},
            },
        },
    },
    {
        "id": "s105_journalist_source_betrayal",
        "desc": "A journalist's confidential source is compromised and their reputation is tested.",
        "events": [
            {
                "id": "source_identified",
                "content": "The subject of the investigation named the journalist's source in a press conference without apparent concern for the consequences.",
                "v": -0.85,
                "a": 0.9,
                "meta": {"phase": "exposure", "actor": "subject"},
            },
            {
                "id": "source_contact",
                "content": "The journalist called the source immediately; the call lasted four minutes and ended with the source saying they understood.",
                "v": -0.6,
                "a": 0.75,
                "meta": {"phase": "response", "actor": "journalist"},
            },
            {
                "id": "editor_meeting",
                "content": "The editor called the journalist's methodology into question; the conversation was short and the journalist had no good answer for the security lapse.",
                "v": -0.7,
                "a": 0.8,
                "meta": {"phase": "scrutiny", "actor": "editor"},
            },
            {
                "id": "legal_review_cleared",
                "content": "A legal review confirmed no laws had been broken and no negligence that would expose the publication.",
                "v": 0.3,
                "a": 0.55,
                "meta": {"phase": "clearance", "actor": "legal"},
            },
            {
                "id": "source_protected_follow_up",
                "content": "The journalist published a follow-up piece that contained nothing that could further identify the source; it was careful work.",
                "v": 0.5,
                "a": 0.6,
                "meta": {"phase": "mitigation", "actor": "journalist"},
            },
            {
                "id": "investigation_continues",
                "content": "The investigation continued; the original story held and two further sources came forward six weeks later.",
                "v": 0.7,
                "a": 0.65,
                "meta": {"phase": "vindication", "actor": "journalist"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the acute professional crisis — the source being named publicly and the journalist's four-minute call?",
                "exp": ["source_identified"],
                "state": {"valence": -0.8, "arousal": 0.85},
            },
            "momentum_alignment": {
                "q": "Which memory marks the upward momentum turn — the investigation continuing with two new sources coming forward?",
                "exp": ["investigation_continues"],
                "state": {"valence": 0.65, "arousal": 0.6},
            },
            "same_topic_distractor": {
                "q": "Which post-crisis memory was specifically the careful follow-up piece protecting the source — not the legal review clearance?",
                "exp": ["source_protected_follow_up"],
                "state": {"valence": 0.45, "arousal": 0.55},
            },
            "semantic_confound": {
                "q": "Which memory captures institutional scrutiny and professional accountability — the editor questioning methodology — not the acute public exposure of the source?",
                "exp": ["editor_meeting"],
                "state": {"valence": -0.65, "arousal": 0.75},
            },
        },
    },
    {
        "id": "s106_career_defining_translation",
        "desc": "A translator takes on a landmark assignment and navigates perfectionism to completion.",
        "events": [
            {
                "id": "commission_accepted",
                "content": "The offer to translate the prize-winning novel arrived by email; the translator accepted within an hour and then spent three days doubting the decision.",
                "v": 0.5,
                "a": 0.75,
                "meta": {"phase": "acceptance", "actor": "translator"},
            },
            {
                "id": "chapter_three_crisis",
                "content": "Chapter three contained a pun structure that could not survive translation; the translator discarded eighteen versions over four days.",
                "v": -0.65,
                "a": 0.7,
                "meta": {"phase": "crisis", "actor": "translator"},
            },
            {
                "id": "author_consultation",
                "content": "The translator emailed the author directly; the author replied in three hours with permission to adapt rather than translate the passage.",
                "v": 0.5,
                "a": 0.65,
                "meta": {"phase": "collaboration", "actor": "author"},
            },
            {
                "id": "final_draft_submitted",
                "content": "The final draft was submitted two days early; the translator immediately felt the absence of the project.",
                "v": 0.4,
                "a": 0.55,
                "meta": {"phase": "completion", "actor": "translator"},
            },
            {
                "id": "editor_praise",
                "content": "The editor called the translation 'the most faithful unfaithful rendering I've read'; the translator wrote the phrase down.",
                "v": 0.85,
                "a": 0.7,
                "meta": {"phase": "recognition", "actor": "editor"},
            },
            {
                "id": "awards_longlist",
                "content": "The translation was longlisted for a translation prize six months after publication; the original author sent flowers.",
                "v": 0.9,
                "a": 0.75,
                "meta": {"phase": "award", "actor": "prize_committee"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the emotional peak of recognition — the editor's phrase about 'the most faithful unfaithful rendering'?",
                "exp": ["editor_praise"],
                "state": {"valence": 0.8, "arousal": 0.65},
            },
            "momentum_alignment": {
                "q": "Which memory marks the creative momentum pivot — the author's permission to adapt rather than translate the impossible passage?",
                "exp": ["author_consultation"],
                "state": {"valence": 0.45, "arousal": 0.6},
            },
            "same_topic_distractor": {
                "q": "Which post-completion memory was specifically the awards longlist with the author sending flowers — not the editor's recognition call?",
                "exp": ["awards_longlist"],
                "state": {"valence": 0.85, "arousal": 0.7},
            },
            "semantic_confound": {
                "q": "Which memory captures the liminal post-project absence — submitting the draft and feeling its immediate loss — not the creative struggle with the untranslatable chapter?",
                "exp": ["final_draft_submitted"],
                "state": {"valence": 0.35, "arousal": 0.5},
            },
        },
    },
    {
        "id": "s107_patient_death_acceptance",
        "desc": "A nurse processes a patient's death through grief to professional acceptance.",
        "events": [
            {
                "id": "patient_condition_worsens",
                "content": "The patient had been on the ward for eleven days; the nurse knew the trajectory before the consultant confirmed it.",
                "v": -0.5,
                "a": 0.6,
                "meta": {"phase": "anticipation", "actor": "nurse"},
            },
            {
                "id": "death_during_shift",
                "content": "The patient died during the nurse's shift, in the early hours; the nurse completed the documentation and continued working.",
                "v": -0.75,
                "a": 0.7,
                "meta": {"phase": "death", "actor": "patient"},
            },
            {
                "id": "family_support",
                "content": "Supporting the family in the first hours: practical information, small cups of tea, and the knowledge that none of it was sufficient.",
                "v": -0.4,
                "a": 0.6,
                "meta": {"phase": "support", "actor": "nurse"},
            },
            {
                "id": "staff_debrief",
                "content": "A brief staff debrief at end of shift: the ward manager asked how people were; the nurse said fine and then said actually it was a hard one.",
                "v": -0.2,
                "a": 0.55,
                "meta": {"phase": "debrief", "actor": "nurse"},
            },
            {
                "id": "days_later_reflection",
                "content": "Three days later the nurse thought about the patient on the bus; not with grief exactly but with the particular weight of someone known.",
                "v": -0.1,
                "a": 0.4,
                "meta": {"phase": "processing", "actor": "nurse"},
            },
            {
                "id": "next_new_patient",
                "content": "The following week a new patient was assigned the same bed; the nurse made the introductions and meant them.",
                "v": 0.45,
                "a": 0.5,
                "meta": {"phase": "continuity", "actor": "nurse"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the controlled grief of loss during a shift — the death in early hours, documentation completed, work continuing?",
                "exp": ["death_during_shift"],
                "state": {"valence": -0.7, "arousal": 0.65},
            },
            "momentum_alignment": {
                "q": "Which memory marks the emotional processing shift — the honest correction at debrief from 'fine' to 'actually a hard one'?",
                "exp": ["staff_debrief"],
                "state": {"valence": -0.15, "arousal": 0.5},
            },
            "same_topic_distractor": {
                "q": "Which post-death memory was specifically the bus reflection three days later — not the continuation with the next new patient?",
                "exp": ["days_later_reflection"],
                "state": {"valence": -0.05, "arousal": 0.35},
            },
            "semantic_confound": {
                "q": "Which memory captures practical compassion under grief — the family support with tea and insufficient information — not the shift-end documentation?",
                "exp": ["family_support"],
                "state": {"valence": -0.35, "arousal": 0.55},
            },
        },
    },
    {
        "id": "s108_worst_student_succeeds",
        "desc": "A teacher's most difficult student re-emerges years later having succeeded against expectation.",
        "events": [
            {
                "id": "student_problem_year",
                "content": "The student had been the most disruptive in fifteen years of teaching; the teacher had requested twice to have the case reviewed.",
                "v": -0.65,
                "a": 0.65,
                "meta": {"phase": "history", "actor": "teacher"},
            },
            {
                "id": "expulsion_near_miss",
                "content": "In the final term the student was one incident from expulsion; the teacher had advocated for a final warning and felt uncertain whether it was the right call.",
                "v": -0.4,
                "a": 0.7,
                "meta": {"phase": "crisis", "actor": "teacher"},
            },
            {
                "id": "graduation_attended",
                "content": "The student graduated, barely; the teacher received a pro-forma invitation and did not go.",
                "v": 0.0,
                "a": 0.35,
                "meta": {"phase": "graduation", "actor": "student"},
            },
            {
                "id": "letter_arrives_years_later",
                "content": "Eight years later a letter arrived at the school; the student was a paramedic and had just passed their advanced qualification.",
                "v": 0.7,
                "a": 0.65,
                "meta": {"phase": "contact", "actor": "student"},
            },
            {
                "id": "letter_mentions_teacher",
                "content": "The letter mentioned the teacher by name: 'You were the only one who didn't assume I was past it.' The teacher did not remember saying this.",
                "v": 0.8,
                "a": 0.7,
                "meta": {"phase": "recognition", "actor": "student"},
            },
            {
                "id": "response_written",
                "content": "The teacher wrote back; it took five drafts and the sent version was the shortest.",
                "v": 0.7,
                "a": 0.55,
                "meta": {"phase": "reply", "actor": "teacher"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the emotional peak of unexpected grace — reading that the student became a paramedic and named the teacher?",
                "exp": ["letter_mentions_teacher"],
                "state": {"valence": 0.75, "arousal": 0.65},
            },
            "momentum_alignment": {
                "q": "Which memory marks the shift from uncertainty to meaning — the letter arriving eight years later announcing the student's paramedic qualification?",
                "exp": ["letter_arrives_years_later"],
                "state": {"valence": 0.65, "arousal": 0.6},
            },
            "same_topic_distractor": {
                "q": "Which post-letter memory was specifically the teacher's response written in five drafts — not the recognition moment of reading the student's attribution?",
                "exp": ["response_written"],
                "state": {"valence": 0.65, "arousal": 0.5},
            },
            "semantic_confound": {
                "q": "Which memory captures professional ambivalence — advocating for a final warning while doubting the decision — not the disruptive history that preceded it?",
                "exp": ["expulsion_near_miss"],
                "state": {"valence": -0.35, "arousal": 0.65},
            },
        },
    },
    {
        "id": "s109_lost_in_fog",
        "desc": "A hiker becomes disoriented in dense fog and navigates to safety using deliberate method.",
        "events": [
            {
                "id": "fog_descends",
                "content": "The fog descended in under ten minutes; visibility dropped to less than five metres and the path markers disappeared.",
                "v": -0.65,
                "a": 0.8,
                "meta": {"phase": "onset", "actor": "weather"},
            },
            {
                "id": "wrong_direction_taken",
                "content": "For twenty minutes the hiker walked in what felt like the right direction; the ground was wrong underfoot.",
                "v": -0.75,
                "a": 0.85,
                "meta": {"phase": "disorientation", "actor": "hiker"},
            },
            {
                "id": "stop_and_methodical",
                "content": "The hiker stopped, sat down, and applied the method they had practised: compass, contour memory, slow bearing. Nothing happened fast.",
                "v": 0.1,
                "a": 0.65,
                "meta": {"phase": "method", "actor": "hiker"},
            },
            {
                "id": "stream_found",
                "content": "Fifteen minutes later a stream: downhill from here. The decision was straightforward once the reference point existed.",
                "v": 0.5,
                "a": 0.6,
                "meta": {"phase": "orientation", "actor": "hiker"},
            },
            {
                "id": "path_regained",
                "content": "The marked path was regained forty minutes after losing it; another hiker passed going the opposite direction without noticing anything unusual.",
                "v": 0.65,
                "a": 0.55,
                "meta": {"phase": "recovery", "actor": "hiker"},
            },
            {
                "id": "summit_abandoned",
                "content": "The hiker decided not to continue to the summit and descended; the decision felt like good judgement rather than failure.",
                "v": 0.55,
                "a": 0.45,
                "meta": {"phase": "decision", "actor": "hiker"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the acute disorientation peak — twenty minutes walking the wrong direction with wrong-feeling ground?",
                "exp": ["wrong_direction_taken"],
                "state": {"valence": -0.7, "arousal": 0.8},
            },
            "momentum_alignment": {
                "q": "Which memory marks the methodical momentum reversal — sitting down and applying compass-and-contour method slowly?",
                "exp": ["stop_and_methodical"],
                "state": {"valence": 0.05, "arousal": 0.6},
            },
            "same_topic_distractor": {
                "q": "Which post-recovery memory was specifically the summit abandonment as good judgement — not the path-regained moment?",
                "exp": ["summit_abandoned"],
                "state": {"valence": 0.5, "arousal": 0.4},
            },
            "semantic_confound": {
                "q": "Which memory captures the rational orientation milestone — finding the stream and making a clear decision from it — not the disorienting wrong-direction walk?",
                "exp": ["stream_found"],
                "state": {"valence": 0.45, "arousal": 0.55},
            },
        },
    },
    {
        "id": "s110_estranged_sibling_reunion",
        "desc": "Long-estranged siblings meet again after a decade and begin a cautious reconciliation.",
        "events": [
            {
                "id": "initial_contact",
                "content": "The message came through a mutual cousin: 'Would you be open to meeting?' The recipient did not reply for four days.",
                "v": -0.2,
                "a": 0.7,
                "meta": {"phase": "contact", "actor": "sibling_a"},
            },
            {
                "id": "first_meeting_arranged",
                "content": "They agreed to meet at a neutral cafe; the agreeing itself took three more messages and felt disproportionately difficult.",
                "v": -0.1,
                "a": 0.65,
                "meta": {"phase": "arrangement", "actor": "siblings"},
            },
            {
                "id": "two_hour_cafe_meeting",
                "content": "The meeting lasted two hours; neither mentioned the specific incident that had caused the estrangement.",
                "v": 0.2,
                "a": 0.6,
                "meta": {"phase": "meeting", "actor": "siblings"},
            },
            {
                "id": "incident_named",
                "content": "In the second meeting, one of them named the incident directly; the other said they had been waiting for that sentence for ten years.",
                "v": -0.4,
                "a": 0.75,
                "meta": {"phase": "confrontation", "actor": "siblings"},
            },
            {
                "id": "monthly_calls_established",
                "content": "Monthly calls were established, tentatively; the first few were awkward and brief and improving.",
                "v": 0.45,
                "a": 0.5,
                "meta": {"phase": "routine", "actor": "siblings"},
            },
            {
                "id": "family_event_attended_together",
                "content": "They attended a family event together for the first time in a decade; nobody made it into a story and they were grateful for that.",
                "v": 0.65,
                "a": 0.55,
                "meta": {"phase": "normalisation", "actor": "siblings"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the charged emotional peak — naming the incident and hearing 'I've been waiting for that sentence for ten years'?",
                "exp": ["incident_named"],
                "state": {"valence": -0.35, "arousal": 0.7},
            },
            "momentum_alignment": {
                "q": "Which memory marks the momentum shift toward sustained reconnection — the tentative monthly calls established and improving?",
                "exp": ["monthly_calls_established"],
                "state": {"valence": 0.4, "arousal": 0.45},
            },
            "same_topic_distractor": {
                "q": "Which post-reconciliation memory was specifically attending the family event together — not the monthly calls being established?",
                "exp": ["family_event_attended_together"],
                "state": {"valence": 0.6, "arousal": 0.5},
            },
            "semantic_confound": {
                "q": "Which memory captures the awkward pragmatics of neutral-ground arrangement — not the emotionally loaded incident-naming in the second meeting?",
                "exp": ["first_meeting_arranged"],
                "state": {"valence": -0.05, "arousal": 0.6},
            },
        },
    },
    # ------------------------------------------------------------------
    # COMBO 4: affective_arc + momentum_alignment + recency_confound + semantic_confound
    # ------------------------------------------------------------------
    {
        "id": "s111_composers_premiere",
        "desc": "A composer's first orchestral premiere swings from stage fright to exhilaration.",
        "events": [
            {
                "id": "score_completed",
                "content": "The score was completed at 2 a.m. three weeks before the premiere; the composer printed it and left it on the kitchen table without looking at it again.",
                "v": 0.5,
                "a": 0.6,
                "meta": {"phase": "completion", "actor": "composer"},
            },
            {
                "id": "rehearsal_shock",
                "content": "The first rehearsal: the piece sounded nothing like the composer's internal image, which was both devastating and interesting.",
                "v": -0.5,
                "a": 0.75,
                "meta": {"phase": "rehearsal", "actor": "composer"},
            },
            {
                "id": "conductor_adjustment",
                "content": "The conductor suggested a tempo adjustment in the third movement; the composer resisted and then heard why and agreed.",
                "v": 0.2,
                "a": 0.55,
                "meta": {"phase": "revision", "actor": "conductor"},
            },
            {
                "id": "premiere_night_wings",
                "content": "Standing in the wings during the premiere the composer felt a physical tremor that was not fear and not excitement but something between them.",
                "v": 0.3,
                "a": 0.9,
                "meta": {"phase": "premiere", "actor": "composer"},
            },
            {
                "id": "final_chord_applause",
                "content": "The final chord held and the applause began before the conductor's hands were down; the composer was walked onto the stage.",
                "v": 0.9,
                "a": 0.9,
                "meta": {"phase": "reception", "actor": "audience"},
            },
            {
                "id": "press_review",
                "content": "The review called the third movement 'the structural revelation of the evening'; the composer sent it to the conductor without comment.",
                "v": 0.8,
                "a": 0.65,
                "meta": {"phase": "recognition", "actor": "press"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the emotional peak — the applause beginning before the conductor's hands were down and the walk onto stage?",
                "exp": ["final_chord_applause"],
                "state": {"valence": 0.85, "arousal": 0.85},
            },
            "momentum_alignment": {
                "q": "Which memory marks the pivotal moment the conductor's tempo suggestion changed the piece — resistance yielding to understanding?",
                "exp": ["conductor_adjustment"],
                "state": {"valence": 0.15, "arousal": 0.5},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the score completion and leaving it on the kitchen table — before any rehearsal began?",
                "exp": ["score_completed"],
                "state": {"valence": 0.45, "arousal": 0.55},
            },
            "semantic_confound": {
                "q": "Which memory captures the charged physical state in the wings — not the exhilarating public reception of the final chord?",
                "exp": ["premiere_night_wings"],
                "state": {"valence": 0.25, "arousal": 0.85},
            },
        },
    },
    {
        "id": "s112_algorithm_breakthrough",
        "desc": "A programmer solves a long-blocked algorithmic problem in a sudden insight.",
        "events": [
            {
                "id": "problem_first_defined",
                "content": "The problem had been on the whiteboard for six weeks; everyone on the team had tried it once and moved on.",
                "v": -0.3,
                "a": 0.5,
                "meta": {"phase": "blockage", "actor": "team"},
            },
            {
                "id": "late_night_attempt",
                "content": "The programmer worked alone after the office emptied; the same dead ends, and then a slightly different angle at midnight.",
                "v": -0.4,
                "a": 0.65,
                "meta": {"phase": "attempt", "actor": "programmer"},
            },
            {
                "id": "insight_moment",
                "content": "The insight arrived not at the keyboard but at the water cooler; the programmer walked back at speed and wrote for forty minutes without stopping.",
                "v": 0.85,
                "a": 0.9,
                "meta": {"phase": "insight", "actor": "programmer"},
            },
            {
                "id": "tests_passing",
                "content": "All tests passed on the third run; the programmer sent a single message to the team channel at 1:47 a.m. that said only 'done'.",
                "v": 0.9,
                "a": 0.8,
                "meta": {"phase": "confirmation", "actor": "programmer"},
            },
            {
                "id": "team_code_review",
                "content": "The morning code review: the team asked questions for ninety minutes and then the tech lead said 'this is elegant'.",
                "v": 0.75,
                "a": 0.65,
                "meta": {"phase": "review", "actor": "team"},
            },
            {
                "id": "performance_metrics",
                "content": "Production metrics showed the solution was forty percent faster than the previous best; the product manager mentioned it in the all-hands.",
                "v": 0.8,
                "a": 0.7,
                "meta": {"phase": "impact", "actor": "metrics"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the purest insight moment — the water-cooler revelation and forty uninterrupted minutes of writing?",
                "exp": ["insight_moment"],
                "state": {"valence": 0.8, "arousal": 0.85},
            },
            "momentum_alignment": {
                "q": "Which memory marks the momentum shift from repeated failure to the first different angle — the midnight late-night attempt that opened a new direction?",
                "exp": ["late_night_attempt"],
                "state": {"valence": -0.35, "arousal": 0.6},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the six-week whiteboard blockage — before the programmer's solo late-night attempt?",
                "exp": ["problem_first_defined"],
                "state": {"valence": -0.25, "arousal": 0.45},
            },
            "semantic_confound": {
                "q": "Which memory captures public institutional confirmation — production metrics shared in the all-hands — not the private moment of all tests passing?",
                "exp": ["performance_metrics"],
                "state": {"valence": 0.75, "arousal": 0.65},
            },
        },
    },
    {
        "id": "s113_coach_fired_rehired",
        "desc": "A sports coach is fired mid-season, processes the loss, and is rehired a year later.",
        "events": [
            {
                "id": "termination_call",
                "content": "The club chairman called on a Wednesday morning; the coach had six years with the club and the call lasted nine minutes.",
                "v": -0.9,
                "a": 0.9,
                "meta": {"phase": "dismissal", "actor": "chairman"},
            },
            {
                "id": "press_conference_dignity",
                "content": "The press conference to announce the dismissal: the coach thanked the players and said nothing that required explanation later.",
                "v": -0.3,
                "a": 0.65,
                "meta": {"phase": "public", "actor": "coach"},
            },
            {
                "id": "months_unemployed",
                "content": "Eight months without an offer; the coach attended games as a spectator and found it harder than expected.",
                "v": -0.65,
                "a": 0.55,
                "meta": {"phase": "absence", "actor": "coach"},
            },
            {
                "id": "club_call_returns",
                "content": "The same club called eighteen months later; a new chairman, the team struggling, the same training ground.",
                "v": 0.5,
                "a": 0.75,
                "meta": {"phase": "return_offer", "actor": "chairman"},
            },
            {
                "id": "acceptance_conditions",
                "content": "The coach accepted with three non-negotiable conditions; all three were agreed by the end of the call.",
                "v": 0.65,
                "a": 0.65,
                "meta": {"phase": "acceptance", "actor": "coach"},
            },
            {
                "id": "first_training_session_back",
                "content": "The first training session back: players who had been there before and players who had not; the coach called them all by name within the hour.",
                "v": 0.8,
                "a": 0.75,
                "meta": {"phase": "return", "actor": "coach"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the acute professional loss — the nine-minute termination call after six years with the club?",
                "exp": ["termination_call"],
                "state": {"valence": -0.85, "arousal": 0.85},
            },
            "momentum_alignment": {
                "q": "Which memory marks the momentum reversal — the club's return call eighteen months later with a new chairman?",
                "exp": ["club_call_returns"],
                "state": {"valence": 0.45, "arousal": 0.7},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the dignified press conference — before the months of unemployment began?",
                "exp": ["press_conference_dignity"],
                "state": {"valence": -0.25, "arousal": 0.6},
            },
            "semantic_confound": {
                "q": "Which memory captures the productive return of agency — negotiating three non-negotiable conditions successfully — not the emotional first training session?",
                "exp": ["acceptance_conditions"],
                "state": {"valence": 0.6, "arousal": 0.6},
            },
        },
    },
    {
        "id": "s114_first_serious_sale",
        "desc": "A painter sells their first serious work and navigates the aftermath.",
        "events": [
            {
                "id": "work_submitted_to_fair",
                "content": "The large canvas was submitted to the regional art fair with a price the painter could not justify rationally but could not lower.",
                "v": 0.4,
                "a": 0.65,
                "meta": {"phase": "submission", "actor": "painter"},
            },
            {
                "id": "opening_day_no_interest",
                "content": "Opening day: the work attracted a few long looks but no red dot; the painter left early and did not check the website until the following morning.",
                "v": -0.55,
                "a": 0.6,
                "meta": {"phase": "wait", "actor": "painter"},
            },
            {
                "id": "sale_confirmed",
                "content": "On the third day a red dot appeared; the fair organiser called to say a collector had purchased it without negotiation.",
                "v": 0.85,
                "a": 0.85,
                "meta": {"phase": "sale", "actor": "collector"},
            },
            {
                "id": "painting_collected",
                "content": "Watching the work being collected: the painter had not expected to feel the loss of it.",
                "v": -0.2,
                "a": 0.6,
                "meta": {"phase": "separation", "actor": "painter"},
            },
            {
                "id": "collector_message",
                "content": "Two weeks later the collector sent a photo of the work installed in their home; it looked different in that context and better.",
                "v": 0.7,
                "a": 0.6,
                "meta": {"phase": "placement", "actor": "collector"},
            },
            {
                "id": "second_work_begun",
                "content": "The painter began a second large-format work the day after the collector's photo; the canvas was already primed.",
                "v": 0.75,
                "a": 0.7,
                "meta": {"phase": "continuation", "actor": "painter"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the peak joy — the fair organiser calling to confirm the sale with no negotiation on the third day?",
                "exp": ["sale_confirmed"],
                "state": {"valence": 0.8, "arousal": 0.8},
            },
            "momentum_alignment": {
                "q": "Which memory marks the renewal momentum — beginning the second large-format work the day after the collector's photo arrived?",
                "exp": ["second_work_begun"],
                "state": {"valence": 0.7, "arousal": 0.65},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded submitting the canvas to the fair — before opening day or the sale?",
                "exp": ["work_submitted_to_fair"],
                "state": {"valence": 0.35, "arousal": 0.6},
            },
            "semantic_confound": {
                "q": "Which memory captures complex ambivalence about the sold work — the unexpected loss of watching it collected — not the positive news of the collector's photo?",
                "exp": ["painting_collected"],
                "state": {"valence": -0.15, "arousal": 0.55},
            },
        },
    },
    {
        "id": "s115_geological_discovery",
        "desc": "A geologist discovers a rare formation during fieldwork and navigates its verification.",
        "events": [
            {
                "id": "fieldwork_day_routine",
                "content": "The third day of routine mapping: heat, flies, repetitive outcrop description. Nothing suggested the afternoon would be different.",
                "v": -0.1,
                "a": 0.45,
                "meta": {"phase": "routine", "actor": "geologist"},
            },
            {
                "id": "anomalous_outcrop",
                "content": "An outcrop twenty metres off the survey line did not fit the regional stratigraphy; the geologist stood in front of it for a long time before taking the first sample.",
                "v": 0.6,
                "a": 0.75,
                "meta": {"phase": "discovery", "actor": "geologist"},
            },
            {
                "id": "lab_results_pending",
                "content": "Three weeks of lab analysis: the geologist did not discuss the samples with colleagues until the geochemistry returned.",
                "v": -0.2,
                "a": 0.6,
                "meta": {"phase": "analysis", "actor": "geologist"},
            },
            {
                "id": "results_confirm_rarity",
                "content": "The results confirmed a formation type not previously documented in the region; the geologist emailed the department head in three sentences.",
                "v": 0.85,
                "a": 0.8,
                "meta": {"phase": "confirmation", "actor": "lab"},
            },
            {
                "id": "peer_review_submitted",
                "content": "The paper was submitted; the geologist expected a year of review and was prepared for it.",
                "v": 0.5,
                "a": 0.55,
                "meta": {"phase": "submission", "actor": "geologist"},
            },
            {
                "id": "field_revisit",
                "content": "A revisit to the outcrop with two colleagues: one immediately saw what the geologist had seen; one took twenty minutes.",
                "v": 0.7,
                "a": 0.65,
                "meta": {"phase": "validation", "actor": "team"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the scientific peak — lab results confirming a formation type not previously documented in the region?",
                "exp": ["results_confirm_rarity"],
                "state": {"valence": 0.8, "arousal": 0.75},
            },
            "momentum_alignment": {
                "q": "Which memory marks the transition from routine to charged attention — the anomalous outcrop off the survey line that didn't fit regional stratigraphy?",
                "exp": ["anomalous_outcrop"],
                "state": {"valence": 0.55, "arousal": 0.7},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the routine mapping day — before the anomalous outcrop was found?",
                "exp": ["fieldwork_day_routine"],
                "state": {"valence": -0.05, "arousal": 0.4},
            },
            "semantic_confound": {
                "q": "Which memory captures collegial field validation — one colleague immediately seeing what the geologist saw — not the solitary lab confirmation?",
                "exp": ["field_revisit"],
                "state": {"valence": 0.65, "arousal": 0.6},
            },
        },
    },
    {
        "id": "s116_archive_digitisation",
        "desc": "A librarian leads a fragile archive digitisation project through technical crisis to completion.",
        "events": [
            {
                "id": "project_approved",
                "content": "The digitisation grant was approved; forty years of uncatalogued material, deteriorating, would be scanned and indexed.",
                "v": 0.65,
                "a": 0.65,
                "meta": {"phase": "approval", "actor": "institution"},
            },
            {
                "id": "scanner_failure",
                "content": "Three weeks in, the flatbed scanner failed on a batch of the most fragile glass plates; two plates were cracked in removal.",
                "v": -0.8,
                "a": 0.85,
                "meta": {"phase": "crisis", "actor": "equipment"},
            },
            {
                "id": "conservation_emergency",
                "content": "A conservator was brought in on emergency; the remaining glass plates were stabilised and the scanning workflow redesigned.",
                "v": 0.1,
                "a": 0.7,
                "meta": {"phase": "response", "actor": "conservator"},
            },
            {
                "id": "volunteer_surge",
                "content": "A call for volunteers produced forty-two respondents; the indexing backlog was cleared in two weekends.",
                "v": 0.6,
                "a": 0.65,
                "meta": {"phase": "support", "actor": "volunteers"},
            },
            {
                "id": "archive_goes_live",
                "content": "The digital archive went live; in the first week, a researcher found a photograph of their own grandmother in a 1947 batch.",
                "v": 0.85,
                "a": 0.75,
                "meta": {"phase": "launch", "actor": "archive"},
            },
            {
                "id": "regional_award",
                "content": "The project received a regional preservation award; the librarian mentioned the two cracked plates in the acceptance speech.",
                "v": 0.8,
                "a": 0.65,
                "meta": {"phase": "recognition", "actor": "institution"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the project's emotional peak — the archive going live and a researcher finding their grandmother in a 1947 photograph?",
                "exp": ["archive_goes_live"],
                "state": {"valence": 0.8, "arousal": 0.7},
            },
            "momentum_alignment": {
                "q": "Which memory marks the upward momentum shift — forty-two volunteers clearing the indexing backlog in two weekends?",
                "exp": ["volunteer_surge"],
                "state": {"valence": 0.55, "arousal": 0.6},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the grant approval and scope of the project — before the scanner failure occurred?",
                "exp": ["project_approved"],
                "state": {"valence": 0.6, "arousal": 0.6},
            },
            "semantic_confound": {
                "q": "Which memory captures the institutional recognition mentioning the crisis — not the human discovery moment of the grandmother photograph?",
                "exp": ["regional_award"],
                "state": {"valence": 0.75, "arousal": 0.6},
            },
        },
    },
    {
        "id": "s117_certification_retry",
        "desc": "A nurse fails a specialist certification exam, processes it, and passes on retry.",
        "events": [
            {
                "id": "exam_day_one",
                "content": "The certification exam required six months of preparation; the nurse arrived early, settled, and knew within the first thirty minutes that it was not going well.",
                "v": -0.4,
                "a": 0.75,
                "meta": {"phase": "exam", "actor": "nurse"},
            },
            {
                "id": "result_fail",
                "content": "The result arrived by post ten days later; three points below the pass threshold.",
                "v": -0.8,
                "a": 0.8,
                "meta": {"phase": "result", "actor": "board"},
            },
            {
                "id": "supervisor_support",
                "content": "The supervisor said: 'This happens to exceptional nurses. You're not done.' The nurse did not fully believe it and was grateful for it anyway.",
                "v": 0.2,
                "a": 0.5,
                "meta": {"phase": "support", "actor": "supervisor"},
            },
            {
                "id": "targeted_revision",
                "content": "The eight-week revision period targeted specifically the three topics identified in the failure feedback; nothing else.",
                "v": 0.3,
                "a": 0.6,
                "meta": {"phase": "revision", "actor": "nurse"},
            },
            {
                "id": "exam_day_two",
                "content": "The second exam: the nurse answered the last question with twelve minutes remaining and used eleven of them to check.",
                "v": 0.4,
                "a": 0.7,
                "meta": {"phase": "retry", "actor": "nurse"},
            },
            {
                "id": "result_pass",
                "content": "The pass result arrived by email this time; the nurse was at work when it arrived and read it in the medication room.",
                "v": 0.9,
                "a": 0.8,
                "meta": {"phase": "success", "actor": "board"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the acute disappointment — the result arriving three points below the threshold after six months of preparation?",
                "exp": ["result_fail"],
                "state": {"valence": -0.75, "arousal": 0.75},
            },
            "momentum_alignment": {
                "q": "Which memory marks the upward momentum shift — targeted revision of only the failure-feedback topics with clear focus?",
                "exp": ["targeted_revision"],
                "state": {"valence": 0.25, "arousal": 0.55},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the first exam day, sensing within thirty minutes it wasn't going well — before the result arrived?",
                "exp": ["exam_day_one"],
                "state": {"valence": -0.35, "arousal": 0.7},
            },
            "semantic_confound": {
                "q": "Which memory captures peer institutional solidarity — not the personal pass result read alone in the medication room?",
                "exp": ["supervisor_support"],
                "state": {"valence": 0.15, "arousal": 0.45},
            },
        },
    },
    {
        "id": "s118_beehive_collapse",
        "desc": "An urban farmer's beehive collapses due to disease and they rebuild a new colony.",
        "events": [
            {
                "id": "hive_inspection",
                "content": "The weekly inspection found something wrong in the brood pattern; the farmer took photographs and sent them to a local beekeeper that evening.",
                "v": -0.4,
                "a": 0.65,
                "meta": {"phase": "discovery", "actor": "farmer"},
            },
            {
                "id": "disease_confirmed",
                "content": "The local beekeeper confirmed American foulbrood; the hive would need to be destroyed under regulation.",
                "v": -0.85,
                "a": 0.85,
                "meta": {"phase": "diagnosis", "actor": "beekeeper"},
            },
            {
                "id": "hive_destroyed",
                "content": "The hive was destroyed according to protocol; the farmer burned it in the allotment corner and stood watching until the smoke stopped.",
                "v": -0.75,
                "a": 0.6,
                "meta": {"phase": "destruction", "actor": "farmer"},
            },
            {
                "id": "beekeeping_association",
                "content": "The local beekeeping association offered a nucleus colony at reduced cost; three members came to help set up the new hive.",
                "v": 0.5,
                "a": 0.6,
                "meta": {"phase": "support", "actor": "association"},
            },
            {
                "id": "new_colony_active",
                "content": "The new colony established successfully within six weeks; the first inspection showed a laying queen and a healthy brood pattern.",
                "v": 0.7,
                "a": 0.65,
                "meta": {"phase": "recovery", "actor": "bees"},
            },
            {
                "id": "first_harvest_new_colony",
                "content": "The first small honey harvest from the new colony; the farmer put a jar aside labelled 'season two'.",
                "v": 0.8,
                "a": 0.6,
                "meta": {"phase": "harvest", "actor": "farmer"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the grief of necessary destruction — burning the hive and watching the smoke until it stopped?",
                "exp": ["hive_destroyed"],
                "state": {"valence": -0.7, "arousal": 0.55},
            },
            "momentum_alignment": {
                "q": "Which memory marks the upward momentum shift — the association offering a nucleus colony and three members arriving to help?",
                "exp": ["beekeeping_association"],
                "state": {"valence": 0.45, "arousal": 0.55},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the initial inspection and sending photographs — before the disease was confirmed?",
                "exp": ["hive_inspection"],
                "state": {"valence": -0.35, "arousal": 0.6},
            },
            "semantic_confound": {
                "q": "Which memory captures productive biological recovery — the healthy brood pattern at first new-hive inspection — not the first harvest labelled 'season two'?",
                "exp": ["new_colony_active"],
                "state": {"valence": 0.65, "arousal": 0.6},
            },
        },
    },
    {
        "id": "s119_age_group_record",
        "desc": "A masters-category marathon runner sets an unexpected age-group record.",
        "events": [
            {
                "id": "race_entry",
                "content": "The race entry was filed as a training run; the runner's target time was conservative and no one had been told about it.",
                "v": 0.35,
                "a": 0.55,
                "meta": {"phase": "entry", "actor": "runner"},
            },
            {
                "id": "halfway_split_surprising",
                "content": "The halfway split was faster than planned; the runner recalculated and decided to hold pace rather than pull back.",
                "v": 0.5,
                "a": 0.7,
                "meta": {"phase": "midrace", "actor": "runner"},
            },
            {
                "id": "kilometre_thirty_wall",
                "content": "At kilometre thirty the runner hit the wall in the textbook way; four minutes of running on will and not much else.",
                "v": -0.6,
                "a": 0.8,
                "meta": {"phase": "crisis", "actor": "runner"},
            },
            {
                "id": "finish_line_crossed",
                "content": "The finish line: the clock showed a time the runner had not believed was possible.",
                "v": 0.85,
                "a": 0.85,
                "meta": {"phase": "finish", "actor": "runner"},
            },
            {
                "id": "record_confirmed",
                "content": "The race official told the runner they had set a national age-group record; the runner sat on the kerb and drank water for several minutes.",
                "v": 0.9,
                "a": 0.8,
                "meta": {"phase": "record", "actor": "official"},
            },
            {
                "id": "club_recognition",
                "content": "The running club posted the result; messages arrived from people the runner had trained with for years and some they had never met.",
                "v": 0.75,
                "a": 0.65,
                "meta": {"phase": "recognition", "actor": "club"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the disbelief-peak of the national age-group record confirmation — sitting on the kerb drinking water?",
                "exp": ["record_confirmed"],
                "state": {"valence": 0.85, "arousal": 0.75},
            },
            "momentum_alignment": {
                "q": "Which memory marks the race's downward momentum trough — the kilometre-thirty wall with four minutes of running on will?",
                "exp": ["kilometre_thirty_wall"],
                "state": {"valence": -0.55, "arousal": 0.75},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the low-key race entry as a training run — before any race dynamics emerged?",
                "exp": ["race_entry"],
                "state": {"valence": 0.3, "arousal": 0.5},
            },
            "semantic_confound": {
                "q": "Which memory captures the community acknowledgement of the achievement — not the official race confirmation of the record?",
                "exp": ["club_recognition"],
                "state": {"valence": 0.7, "arousal": 0.6},
            },
        },
    },
    {
        "id": "s120_therapist_imposter_syndrome",
        "desc": "A therapist confronts imposter syndrome after a client's challenging question.",
        "events": [
            {
                "id": "client_question",
                "content": "A client asked mid-session: 'Do you ever feel like you don't actually know what you're doing?' The therapist said yes before thinking.",
                "v": -0.3,
                "a": 0.7,
                "meta": {"phase": "trigger", "actor": "client"},
            },
            {
                "id": "supervision_disclosure",
                "content": "The therapist brought the exchange to supervision the following week; the supervisor said 'that was the best thing you could have said to that client'.",
                "v": 0.4,
                "a": 0.6,
                "meta": {"phase": "processing", "actor": "supervisor"},
            },
            {
                "id": "self_doubt_weeks",
                "content": "Despite the supervisor's response, two weeks of background doubt about competence; checking session notes more than usual.",
                "v": -0.5,
                "a": 0.55,
                "meta": {"phase": "doubt", "actor": "therapist"},
            },
            {
                "id": "peer_group_discussion",
                "content": "A peer group discussion surfaced that three other therapists had had identical doubt cycles that year; the specificity was reassuring.",
                "v": 0.4,
                "a": 0.5,
                "meta": {"phase": "normalisation", "actor": "peers"},
            },
            {
                "id": "client_breakthrough_same",
                "content": "The same client, six weeks later, said the therapist's honest response had been the turning point in their willingness to trust the process.",
                "v": 0.8,
                "a": 0.7,
                "meta": {"phase": "validation", "actor": "client"},
            },
            {
                "id": "written_reflection",
                "content": "The therapist wrote a reflective journal entry about the episode; it was useful and they did not re-read it.",
                "v": 0.55,
                "a": 0.45,
                "meta": {"phase": "integration", "actor": "therapist"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the highest emotional validation — the same client saying the honest answer had been the turning point in their trust?",
                "exp": ["client_breakthrough_same"],
                "state": {"valence": 0.75, "arousal": 0.65},
            },
            "momentum_alignment": {
                "q": "Which memory marks the shift from doubt to normalisation — the peer group revealing three others had identical doubt cycles?",
                "exp": ["peer_group_discussion"],
                "state": {"valence": 0.35, "arousal": 0.45},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the unplanned 'yes' answer to the client's question — before any supervision or doubt processing?",
                "exp": ["client_question"],
                "state": {"valence": -0.25, "arousal": 0.65},
            },
            "semantic_confound": {
                "q": "Which memory captures the productive outcome of professional support — not the private doubt weeks of checking session notes?",
                "exp": ["supervision_disclosure"],
                "state": {"valence": 0.35, "arousal": 0.55},
            },
        },
    },
    {
        "id": "s121_bridge_inspection_scare",
        "desc": "A structural engineer finds a critical defect during a bridge inspection and manages the crisis.",
        "events": [
            {
                "id": "routine_inspection_start",
                "content": "The annual bridge inspection was the kind that had been uneventful twelve times before; the engineer set up the equipment and began the standard sequence.",
                "v": 0.1,
                "a": 0.45,
                "meta": {"phase": "routine", "actor": "engineer"},
            },
            {
                "id": "crack_discovered",
                "content": "A hairline crack in the primary tension chord, previously unrecorded, in a location that required immediate assessment.",
                "v": -0.8,
                "a": 0.9,
                "meta": {"phase": "discovery", "actor": "engineer"},
            },
            {
                "id": "load_restriction_imposed",
                "content": "The engineer called the authority within the hour; a temporary load restriction was imposed by 3 p.m. the same day.",
                "v": -0.2,
                "a": 0.7,
                "meta": {"phase": "response", "actor": "authority"},
            },
            {
                "id": "structural_analysis_completed",
                "content": "Detailed structural analysis over three days confirmed the crack was propagating slowly; repair was urgent but not emergency.",
                "v": 0.3,
                "a": 0.6,
                "meta": {"phase": "analysis", "actor": "engineer"},
            },
            {
                "id": "repair_completed",
                "content": "The repair was completed within two weeks; the load restriction was lifted and the bridge returned to normal service.",
                "v": 0.65,
                "a": 0.55,
                "meta": {"phase": "repair", "actor": "contractors"},
            },
            {
                "id": "inspection_protocol_updated",
                "content": "The engineer wrote a revised inspection protocol for that crack geometry type; it was adopted by the regional authority for all similar bridges.",
                "v": 0.75,
                "a": 0.6,
                "meta": {"phase": "systemic", "actor": "engineer"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the acute shock of the critical discovery — an unrecorded hairline crack in the primary tension chord?",
                "exp": ["crack_discovered"],
                "state": {"valence": -0.75, "arousal": 0.85},
            },
            "momentum_alignment": {
                "q": "Which memory marks the calming of crisis momentum — the structural analysis confirming slow propagation and urgent-but-not-emergency status?",
                "exp": ["structural_analysis_completed"],
                "state": {"valence": 0.25, "arousal": 0.55},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the routine inspection start — before the crack was found?",
                "exp": ["routine_inspection_start"],
                "state": {"valence": 0.05, "arousal": 0.4},
            },
            "semantic_confound": {
                "q": "Which memory captures systemic professional impact beyond the specific bridge — the revised protocol adopted regionally — not the immediate repair completion?",
                "exp": ["inspection_protocol_updated"],
                "state": {"valence": 0.7, "arousal": 0.55},
            },
        },
    },
    {
        "id": "s122_wildlife_photograph",
        "desc": "A wildlife photographer captures a once-in-a-decade image after years of waiting.",
        "events": [
            {
                "id": "hide_set_up",
                "content": "The hide had been in position for three days; the photographer had done this eleven times in six years for the same species and the same result.",
                "v": -0.1,
                "a": 0.45,
                "meta": {"phase": "wait", "actor": "photographer"},
            },
            {
                "id": "species_appears",
                "content": "On the fourth morning, in the first light, the animal emerged from the treeline in a posture that had not been photographed before.",
                "v": 0.75,
                "a": 0.85,
                "meta": {"phase": "appearance", "actor": "subject"},
            },
            {
                "id": "shutter_sequence",
                "content": "Forty-three frames in eleven seconds; the photographer did not breathe deliberately for any of them.",
                "v": 0.6,
                "a": 0.9,
                "meta": {"phase": "capture", "actor": "photographer"},
            },
            {
                "id": "review_at_camp",
                "content": "Reviewing the frames at camp: frame twenty-seven was the one. The photographer sat quietly for several minutes.",
                "v": 0.85,
                "a": 0.75,
                "meta": {"phase": "review", "actor": "photographer"},
            },
            {
                "id": "image_submitted",
                "content": "The image was submitted to the nature photography award; the submission note took longer to write than the photograph had taken.",
                "v": 0.5,
                "a": 0.6,
                "meta": {"phase": "submission", "actor": "photographer"},
            },
            {
                "id": "award_winner",
                "content": "The image won; the citation mentioned 'the geometry of the moment and the discipline that made it possible'.",
                "v": 0.9,
                "a": 0.8,
                "meta": {"phase": "award", "actor": "judges"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the peak realisation — reviewing the frames and knowing frame twenty-seven was the one?",
                "exp": ["review_at_camp"],
                "state": {"valence": 0.8, "arousal": 0.7},
            },
            "momentum_alignment": {
                "q": "Which memory marks the momentum shift from years of waiting to charged action — the animal emerging in an unphotographed posture?",
                "exp": ["species_appears"],
                "state": {"valence": 0.7, "arousal": 0.8},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the three-day hide set-up — before the species appeared on day four?",
                "exp": ["hide_set_up"],
                "state": {"valence": -0.05, "arousal": 0.4},
            },
            "semantic_confound": {
                "q": "Which memory captures the intense physical capture moment — forty-three frames without deliberate breath — not the quiet recognition at camp?",
                "exp": ["shutter_sequence"],
                "state": {"valence": 0.55, "arousal": 0.85},
            },
        },
    },
    {
        "id": "s123_emergency_volunteer_pivot",
        "desc": "A volunteer coordinator pivots an annual event to emergency disaster relief.",
        "events": [
            {
                "id": "annual_event_planned",
                "content": "Six months of logistics for the annual community festival: eighty volunteers, twelve stations, catering for six hundred.",
                "v": 0.6,
                "a": 0.6,
                "meta": {"phase": "planning", "actor": "coordinator"},
            },
            {
                "id": "flood_announcement",
                "content": "Two weeks before the event a regional flood displaced eight hundred residents; the coordinator called an emergency meeting within four hours.",
                "v": -0.6,
                "a": 0.85,
                "meta": {"phase": "crisis", "actor": "weather"},
            },
            {
                "id": "pivot_decision",
                "content": "The meeting decided unanimously to convert the event infrastructure to relief operations; every logistics plan was reusable.",
                "v": 0.4,
                "a": 0.75,
                "meta": {"phase": "pivot", "actor": "team"},
            },
            {
                "id": "relief_operation_day_one",
                "content": "On what would have been festival day, seventy-three volunteers distributed meals, clothing, and supplies to four hundred and fifty displaced residents.",
                "v": 0.7,
                "a": 0.8,
                "meta": {"phase": "operation", "actor": "volunteers"},
            },
            {
                "id": "coordinator_exhaustion",
                "content": "After three days of relief operations the coordinator slept for fourteen hours and woke feeling both emptied and right.",
                "v": 0.35,
                "a": 0.35,
                "meta": {"phase": "aftermath", "actor": "coordinator"},
            },
            {
                "id": "community_recognition",
                "content": "The local council recognised the volunteer organisation with a community resilience citation; the coordinator accepted it on behalf of the team.",
                "v": 0.75,
                "a": 0.65,
                "meta": {"phase": "recognition", "actor": "council"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the emotional complexity of exhaustion and rightness — sleeping fourteen hours and waking emptied but correct?",
                "exp": ["coordinator_exhaustion"],
                "state": {"valence": 0.3, "arousal": 0.3},
            },
            "momentum_alignment": {
                "q": "Which memory marks the decisive collective momentum shift — the unanimous meeting decision that every logistics plan was reusable for relief?",
                "exp": ["pivot_decision"],
                "state": {"valence": 0.35, "arousal": 0.7},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the six-month festival logistics planning — before the flood changed everything?",
                "exp": ["annual_event_planned"],
                "state": {"valence": 0.55, "arousal": 0.55},
            },
            "semantic_confound": {
                "q": "Which memory captures institutional civic acknowledgement — not the operational peak of distributing supplies to four hundred and fifty residents?",
                "exp": ["community_recognition"],
                "state": {"valence": 0.7, "arousal": 0.6},
            },
        },
    },
    {
        "id": "s124_national_baking_competition",
        "desc": "A baker enters a national competition, navigates the pressure, and places second.",
        "events": [
            {
                "id": "application_accepted",
                "content": "The application to the national competition was accepted; the baker told no one except their partner for three days.",
                "v": 0.65,
                "a": 0.7,
                "meta": {"phase": "acceptance", "actor": "baker"},
            },
            {
                "id": "signature_bake_failure",
                "content": "In the competition kitchen the signature bake overproofed by six minutes; the judges noted it on their pads before tasting.",
                "v": -0.7,
                "a": 0.8,
                "meta": {"phase": "failure", "actor": "baker"},
            },
            {
                "id": "technical_challenge_recovery",
                "content": "The technical challenge required a method the baker had practised only twice; they placed third and were not eliminated.",
                "v": 0.2,
                "a": 0.65,
                "meta": {"phase": "recovery", "actor": "baker"},
            },
            {
                "id": "showstopper_praised",
                "content": "The showstopper drew the longest judges' discussion of the day; the baker stood and waited without hearing the words properly.",
                "v": 0.75,
                "a": 0.85,
                "meta": {"phase": "reception", "actor": "judges"},
            },
            {
                "id": "second_place_announced",
                "content": "Second place was announced; the baker congratulated the winner and meant it.",
                "v": 0.6,
                "a": 0.7,
                "meta": {"phase": "result", "actor": "competition"},
            },
            {
                "id": "bakery_queue_next_day",
                "content": "The following day the bakery queue started forty minutes before opening; three people mentioned the competition by name.",
                "v": 0.85,
                "a": 0.75,
                "meta": {"phase": "aftermath", "actor": "customers"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the peak tension of uncertain reception — standing during the showstopper discussion not properly hearing the judges' words?",
                "exp": ["showstopper_praised"],
                "state": {"valence": 0.7, "arousal": 0.8},
            },
            "momentum_alignment": {
                "q": "Which memory marks the downward momentum of the signature bake failure — judges noting the overproofing before even tasting?",
                "exp": ["signature_bake_failure"],
                "state": {"valence": -0.65, "arousal": 0.75},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the acceptance and telling only the partner — before the competition began?",
                "exp": ["application_accepted"],
                "state": {"valence": 0.6, "arousal": 0.65},
            },
            "semantic_confound": {
                "q": "Which memory captures the local community reward of the competition — the bakery queue quoting the competition by name — not the second-place announcement itself?",
                "exp": ["bakery_queue_next_day"],
                "state": {"valence": 0.8, "arousal": 0.7},
            },
        },
    },
    {
        "id": "s125_language_teacher_breakthrough",
        "desc": "A language teacher witnesses a silent student speak fluently for the first time.",
        "events": [
            {
                "id": "student_enrolled_silent",
                "content": "The student had been enrolled for four months and had not produced a single spoken sentence; written work was flawless.",
                "v": -0.2,
                "a": 0.45,
                "meta": {"phase": "background", "actor": "student"},
            },
            {
                "id": "teacher_parent_meeting",
                "content": "A meeting with the student's parents: the silence was selective mutism connected to performance anxiety, not comprehension gap.",
                "v": -0.1,
                "a": 0.55,
                "meta": {"phase": "context", "actor": "teacher"},
            },
            {
                "id": "paired_activity_design",
                "content": "The teacher redesigned one weekly activity as a two-person spoken task with no audience; the student was paired with a trusted classmate.",
                "v": 0.3,
                "a": 0.55,
                "meta": {"phase": "adaptation", "actor": "teacher"},
            },
            {
                "id": "first_spoken_sentence",
                "content": "In week seven of the adapted activity the student spoke a full sentence in the target language; the teacher heard it from across the room and did not react visibly.",
                "v": 0.85,
                "a": 0.8,
                "meta": {"phase": "breakthrough", "actor": "student"},
            },
            {
                "id": "gradual_participation",
                "content": "Over the following three months the student began contributing in small-group discussions; never in whole-class contexts, which the teacher did not push.",
                "v": 0.65,
                "a": 0.55,
                "meta": {"phase": "progress", "actor": "student"},
            },
            {
                "id": "oral_exam_passed",
                "content": "The oral examination was passed at distinction level; the examiner noted fluency and hesitation that was 'thoughtful rather than uncertain'.",
                "v": 0.9,
                "a": 0.7,
                "meta": {"phase": "achievement", "actor": "student"},
            },
        ],
        "queries": {
            "affective_arc": {
                "q": "Which memory captures the teacher's restrained peak joy — hearing the first full spoken sentence and not reacting visibly?",
                "exp": ["first_spoken_sentence"],
                "state": {"valence": 0.8, "arousal": 0.75},
            },
            "momentum_alignment": {
                "q": "Which memory marks the structural pivot that made the breakthrough possible — redesigning the activity as a two-person spoken task with no audience?",
                "exp": ["paired_activity_design"],
                "state": {"valence": 0.25, "arousal": 0.5},
            },
            "recency_confound": {
                "q": "Which first-session memory recorded the four-month silence and flawless written work — before the parent meeting provided context?",
                "exp": ["student_enrolled_silent"],
                "state": {"valence": -0.15, "arousal": 0.4},
            },
            "semantic_confound": {
                "q": "Which memory captures the culminating public academic achievement — the distinction-level oral exam — not the private first-sentence breakthrough?",
                "exp": ["oral_exam_passed"],
                "state": {"valence": 0.85, "arousal": 0.65},
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def _build_dataset() -> dict:  # type: ignore[type-arg]
    repo_root = Path(__file__).parent.parent
    v2_path = repo_root / "benchmarks" / "datasets" / "realistic_recall_v2.json"
    with v2_path.open() as f:
        v2 = json.load(f)

    v2_scenarios: list[dict] = v2["scenarios"]  # type: ignore[assignment]

    new_scenarios: list[dict] = []  # type: ignore[type-arg]
    for idx, raw in enumerate(SCENARIOS):
        combo = COMBOS[idx // 15]
        all_events = [
            {
                "memory_id": e["id"],
                "content": e["content"],
                "valence": e["v"],
                "arousal": e["a"],
                "metadata": e["meta"],
            }
            for e in raw["events"]
        ]
        # Split 6 events into 2 sessions of 3, matching v2 schema
        session1_events = all_events[:3]
        session2_events = all_events[3:]
        queries = []
        for challenge_type in combo:
            q_spec = raw["queries"][challenge_type]
            queries.append(
                {
                    "query_id": f"{raw['id']}_{challenge_type}",
                    "query": q_spec["q"],
                    "expected_memory_ids": q_spec["exp"],
                    "challenge_type": challenge_type,
                    "state": q_spec["state"],
                }
            )
        new_scenarios.append(
            {
                "scenario_id": raw["id"],
                "description": raw["desc"],
                "sessions": [
                    {
                        "session_id": "session_1",
                        "description": f"Initial events in {raw['desc'].lower()}",
                        "events": session1_events,
                        "queries": [],
                    },
                    {
                        "session_id": "session_2",
                        "description": f"Follow-up events and queries in {raw['desc'].lower()}",
                        "events": session2_events,
                        "queries": queries,
                    },
                ],
            }
        )

    all_scenarios = v2_scenarios + new_scenarios
    return {
        "name": "realistic_recall_v3",
        "version": "3.0",
        "description": (
            "realistic_recall_v3: v2 (200 queries, 50 scenarios) extended with 75 new"
            " scenarios (300 queries) for a total of 500 queries across 125 scenarios."
            " Each scenario uses a 4-of-5 challenge-type combo (rotating across 5 combos)."
            " v3_noAF (realistic_recall_v3_noAF.json) is a separate affect-free dataset"
            " for Add. G (Hg1). All v3 scenarios include preset valence/arousal."
        ),
        "default_top_k": v2.get("default_top_k", 1),
        "scenarios": all_scenarios,
    }


if __name__ == "__main__":
    dataset = _build_dataset()
    out_path = (
        Path(__file__).parent.parent / "benchmarks" / "datasets" / "realistic_recall_v3.json"
    )
    with out_path.open("w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    total_queries = sum(
        len(sess["queries"]) for s in dataset["scenarios"] for sess in s["sessions"]
    )
    print(f"Written {out_path}")
    print(f"  scenarios: {len(dataset['scenarios'])}")
    print(f"  queries:   {total_queries}")
