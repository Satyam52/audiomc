# Methodology Examples

Each test is a spoken conversation. The model must answer **only the final user turn**. That reply is graded against rubric items (did it remember facts, follow instructions, honour corrections, stay consistent).

---

## One-Shot Flow

The model hears the **full conversation in one go** — every user turn as audio, every prior assistant turn as text — then replies to the last user turn.

**Conversation**

**User (turn 1, audio):** "I'm booking a train to Edinburgh. My sister Mei is travelling with me."

**Assistant (turn 1):** "When would you like to travel?"

**User (turn 2, audio):** "Tuesday morning. Mei is allergic to peanuts, so keep that in mind for any food suggestions."

**Assistant (turn 2):** "Noted. I'll look for morning trains on Tuesday."

**User (turn 3, audio):** "Actually, scratch that — we need Thursday morning, not Tuesday."

**Assistant (turn 3):** "Understood, Thursday morning it is."

**User (turn 4, audio):** "Great. Can you suggest a café near the station for when we arrive? Remember Mei's allergy."

**How the flow works**

All four user turns are played as audio. Turns 1–3 assistant replies are given as text. The model generates one reply to turn 4.

**Model reply (expected):** Suggest a café near Edinburgh station, respect Mei's peanut allergy, and treat Thursday (not Tuesday) as the travel day.

---

## Turnwise Notetaking Flow

The model **never sees the full conversation**. For each earlier turn:

1. **Qwen captions** the user audio alone (acoustic, paralinguistic, semantic transcript).
2. **Notetaker** updates notes from those captions plus the assistant's seed reply for that turn.

At the end, **Qwen** gets only the rendered notes and the final user audio.

Same conversation as above.

**Turn 1**

**User (audio):** "I'm booking a train to Edinburgh. My sister Mei is travelling with me."

→ Qwen caption (audio only):
- Acoustic: quiet indoor setting
- Paralinguistic: calm, informative tone
- Semantic: "I'm booking a train to Edinburgh. My sister Mei is travelling with me."

**Assistant (seed text):** "When would you like to travel?"

→ Notetaker updates notes from caption + assistant reply:
- Fact: booking a train to Edinburgh
- Fact: sister Mei is travelling with them
- Assistant state: asked about travel date

**Turn 2**

**User (audio):** "Tuesday morning. Mei is allergic to peanuts, so keep that in mind for any food suggestions."

→ Qwen caption (audio only):
- Acoustic: (none notable)
- Paralinguistic: matter-of-fact tone
- Semantic: "Tuesday morning. Mei is allergic to peanuts, so keep that in mind for any food suggestions."

**Assistant (seed text):** "Noted. I'll look for morning trains on Tuesday."

→ Notetaker updates notes from caption + assistant reply:
- Fact: travel Tuesday morning
- Instruction (active): accommodate Mei's peanut allergy in food suggestions
- Assistant state: will look for morning trains on Tuesday

**Turn 3**

**User (audio):** "Actually, scratch that — we need Thursday morning, not Tuesday."

→ Qwen caption (audio only):
- Acoustic: (none notable)
- Paralinguistic: corrective, slightly hurried
- Semantic: "Actually, scratch that — we need Thursday morning, not Tuesday."

**Assistant (seed text):** "Understood, Thursday morning it is."

→ Notetaker updates notes from caption + assistant reply:
- Correction: travel day changed from Tuesday to Thursday morning
- Assistant state: confirmed Thursday morning

**Turn 4 — final answer**

Turn 4 is **not** captioned or passed through the notetaker. Qwen receives the rendered notes (same layout as in real runs) plus the final user audio:

```
=== ACTIVE INSTRUCTIONS ===
- Accommodate Mei's peanut allergy in any food suggestions. [active]

=== CURRENT FACTS ===
- User is booking a train to Edinburgh.
- User's sister Mei is travelling with them.
- Travel time is Thursday morning.

---

=== TURN 1 ===
Acoustic: quiet indoor setting
Paralinguistic: calm tone; informative; steady speaking rate
Transcript: I'm booking a train to Edinburgh. My sister Mei is travelling with me.
Assistant:
- Asked when the user would like to travel.

---

=== TURN 2 ===
Acoustic: (none detected)
Paralinguistic: matter-of-fact tone; clear voice
Transcript: Tuesday morning. Mei is allergic to peanuts, so keep that in mind for any food suggestions.
Assistant:
- Noted the request; will look for morning trains on Tuesday.

---

=== TURN 3 ===
Acoustic: (none detected)
Paralinguistic: corrective tone; slightly hurried
Transcript: Actually, scratch that — we need Thursday morning, not Tuesday.
Corrections:
- User corrected travel day from Tuesday to Thursday morning.
Assistant:
- Confirmed Thursday morning.
```

**User (audio):** "Great. Can you suggest a café near the station for when we arrive? Remember Mei's allergy."

**Model reply (expected):** Same as one-shot — café near the station, allergy-safe, Thursday not Tuesday.
