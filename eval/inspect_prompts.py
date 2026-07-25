"""Prompts for AudioMC inspection summaries."""

SUMMARIZE_SYSTEM = """You summarize multi-turn conversations for error analysis.

You will receive:
- The full conversation (user transcripts and assistant replies)
- The model-generated final assistant reply
- Grading rubrics for the final turn

Produce ONE overall summary for the entire example:
1. context — 2-4 sentences: what the conversation is about overall
2. question — 1-2 sentences: what the user asked or said in the final turn
3. answer — 1-2 sentences: what the model generated in the final turn
4. gt — 1-3 sentences: what a correct final answer should satisfy (based on rubrics; no reference transcript exists)

Return ONLY valid JSON (no markdown fences):
{"context": "...", "question": "...", "answer": "...", "gt": "..."}"""
