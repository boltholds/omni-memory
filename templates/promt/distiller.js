Extract durable memory candidates from the user message.

Return only valid JSON with this shape:
{
  "facts": [
    {
      "subject": "...",
      "predicate": "...",
      "object": "...",
      "confidence": 0.0,
      "evidence": "...",
      "volatility": "low|medium|high"
    }
  ],
  "episodes": [
    {
      "summary": "...",
      "participants": [],
      "entities": [],
      "confidence": 0.0,
      "evidence": "..."
    }
  ],
  "notes": [
    {
      "text": "...",
      "confidence": 0.0,
      "evidence": "..."
    }
  ],
  "rejected": []
}

Rules:
- Extract only information explicitly stated in the message.
- Do not infer hidden facts.
- Do not store temporary or trivial details.
- If the text contains uncertainty, lower confidence.
- Use normalized predicates like: location, preference, works_with, owns, status, skill, goal.
- Evidence must be a short quote from the input.