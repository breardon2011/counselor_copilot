

SYSTEM_PROMPT = """You are a licensed mental-health supervisor.
Guideline: respond using the **{strategy}** support strategy.

You are giving advice to a counselor who is trying to help patients.

Describe the context of the support strategy in a clinical context. 

Give an example of how to use the support strategy from the perspective of the provider to answer the patient challenge given.
 …"""

def build_prompt(user_summary: str, strategy: str, examples: list[tuple[str, str]]):
    system_msg = (
        f"{SYSTEM_PROMPT}"
        f"Strategy detected: **{strategy}**.\n"
    )

    examples_md = "\n".join(
        f"• Patient: {ex}\n  Therapist ({strategy}): {tx}"
        for ex, tx in examples
    ) or "—"

    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": f"Counselor summary: {user_summary}"},
        {"role": "assistant", "content": f"Here are similar past exchanges:\n{examples_md}\n"},
    ]

