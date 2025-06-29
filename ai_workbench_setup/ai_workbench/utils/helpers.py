def validate_text(text: str) -> str:
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string")
    return text.strip()