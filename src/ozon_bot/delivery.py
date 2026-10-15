"""Telegram's text limit counts UTF-16 code units, including emoji pairs."""


def split_telegram_text(text: str, limit: int = 3500) -> list[str]:
    if limit < 2:
        raise ValueError("Лимит должен быть не меньше двух символов UTF-16.")
    # Reject control characters that Telegram cannot represent in messages.
    text = "".join(
        char
        for char in text
        if (ord(char) >= 32 or char in "\n\r\t") and not 0xD800 <= ord(char) <= 0xDFFF
    )
    if not text.strip():
        return ["Нет данных."]
    chunks: list[str] = []
    while text:
        units = 0
        end = 0
        for char in text:
            width = 2 if ord(char) > 0xFFFF else 1
            if units + width > limit:
                break
            units += width
            end += 1
        if end < len(text):
            line = text.rfind("\n", 0, end)
            if line > end // 2:
                end = line + 1
        chunks.append(text[:end])
        text = text[end:]
    return chunks
