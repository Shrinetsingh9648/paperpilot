from paperpilot.prompts import NOT_FOUND, SYSTEM, format_context, parse_citations

HITS = [{"text": "alpha", "meta": {"title": "T1", "section": "Methods", "page": 3}},
        {"text": "beta", "meta": {"title": "T2", "section": "Results", "page": 5}}]


def test_valid_and_invalid_citations_are_separated():
    valid, invalid = parse_citations("Fact A [1]. Fact B [2][5].", HITS)
    assert valid == [1, 2] and invalid == [5]


def test_context_is_numbered_with_page_info():
    ctx = format_context(HITS)
    assert ctx.startswith("[1] (T1 | Methods | p.3)") and "[2] (T2 | Results | p.5)" in ctx


def test_system_prompt_contains_fallback_sentence():
    assert NOT_FOUND in SYSTEM


def test_missing_key_gives_friendly_error(monkeypatch):
    import pytest
    from paperpilot import llm
    from paperpilot.config import Settings
    # Create a fresh Settings with google provider and clear the Google key
    fake_settings = Settings(llm_provider="google", llm_model="gemini-flash-latest")
    monkeypatch.setattr(llm, "settings", fake_settings)
    monkeypatch.setenv("GOOGLE_API_KEY", "paste-your-gemini-key-here")
    assert not llm.has_key()
    with pytest.raises(llm.LLMError, match="GOOGLE_API_KEY"):
        llm.complete("s", "u")
