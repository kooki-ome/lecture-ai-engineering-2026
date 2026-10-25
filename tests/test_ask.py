from types import SimpleNamespace

from app import ask


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        text="回答",
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=34,
        ),
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_: response)
    )
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask.genai, "Client", lambda **_: client)

    assert ask.ask("質問") == "回答"
    assert "[トークン] 入力: 12 / 出力: 34" in capsys.readouterr().out


def test_ask_reports_missing_token_metadata(monkeypatch, capsys):
    response = SimpleNamespace(text="回答", usage_metadata=None)
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_: response)
    )
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask.genai, "Client", lambda **_: client)

    ask.ask("質問")
    assert "[トークン] 入力: 不明 / 出力: 不明" in capsys.readouterr().out
