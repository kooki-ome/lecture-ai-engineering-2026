import sys
from types import SimpleNamespace

from app import ask


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    answer_content = ask.types.Content(
        role="model",
        parts=[ask.types.Part(text="回答")],
    )
    response = SimpleNamespace(
        text="回答",
        candidates=[SimpleNamespace(content=answer_content)],
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

    result = ask.ask("質問")
    assert result.text == "回答"
    assert result.content is answer_content
    assert "[トークン] 入力: 12 / 出力: 34" in capsys.readouterr().out


def test_ask_reports_missing_token_metadata(monkeypatch, capsys):
    answer_content = ask.types.Content(
        role="model",
        parts=[ask.types.Part(text="回答")],
    )
    response = SimpleNamespace(
        text="回答",
        candidates=[SimpleNamespace(content=answer_content)],
        usage_metadata=None,
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_: response)
    )
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask.genai, "Client", lambda **_: client)

    ask.ask("質問")
    assert "[トークン] 入力: 不明 / 出力: 不明" in capsys.readouterr().out


def test_ask_uses_configured_model_and_passes_history(monkeypatch):
    answer_content = ask.types.Content(
        role="model",
        parts=[ask.types.Part(text="回答")],
    )
    response = SimpleNamespace(
        text="回答",
        candidates=[SimpleNamespace(content=answer_content)],
        usage_metadata=None,
    )
    captured = {}

    def generate_content(**kwargs):
        captured.update(kwargs)
        return response

    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask.genai, "Client", lambda **_: client)
    prior_turn = ask.types.Content(
        role="user",
        parts=[ask.types.Part(text="前の質問")],
    )

    # 指定した履歴と今回の質問、temperature、設定済みモデルがAPIへ渡ることを確認する。
    result = ask.ask("続きの質問", temperature=1.0, history=[prior_turn])
    assert result.text == "回答"
    assert result.content is answer_content

    assert captured["model"] == ask.MODEL
    assert captured["config"]["temperature"] == 1.0
    assert captured["contents"] == [
        prior_turn,
        ask.types.Content(
            role="user",
            parts=[ask.types.Part(text="続きの質問")],
        ),
    ]


def test_main_reuses_same_history_for_both_temperatures(monkeypatch, capsys):
    calls = []

    def fake_ask(question, temperature, history, model_name):
        calls.append((question, temperature, tuple(history), model_name))
        return ask.AskResult(
            text=f"回答{temperature}",
            content=ask.types.Content(
                role="model",
                parts=[
                    ask.types.Part(
                        text=f"回答{temperature}",
                        thought_signature=f"signature-{temperature}".encode(),
                    )
                ],
            ),
        )

    answers = iter(["2", "続きの質問", "q"])
    monkeypatch.setattr(ask, "ask", fake_ask)
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(sys, "argv", ["app.ask", "最初の質問"])

    # 最初の比較後にtemperature 1.0の回答を選び、次の比較の履歴が共通か確認する。
    ask.main()

    assert calls[0][2] == calls[1][2] == ()
    assert calls[2][0] == calls[3][0] == "続きの質問"
    assert calls[2][2] == calls[3][2]
    assert [message.role for message in calls[2][2]] == ["user", "model"]
    assert calls[2][2][1].parts[0].text == "回答1.0"
    assert calls[2][2][1].parts[0].thought_signature == b"signature-1.0"
    assert "temperature" in capsys.readouterr().out


def test_main_tries_fallback_model_after_api_error(monkeypatch, capsys):
    calls = []

    class FakeAPIError(Exception):
        code = 503
        message = "混雑中"

    def fake_ask(question, temperature, history, model_name):
        calls.append((temperature, model_name))
        if model_name == "main-model" and temperature == 1.0:
            raise FakeAPIError
        return ask.AskResult(
            text=f"{model_name}の回答{temperature}",
            content=ask.types.Content(
                role="model",
                parts=[ask.types.Part(text=f"{model_name}の回答{temperature}")],
            ),
        )

    monkeypatch.setattr(ask.errors, "APIError", FakeAPIError)
    monkeypatch.setattr(ask, "MODEL", "main-model")
    monkeypatch.setattr(ask, "FALLBACK_MODELS", ("backup-model",))
    monkeypatch.setattr(ask, "ask", fake_ask)
    monkeypatch.setattr("builtins.input", lambda _: "q")
    monkeypatch.setattr(sys, "argv", ["app.ask", "質問"])

    ask.main()

    assert calls == [
        (0.2, "main-model"),
        (1.0, "main-model"),
        (0.2, "backup-model"),
        (1.0, "backup-model"),
    ]
    output = capsys.readouterr().out
    assert "[失敗] main-model" in output
    assert "[予備モデルを使用] backup-model" in output
    assert "main-modelの回答0.2" not in output


def test_main_stops_with_message_when_all_models_fail(monkeypatch, capsys):
    class FakeAPIError(Exception):
        code = 503
        message = "混雑中"

    def fake_ask(*_args):
        raise FakeAPIError

    monkeypatch.setattr(ask.errors, "APIError", FakeAPIError)
    monkeypatch.setattr(ask, "MODEL", "main-model")
    monkeypatch.setattr(ask, "FALLBACK_MODELS", ("backup-model",))
    monkeypatch.setattr(ask, "ask", fake_ask)
    monkeypatch.setattr(sys, "argv", ["app.ask", "質問"])

    try:
        ask.main()
    except SystemExit as error:
        assert error.code == 1
    else:
        raise AssertionError("全モデル失敗時は失敗終了する")

    output = capsys.readouterr().out
    assert "[失敗] main-model" in output
    assert "[失敗] backup-model" in output
    assert "利用できるモデルがありません" in output


def test_main_accepts_q_with_surrounding_whitespace(monkeypatch):
    calls = []

    def fake_ask(question, temperature, history, model_name):
        calls.append(temperature)
        return ask.AskResult(
            text=f"回答{temperature}",
            content=ask.types.Content(
                role="model",
                parts=[ask.types.Part(text=f"回答{temperature}")],
            ),
        )

    answers = iter(["1", "  q  "])
    monkeypatch.setattr(ask, "ask", fake_ask)
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(sys, "argv", ["app.ask", "最初の質問"])

    ask.main()

    # qの前後に空白があっても次の会話へ進まず、1ターンで終了する。
    assert calls == [0.2, 1.0]


def test_main_reprompts_after_invalid_answer_selection(monkeypatch, capsys):
    calls = []

    def fake_ask(question, temperature, history, model_name):
        calls.append((question, temperature, tuple(history)))
        return ask.AskResult(
            text=f"回答{temperature}",
            content=ask.types.Content(
                role="model",
                parts=[ask.types.Part(text=f"回答{temperature}")],
            ),
        )

    answers = iter(["間違い", "1.0", "続きの質問", "q"])
    monkeypatch.setattr(ask, "ask", fake_ask)
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(sys, "argv", ["app.ask", "最初の質問"])

    ask.main()

    # 誤入力後に再選択でき、温度値「1.0」で会話を続けられる。
    assert len(calls) == 4
    assert calls[2][0] == calls[3][0] == "続きの質問"
    assert calls[2][2] == calls[3][2]
    assert calls[2][2][1].parts[0].text == "回答1.0"
    assert "選択を読み取れませんでした" in capsys.readouterr().out
