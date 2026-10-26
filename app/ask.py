"""1つのモデルに、共通の会話履歴を与えてtemperatureを比較するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask

各ターンでtemperature 0.2と1.0の回答を表示する。
次のターンに進むときに採用する回答を選ぶことで、両方の条件に同じ会話履歴を渡す。
"""

import argparse

# Gemini SDKの会話履歴はContent形式で渡すため、その型と履歴引数用のSequenceを使う。
from collections.abc import Sequence
from dataclasses import dataclass

from google import genai
from google.genai import errors, types

from app.config import FALLBACK_MODELS, MODEL, api_error_message, get_api_key

DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
# 毎ターン、この2つのtemperatureで同じ質問・履歴に対する回答を比較する。
TEMPERATURES = (0.2, 1.0)


@dataclass(frozen=True)
class AskResult:
    """表示用の回答文と、会話履歴に使うSDKの応答を保持する。"""

    text: str
    content: types.Content


class EmptyModelResponseError(RuntimeError):
    """モデルが会話に追加できる回答を返さなかったことを示す。"""


def ask(
    question: str,
    temperature: float = 0.2,
    history: Sequence[types.Content] = (),
    model_name: str = MODEL,
) -> AskResult:
    """共通の会話履歴と質問を指定temperatureでモデルに送り、回答を返す。"""
    client = genai.Client(api_key=get_api_key())
    # 過去の会話の後ろに今回の質問を追加し、同じ履歴を各temperatureに渡す。
    contents = [
        *history,
        types.Content(role="user", parts=[types.Part(text=question)]),
    ]
    response = client.models.generate_content(
        # 呼び出し元が選んだモデルを使い、temperatureだけを比較対象にする。
        model=model_name,
        contents=contents,
        config={"temperature": temperature, "max_output_tokens": 300},
    )

    # APIがトークン使用量を返さない場合もあるため、不明として表示する。
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage is not None else None
    output_tokens = usage.candidates_token_count if usage is not None else None
    print(
        f"[トークン] 入力: {input_tokens if input_tokens is not None else '不明'} / "
        f"出力: {output_tokens if output_tokens is not None else '不明'}"
    )

    # 回答本文だけで作り直すとthought signatureなどSDK応答の情報が失われる。
    candidates = response.candidates
    if not candidates or candidates[0].content is None or response.text is None:
        raise EmptyModelResponseError("モデルから会話に使えるテキスト回答がありませんでした。")
    return AskResult(text=response.text, content=candidates[0].content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="会話を続けながらtemperatureによる回答の違いを比べる"
    )
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    # 両条件の比較で共有する履歴。会話開始時は空にする。
    history: list[types.Content] = []
    question = args.question
    print(f"[質問] {question}")

    # 設定モデルを最初に試し、失敗時は重複を除いた予備モデルへ進む。
    model_candidates = tuple(dict.fromkeys((MODEL, *FALLBACK_MODELS)))

    while True:
        for model_name in model_candidates:
            answers: dict[float, AskResult] = {}
            print(f"\n[model] {model_name}")
            try:
                # 1つのモデルで両temperatureが成功したときだけ、その比較を採用する。
                for temperature in TEMPERATURES:
                    print(f"\n[temperature] {temperature}")
                    answers[temperature] = ask(
                        question, temperature, history, model_name
                    )
            except EmptyModelResponseError as e:
                # 空の回答を会話履歴に混ぜず、次のモデルを試す。
                print(f"[失敗] {model_name}: {e}")
                continue
            except errors.APIError as e:
                # 一方の温度で失敗しても部分的な比較結果は使わず、次のモデルへ進む。
                print(
                    f"[失敗] {model_name}: {api_error_message(e.code, e.message)}"
                )
                continue

            for temperature in TEMPERATURES:
                print(f"\n[temperature] {temperature}")
                print(answers[temperature].text)
            break
        else:
            # すべての候補が失敗した場合は理由を示して失敗終了する。
            print("[エラー] 利用できるモデルがありません。モデル設定を確認してください。")
            raise SystemExit(1)

        # 成功したモデル名を表示することで、予備モデルへの切り替えを知らせる。
        if model_name != MODEL:
            print(f"\n[予備モデルを使用] {model_name}")

        # 次のターンに採用する回答を選び、想定外の入力は選び直してもらう。
        while True:
            choice = input(
                "\n次の会話に使う回答を選んでください "
                "(1 または 0.2: temperature 0.2 / "
                "2 または 1.0: temperature 1.0 / q: 終了): "
            ).strip().lower()
            if choice == "q":
                break
            if choice in ("1", "0.2"):
                selected_temperature = TEMPERATURES[0]
                break
            if choice in ("2", "1.0"):
                selected_temperature = TEMPERATURES[1]
                break
            print("選択を読み取れませんでした。1、0.2、2、1.0、または q を入力してください。")

        if choice == "q":
            break

        # 選ばれた回答だけを履歴に追加し、次ターンも両条件で共有する。
        history.extend(
            [
                types.Content(
                    role="user",
                    parts=[types.Part(text=question)],
                ),
                # 返ってきたContentをそのまま使い、signatureなどの情報を保持する。
                answers[selected_temperature].content,
            ]
        )
        # 続きの質問を受け取り、qまたは空入力なら会話を終える。
        question = input("\n次の質問 (終了: q): ").strip()
        if question.lower() == "q":
            break
        if not question:
            print("質問が空なので終了します。")
            break


if __name__ == "__main__":
    main()
