"""第1回: 素のLLMに1回だけ質問するスクリプトを変更して
    モデルとtemperatureを変更して計6回違う条件で聞くスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

# argparseは、実行時にコマンドライン引数（質問文）を受け取るための標準ライブラリ。
import argparse

# google-genai SDKから、Gemini APIの呼び出しに使うクライアントとエラー型を読み込む。
from google import genai
from google.genai import errors

# アプリ共通設定から、モデル名・APIエラー文の生成・APIキー取得の関数を読み込む。
from app.config import MODEL, api_error_message, get_api_key

# 質問を省略したときに使う、講座固有の例題。
# このスクリプトは用語集を参照しないため、正しい情報を持たない状態で回答する。
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, model_name: str, temperature: float) -> str:
    # APIキーを使ってクライアントを作り、指定モデルに質問を1回だけ送る。
    # 会話履歴や追加の資料は渡していない。
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(
        model=model_name,
        contents=question,
        config={"temperature": temperature, "max_output_tokens": 300}
        )

    # APIが利用量を返していれば入力・出力トークン数を表示する。
    # 利用量情報がない場合は None を保持し、表示時に「不明」とする。
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage is not None else None
    output_tokens = usage.candidates_token_count if usage is not None else None
    print(
        f"[トークン] 入力: {input_tokens if input_tokens is not None else '不明'} / "
        f"出力: {output_tokens if output_tokens is not None else '不明'}"
    )

    # 呼び出し元にモデルの回答本文を返す。
    return response.text


def main() -> None:
    model_list = ["gemini-3.8-flash","gemini-3.1-flash-lite","gemini-4-31b-it"]
    temperature_list = [0.2, 1.0]
    # コマンドライン引数の質問を受け取る。省略時は講座固有の例題を使う。
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    # 使用するモデルと、実際に送る質問を先に表示する。
    # コメントアウトした　print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")

    # APIエラーは受講生向けの説明文に変換して表示し、失敗終了にする。
    try:
        for model_name in model_list:
            for temperature in temperature_list:
                print(f"\n[model]{model_name} / [temperature]{temperature}")
                answer = ask(args.question, model_name, temperature)
                print(answer)
    
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    # `python -m app.ask` などで直接実行されたときだけ main を開始する。
    main()
