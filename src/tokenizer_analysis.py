"""
Tokenizer analysis script.

This script will be used to compare subword fragmentation across
English, Chinese, and Korean datasets under multilingual tokenizers.
"""

from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd


def count_words(text):
    """
    Count words using whitespace splitting.
    This is a simple approximation for tokenizer analysis.
    """
    return len(text.split())


def count_subword_tokens(text, tokenizer):
    """
    Count subword tokens produced by a multilingual tokenizer.
    """
    return len(tokenizer.tokenize(text))


def analyze_dataset_texts(texts, tokenizer, language_name):
    """
    Analyze average word count, subword token count, and subword/word ratio.
    """
    records = []

    for text in texts:
        word_count = max(count_words(text), 1)
        subword_count = count_subword_tokens(text, tokenizer)
        ratio = subword_count / word_count

        records.append({
            "language": language_name,
            "word_count": word_count,
            "subword_count": subword_count,
            "subword_word_ratio": ratio
        })

    return pd.DataFrame(records)


def main():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    # Placeholder examples.
    # Full dataset-based analysis will be added later.
    examples = {
        "English": [
            "This project studies multilingual Transformer distillation."
        ],
        "Chinese": [
            "本项目研究多语言Transformer模型蒸馏。"
        ],
        "Korean": [
            "이 프로젝트는 다국어 Transformer 지식 증류를 연구합니다."
        ]
    }

    all_results = []

    for language, texts in examples.items():
        df = analyze_dataset_texts(texts, tokenizer, language)
        all_results.append(df)

    result_df = pd.concat(all_results, ignore_index=True)
    print(result_df)

    summary = result_df.groupby("language").mean(numeric_only=True)
    print(summary)


if __name__ == "__main__":
    main()
