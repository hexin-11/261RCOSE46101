from datasets import load_dataset


def load_klue_nli():
    """Load Korean NLI dataset from KLUE."""
    dataset = load_dataset("klue", "nli")
    print("KLUE-NLI loaded successfully.")
    print(dataset)
    return dataset


def load_klue_ner():
    """Load Korean NER dataset from KLUE."""
    dataset = load_dataset("klue", "ner")
    print("KLUE-NER loaded successfully.")
    print(dataset)
    return dataset


def load_xnli(language="en"):
    """Load XNLI dataset for a selected language."""
    dataset = load_dataset("facebook/xnli", language)
    print(f"XNLI-{language} loaded successfully.")
    print(dataset)
    return dataset


if __name__ == "__main__":
    load_klue_nli()
    load_klue_ner()
    load_xnli("en")
    load_xnli("zh")
