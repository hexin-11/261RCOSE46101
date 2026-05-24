"""
Improved teacher fine-tuning for KLUE-NLI.

This script fine-tunes XLM-RoBERTa-base on a larger subset of KLUE-NLI
to obtain a stronger teacher model for later knowledge distillation.
"""

import os
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate


def main():
    model_name = "xlm-roberta-base"

    output_dir = "experiments/teacher_klue_nli_improved"
    result_dir = "experiments/results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    print("Loading dataset...")
    dataset = load_dataset("klue", "nli")

    train_dataset = dataset["train"].select(range(5000))
    eval_dataset = dataset["validation"].select(range(1000))

    print("Loading tokenizer and teacher model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )

    def preprocess_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    print("Tokenizing dataset...")
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    train_dataset = train_dataset.rename_column("label", "labels")
    eval_dataset = eval_dataset.rename_column("label", "labels")

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    eval_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=False,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("Training improved teacher model...")
    trainer.train()

    print("Evaluating improved teacher model...")
    results = trainer.evaluate()

    print(results)

    output_path = os.path.join(result_dir, "teacher_klue_nli_improved_results.txt")
    with open(output_path, "w") as f:
        f.write(str(results))

    print(f"Saved results to {output_path}")
    print(f"Improved teacher checkpoint saved to {output_dir}")


if __name__ == "__main__":
    main()
