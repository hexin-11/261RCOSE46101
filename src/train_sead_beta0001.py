"""
SEAD beta=0.001 for KLUE-NLI.

This script trains a student model using:
1. Cross-entropy loss with ground-truth labels.
2. Vanilla knowledge distillation loss from the improved XLM-R teacher.
3. Soft attention entropy alignment loss with beta=0.001.

The trained SEAD student model is saved for later attention entropy analysis.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm
import evaluate


def preprocess_dataset(dataset, tokenizer, max_length=128):
    def preprocess_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    dataset = dataset.map(preprocess_function, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    return dataset


def compute_attention_entropy(attentions, attention_mask=None, eps=1e-12):
    entropy_values = []

    for layer_attention in attentions:
        attention_probs = layer_attention.clamp(min=eps)

        entropy = -torch.sum(
            attention_probs * torch.log(attention_probs),
            dim=-1
        )

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            entropy = entropy * mask
            entropy = entropy.sum() / mask.sum().clamp(min=1)
        else:
            entropy = entropy.mean()

        entropy_values.append(entropy)

    return torch.stack(entropy_values).mean()


def sead_loss_fn(
    student_logits,
    teacher_logits,
    labels,
    student_attentions,
    teacher_attentions,
    attention_mask,
    temperature=2.0,
    alpha=0.5,
    beta=0.001,
):
    ce_loss = F.cross_entropy(student_logits, labels)

    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    kd_loss = F.kl_div(
        student_soft,
        teacher_soft,
        reduction="batchmean"
    ) * (temperature ** 2)

    student_entropy = compute_attention_entropy(
        student_attentions,
        attention_mask=attention_mask
    )

    teacher_entropy = compute_attention_entropy(
        teacher_attentions,
        attention_mask=attention_mask
    )

    entropy_loss = F.mse_loss(student_entropy, teacher_entropy.detach())

    total_loss = (
        (1 - alpha) * ce_loss
        + alpha * kd_loss
        + beta * entropy_loss
    )

    return total_loss, ce_loss, kd_loss, entropy_loss


def evaluate_model(model, dataloader, device):
    accuracy = evaluate.load("accuracy")
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            logits = outputs.logits
            loss = outputs.loss
            preds = torch.argmax(logits, dim=-1)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy.compute(predictions=all_preds, references=all_labels)

    return {
        "eval_loss": total_loss / len(dataloader),
        "eval_accuracy": acc["accuracy"],
    }


def main():
    teacher_model_path = "experiments/teacher_klue_nli_improved/checkpoint-1250"
    student_model_name = "distilbert-base-multilingual-cased"

    result_dir = "experiments/results"
    student_output_dir = "experiments/sead_beta0001_student"

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(student_output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = load_dataset("klue", "nli")

    train_dataset = dataset["train"].select(range(5000))
    eval_dataset = dataset["validation"].select(range(1000))

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(student_model_name)

    print("Preprocessing dataset...")
    train_dataset = preprocess_dataset(train_dataset, tokenizer)
    eval_dataset = preprocess_dataset(eval_dataset, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=16)

    print("Loading improved teacher model...")
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        teacher_model_path,
        output_attentions=True,
    ).to(device)

    print("Loading student model...")
    student_model = AutoModelForSequenceClassification.from_pretrained(
        student_model_name,
        num_labels=3,
        output_attentions=True,
    ).to(device)

    teacher_model.eval()
    student_model.train()

    optimizer = AdamW(student_model.parameters(), lr=2e-5)

    num_epochs = 2
    total_steps = len(train_loader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    temperature = 2.0
    alpha = 0.5
    beta = 0.001

    print("Training student model with SEAD beta=0.001...")

    for epoch in range(num_epochs):
        total_train_loss = 0.0
        total_ce_loss = 0.0
        total_kd_loss = 0.0
        total_entropy_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )
                teacher_logits = teacher_outputs.logits
                teacher_attentions = teacher_outputs.attentions

            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

            student_logits = student_outputs.logits
            student_attentions = student_outputs.attentions

            loss, ce_loss, kd_loss, entropy_loss = sead_loss_fn(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                student_attentions=student_attentions,
                teacher_attentions=teacher_attentions,
                attention_mask=attention_mask,
                temperature=temperature,
                alpha=alpha,
                beta=beta,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_kd_loss += kd_loss.item()
            total_entropy_loss += entropy_loss.item()

            progress_bar.set_postfix({
                "loss": loss.item(),
                "entropy": entropy_loss.item()
            })

        print(f"Epoch {epoch + 1} average total loss: {total_train_loss / len(train_loader):.4f}")
        print(f"Epoch {epoch + 1} average CE loss: {total_ce_loss / len(train_loader):.4f}")
        print(f"Epoch {epoch + 1} average KD loss: {total_kd_loss / len(train_loader):.4f}")
        print(f"Epoch {epoch + 1} average entropy loss: {total_entropy_loss / len(train_loader):.4f}")

    print("Evaluating SEAD beta=0.001 student model...")
    results = evaluate_model(student_model, eval_loader, device)

    print(results)

    output_path = os.path.join(result_dir, "sead_beta0001_klue_nli_results.txt")
    with open(output_path, "w") as f:
        f.write(str(results))

    print(f"Saved results to {output_path}")

    student_model.save_pretrained(student_output_dir)
    tokenizer.save_pretrained(student_output_dir)
    print(f"Saved SEAD beta=0.001 student model to {student_output_dir}")


if __name__ == "__main__":
    main()
