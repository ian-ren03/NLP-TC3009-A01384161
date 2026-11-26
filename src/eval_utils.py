import json
import math
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import sacrebleu
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from src.data import EOS_IDX, PAD_IDX, SOS_IDX, UNK_IDX, clean_text, encode
from src.persist import save_experiment
from src.transformer import count_trainable_parameters


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    grad_clip: float = 1.0,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    use_amp = scaler is not None and str(device).startswith("cuda")

    for src, _, tgt in loader:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        n_tokens = (tgt_out != PAD_IDX).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for src, _, tgt in loader:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        n_tokens = (tgt_out != PAD_IDX).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20.0))
    return avg_loss, perplexity


def train_transformer(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    model_name: str,
    save_dir: str,
    device: str,
    grad_clip: float = 1.0,
    patience: int | None = None,
    use_amp: bool = True,
    scheduler=None,
) -> dict:
    """Entrena un modelo Transformer guardando el mejor checkpoint según val_loss.

    Nunca accede a test_pairs (regla de aislamiento absoluto).
    """
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scaler = torch.amp.GradScaler("cuda") if (use_amp and str(device).startswith("cuda")) else None
    os.makedirs(save_dir, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_ppl": [],
        "epoch_time_s": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
    }

    no_improve = 0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device, grad_clip, scaler)
        val_loss, val_ppl = evaluate_loss(model, val_loader, criterion, device)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["epoch_time_s"].append(elapsed)

        marker = ""
        if val_loss < history["best_val_loss"]:
            history["best_val_loss"] = val_loss
            history["best_epoch"] = epoch
            no_improve = 0
            save_experiment(model, optimizer, history, f"{model_name}_best", save_dir)
            marker = " *"
        else:
            no_improve += 1

        print(
            f"Época {epoch:02d}/{epochs} | "
            f"Train Loss: {tr_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) | "
            f"Tiempo: {elapsed:.1f}s{marker}"
        )

        if patience is not None and no_improve >= patience:
            print(f"Early stopping activado en época {epoch} (paciencia={patience}).")
            break

    # Guardar también el estado final para tener el historial completo de sobreajuste
    save_experiment(model, optimizer, history, f"{model_name}_last", save_dir)
    return history


def decode_ids(ids: list[int], idx2word: dict[int, str]) -> str:
    tokens = []
    for idx in ids:
        if idx == SOS_IDX:
            continue
        if idx == EOS_IDX:
            break
        if idx == PAD_IDX:
            continue
        tokens.append(idx2word.get(idx, "<unk>"))
    return " ".join(tokens)


@torch.no_grad()
def translate_pairs(
    model: nn.Module,
    pairs: list[tuple[str, str]],
    eng_word2idx: dict[str, int],
    spa_idx2word: dict[int, str],
    device: str,
    batch_size: int = 64,
    max_len: int = 64,
) -> list[str]:
    model.eval()
    predictions = []

    for i in tqdm(range(0, len(pairs), batch_size), desc="Traduciendo test"):
        batch_pairs = pairs[i : i + batch_size]
        src_tensors = [
            torch.tensor(encode(en, eng_word2idx), dtype=torch.long)
            for en, _ in batch_pairs
        ]
        src_batch = pad_sequence(src_tensors, batch_first=True, padding_value=PAD_IDX).to(device)
        out_ids = model.generate_greedy(src_batch, SOS_IDX, EOS_IDX, max_len=max_len)
        for row in out_ids.cpu().tolist():
            predictions.append(decode_ids(row, spa_idx2word))

    return predictions


@torch.no_grad()
def measure_inference_latency_ms(
    model: nn.Module,
    pairs: list[tuple[str, str]],
    eng_word2idx: dict[str, int],
    device: str,
    n_samples: int = 200,
    max_len: int = 64,
) -> float:
    """Mide latencia promedio de inferencia (ms/oración) con batch_size=1."""
    model.eval()
    sample_pairs = pairs[: min(n_samples, len(pairs))]

    # Warm-up
    for en, _ in sample_pairs[:5]:
        src = torch.tensor([encode(en, eng_word2idx)], dtype=torch.long, device=device)
        _ = model.generate_greedy(src, SOS_IDX, EOS_IDX, max_len=max_len)

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for en, _ in sample_pairs:
        src = torch.tensor([encode(en, eng_word2idx)], dtype=torch.long, device=device)
        _ = model.generate_greedy(src, SOS_IDX, EOS_IDX, max_len=max_len)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms / len(sample_pairs)


def compute_bleu_breakdown(
    predictions: list[str],
    pairs: list[tuple[str, str]],
    train_en_set: set[str] | None = None,
) -> dict:
    """Calcula sacrebleu con normalización simétrica (clean_text) global y por longitud."""
    preds_clean = [clean_text(p) for p in predictions]
    refs_clean = [clean_text(es) for _, es in pairs]

    bleu_global = sacrebleu.corpus_bleu(preds_clean, [refs_clean]).score

    med_idx = [i for i, (en, _) in enumerate(pairs) if 8 <= len(en.split()) <= 18]
    long_idx = [i for i, (en, _) in enumerate(pairs) if len(en.split()) > 18]

    bleu_medium = sacrebleu.corpus_bleu(
        [preds_clean[i] for i in med_idx],
        [[refs_clean[i] for i in med_idx]],
    ).score

    bleu_long = sacrebleu.corpus_bleu(
        [preds_clean[i] for i in long_idx],
        [[refs_clean[i] for i in long_idx]],
    ).score

    result = {
        "bleu_global": round(bleu_global, 2),
        "bleu_medium_8_18": round(bleu_medium, 2),
        "bleu_long_gt_18": round(bleu_long, 2),
        "n_medium": len(med_idx),
        "n_long": len(long_idx),
    }

    if train_en_set is not None:
        unseen_idx = [i for i, (en, _) in enumerate(pairs) if en not in train_en_set]
        bleu_unseen = sacrebleu.corpus_bleu(
            [preds_clean[i] for i in unseen_idx],
            [[refs_clean[i] for i in unseen_idx]],
        ).score
        result["bleu_unseen_test"] = round(bleu_unseen, 2)
        result["n_unseen"] = len(unseen_idx)

    return result


def get_qualitative_indices(test_pairs: list[tuple[str, str]]) -> list[int]:
    """Selecciona 5 índices deterministas de test_pairs (2 medias y 3 complejas > 20 palabras)."""
    med = [i for i, (en, _) in enumerate(test_pairs) if 10 <= len(en.split()) <= 15]
    long_complex = [i for i, (en, _) in enumerate(test_pairs) if len(en.split()) > 20]
    return [med[0], med[10], long_complex[0], long_complex[5], long_complex[10]]


def evaluate_on_test(
    model: nn.Module,
    model_name: str,
    test_loader,
    test_pairs: list[tuple[str, str]],
    eng_word2idx: dict[str, int],
    spa_word2idx: dict[str, int],
    history: dict,
    device: str,
    train_pairs: list[tuple[str, str]] | None = None,
    results_dir: str = "results",
    batch_size: int = 64,
    max_len: int = 64,
) -> tuple[dict, list[str]]:
    """Evaluación final completa sobre test_pairs para llenar el Cuadro 1."""
    os.makedirs(results_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    test_loss, test_ppl = evaluate_loss(model, test_loader, criterion, device)

    spa_idx2word = {v: k for k, v in spa_word2idx.items()}
    predictions = translate_pairs(
        model, test_pairs, eng_word2idx, spa_idx2word, device, batch_size=batch_size, max_len=max_len
    )

    train_en_set = {en for en, _ in train_pairs} if train_pairs is not None else None
    bleu_stats = compute_bleu_breakdown(predictions, test_pairs, train_en_set=train_en_set)
    latency_ms = measure_inference_latency_ms(
        model, test_pairs, eng_word2idx, device, n_samples=200, max_len=max_len
    )

    qual_indices = get_qualitative_indices(test_pairs)
    qualitative = [
        {
            "test_idx": idx,
            "src_en": test_pairs[idx][0],
            "ref_es": test_pairs[idx][1],
            "ref_clean": clean_text(test_pairs[idx][1]),
            "pred_es": predictions[idx],
            "src_words": len(test_pairs[idx][0].split()),
        }
        for idx in qual_indices
    ]

    summary = {
        "model_name": model_name,
        "trainable_params": count_trainable_parameters(model),
        "best_epoch": history["best_epoch"],
        "best_val_loss": round(history["best_val_loss"], 4),
        "avg_epoch_time_s": round(float(np.mean(history["epoch_time_s"])), 2),
        "test_loss": round(test_loss, 4),
        "test_perplexity": round(test_ppl, 2),
        "inference_latency_ms": round(latency_ms, 2),
        **bleu_stats,
        "history": history,
        "qualitative_examples": qualitative,
    }

    out_path = os.path.join(results_dir, f"{model_name}_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Resultados guardados en: {out_path}")
    return summary, predictions


def plot_learning_curves(history: dict, title: str, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    best_ep = history["best_epoch"]
    best_val = history["best_val_loss"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="s", label="Validation Loss")
    plt.axvline(best_ep, color="gray", linestyle="--", alpha=0.7, label=f"Mejor época ({best_ep}: {best_val:.3f})")
    plt.title(title)
    plt.xlabel("Época")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Gráfica guardada en: {save_path}")


@torch.no_grad()
def plot_attention_heatmap(
    model: nn.Module,
    sentence_en: str,
    eng_word2idx: dict[str, int],
    spa_idx2word: dict[int, str],
    device: str,
    title: str,
    save_path: str,
    max_len: int = 50,
):
    """Genera y guarda el mapa de calor de Cross-Attention (última capa del Decoder)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()

    src_tokens = clean_text(sentence_en).split()
    src_ids = [eng_word2idx.get(w, UNK_IDX) for w in src_tokens]
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    out_ids, attn = model.generate_greedy(
        src_tensor, SOS_IDX, EOS_IDX, max_len=max_len, return_attn=True
    )
    # attn tiene forma (1, tgt_len_in, src_len); out_ids tiene <sos> al inicio
    pred_ids = out_ids[0].cpu().tolist()[1:]  # sin <sos>
    tgt_tokens = []
    for idx in pred_ids:
        tok = spa_idx2word.get(idx, "<unk>")
        tgt_tokens.append(tok)
        if idx == EOS_IDX:
            break

    attn_matrix = attn[0, : len(tgt_tokens), : len(src_tokens)].cpu().numpy()

    fig, ax = plt.subplots(figsize=(max(7, len(src_tokens) * 0.65), max(5, len(tgt_tokens) * 0.55)))
    im = ax.imshow(attn_matrix, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="Peso de atención $\\alpha_{ij}$")

    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens)
    ax.set_xlabel("Tokens de entrada (Inglés)")
    ax.set_ylabel("Tokens generados (Español)")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Heatmap guardado en: {save_path}")
    return " ".join(t for t in tgt_tokens if t != "<eos>")
