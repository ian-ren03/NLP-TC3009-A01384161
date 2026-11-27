import time, torch, torch.nn as nn
from src.data import PAD_IDX


def run_epoch(model, loader, device, optimizer=None, scaler=None, clip=1.0, use_amp=True):
    """Una época. Con optimizer: entrena. Sin optimizer: evalúa (sin dropout, sin gradientes)."""
    train = optimizer is not None
    model.train(train)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction='sum')
    total_loss, total_tokens = 0.0, 0
    with torch.set_grad_enabled(train):
        for src, src_lens, tgt in loader:
            src, src_lens, tgt = src.to(device), src_lens.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                logits, _ = model(src, src_lens, tgt_in)
                loss_sum = crit(logits.reshape(-1, logits.size(-1)).float(), tgt_out.reshape(-1))
            n_tok = (tgt_out != PAD_IDX).sum().item()
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_sum / n_tok
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
            total_loss += loss_sum.item()
            total_tokens += n_tok
    return total_loss / total_tokens
