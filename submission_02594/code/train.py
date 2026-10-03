"""train.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # TODO: chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    model.eval()
    preds = []
    n_samples = len(X)
    for i in range(0, n_samples, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds.append(logits.argmax(dim=1))
    return torch.cat(preds, dim=0)


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        y_one_hot = F.one_hot(y, num_classes=logits.shape[-1]).float()
        return F.mse_loss(logits, y_one_hot)
    else:
        raise ValueError(f"Hàm mất mát không hợp lệ: '{loss_name}'. Chỉ chấp nhận 'ce' hoặc 'mse'")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    model.eval()
    total_loss = 0.0
    n_samples = len(X)
    all_preds = []

    for i in range(0, n_samples, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        if loss_name == "ce":
            loss = F.cross_entropy(logits, yb, reduction="sum")
        elif loss_name == "mse":
            y_one_hot = F.one_hot(yb, num_classes=logits.shape[-1]).float()
            loss = F.mse_loss(logits, y_one_hot, reduction="sum")
        else:
            raise ValueError(f"Hàm loss không hợp lệ: '{loss_name}'")

        total_loss += float(loss.item())
        all_preds.append(logits.argmax(dim=1))

    preds = torch.cat(all_preds, dim=0)
    acc = float((preds == y).float().mean().item())

    # Ma trận nhầm lẫn (7, 7)
    y_true = y.cpu().numpy()
    y_pred = preds.cpu().numpy()
    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    macro_f1 = macro_f1_from_confusion(cm)

    return {"loss": total_loss / n_samples, "acc": acc, "macro_f1": macro_f1}


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)
    """
    set_seed(cfg.get("seed", 42))

    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init = cfg.get("init", "he")
    in_features = data["X_tr"].shape[1]
    num_classes = 7

    model = MLP(hidden=hidden, dropout=dropout, init=init,
                in_features=in_features, num_classes=num_classes)
    expected = EXPECTED_PARAMS.get(hidden)
    if expected is not None:
        assert count_params(model) == expected, f"Số tham số {count_params(model)} != {expected}"

    device = data["X_tr"].device
    model.to(device)

    lr = cfg["lr"]
    if lr is None:
        raise ValueError("lr không được là None. Hãy chọn lr bằng tập val trước.")

    optimizer = build_optimizer(
        name=cfg["optimizer"],
        params=model.parameters(),
        lr=lr,
        weight_decay=cfg.get("weight_decay", 0.0),
        momentum=cfg.get("momentum", 0.9),
    )

    precision = cfg.get("precision", "fp32")
    is_cuda = device.type == "cuda"
    scaler = None
    amp_dtype = None

    if precision == "fp16" and is_cuda:
        scaler = torch.amp.GradScaler("cuda")
        amp_dtype = torch.float16
    elif precision == "bf16" and is_cuda:
        amp_dtype = torch.bfloat16

    # 1. Loss bước 0 trên val TRƯỚC bước cập nhật đầu tiên
    step0_loss = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])["loss"]

    epochs = cfg.get("epochs", 20)
    batch_size = cfg.get("batch", 512)
    clip_norm = cfg.get("clip_norm", None)
    loss_name = cfg.get("loss", "ce")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    # Tập con train cố định 50.000 mẫu để đo train_loss ở eval mode nhanh hơn
    train_subset_size = min(50_000, len(data["X_tr"]))
    X_tr_eval = data["X_tr"][:train_subset_size]
    y_tr_eval = data["y_tr"][:train_subset_size]

    best_val_loss = float("inf")
    best_epoch = 1
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    diverged = False

    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.get("seed", 42))

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        model.train()
        epoch_grad_norms = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size, generator=gen, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            if amp_dtype is not None and is_cuda:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, loss_name)
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, loss_name)

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                optimizer.step()

            epoch_grad_norms.append(gn)

        if diverged:
            print(f"[{cfg.get('exp_id', 'exp')}] Diverged (NaN/inf loss) tại epoch {epoch}!")
            break

        if is_cuda:
            torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - t0

        # Đánh giá cuối epoch ở chế độ eval()
        tr_eval = evaluate(model, X_tr_eval, y_tr_eval, loss_name=loss_name)
        val_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=loss_name)
        mean_gn = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_eval["loss"])
        history["val_loss"].append(val_eval["loss"])
        history["val_acc"].append(val_eval["acc"])
        history["val_macro_f1"].append(val_eval["macro_f1"])
        history["grad_norm"].append(mean_gn)
        history["epoch_time_s"].append(epoch_time)

        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    peak_mem_MB = 0.0
    if is_cuda:
        peak_mem_MB = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    best_idx = best_epoch - 1 if history["epoch"] else -1
    summary = {
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss) if not diverged else float("nan"),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else float("nan"),
        "final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else float("nan"),
        "val_acc": float(history["val_acc"][best_idx]) if best_idx >= 0 else 0.0,
        "val_macro_f1": float(history["val_macro_f1"][best_idx]) if best_idx >= 0 else 0.0,
        "time_per_epoch_s": float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0,
        "peak_mem_MB": round(peak_mem_MB, 2),
        "diverged": diverged,
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    import pandas as pd
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"row_id": row_id, "pred": preds})
    df.to_csv(path, index=False)
    print(f"Đã ghi {len(df)} dòng dự đoán eval vào {path}")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    hidden = tuple(cfg.get("hidden", (256, 128)))
    model = MLP(
        hidden=hidden,
        dropout=0.0,  # Eval mode luôn tắt dropout
        in_features=data["X_eval"].shape[1],
        num_classes=7,
    )
    model.load_state_dict(result["best_state"])
    model.to(data["X_eval"].device)

    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
