"""plots.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from pathlib import Path


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cfg = result.get("cfg", {})
    history = result.get("history", {})
    summary = result.get("summary", {})

    epochs = history.get("epoch", [])
    if not epochs:
        return

    best_epoch = summary.get("best_epoch", 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Ô 1: Train & Val Loss
    axes[0].plot(epochs, history.get("train_loss", []), label="Train Loss (eval mode)", color="royalblue", lw=1.8)
    axes[0].plot(epochs, history.get("val_loss", []), label="Val Loss", color="crimson", lw=1.8)
    axes[0].axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Ep ({best_epoch})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train & Val Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Ô 2: Val Accuracy & Macro-F1
    axes[1].plot(epochs, history.get("val_acc", []), label="Val Accuracy", color="forestgreen", lw=1.8)
    if "val_macro_f1" in history:
        axes[1].plot(epochs, history["val_macro_f1"], label="Val Macro-F1", color="darkorange", lw=1.8)
    axes[1].axvline(best_epoch, color="gray", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Val Accuracy & Macro-F1")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    # Ô 3: Gradient Norm (trước khi clip)
    axes[2].plot(epochs, history.get("grad_norm", []), label="Grad Norm (pre-clip)", color="purple", lw=1.8)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Norm")
    axes[2].set_title("Global Gradient Norm")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend()

    title_str = (
        f"Exp: {cfg.get('exp_id', 'N/A')} | Opt: {cfg.get('optimizer', 'N/A')} "
        f"(lr={cfg.get('lr')}, batch={cfg.get('batch')}, dropout={cfg.get('dropout')}, init={cfg.get('init')})"
    )
    fig.suptitle(title_str, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for res in results:
        cfg = res.get("cfg", {})
        history = res.get("history", {})
        exp_id = cfg.get("exp_id", "exp")
        epochs = history.get("epoch", [])
        vals = history.get(metric, [])
        if epochs and vals:
            ax.plot(epochs, vals, marker="o", markersize=3, lw=1.5, label=exp_id)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(title or f"So sánh {metric} giữa các thí nghiệm", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
