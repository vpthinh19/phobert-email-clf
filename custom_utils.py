from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import torch
import regex as re
import os
import json
from typing import List, Dict, Any, Optional

try:
    from underthesea import word_tokenize, text_normalize
    _HAS_UNDERTHESEA = True
except ImportError:
    _HAS_UNDERTHESEA = False

def preprocessing(text: str, use_word_segmentation: bool = False) -> str:
    """
    Tiền xử lý chuỗi văn bản thô.
    - use_word_segmentation: Bật True cho PhoBERT, False cho các mô hình đa ngôn ngữ (mBERT, XLM-R...).
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # text = re.sub(r'http\S+|www\S+|https\S+', ' địa_chỉ_website_lạ ', text, flags=re.MULTILINE)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    
    emoji_pattern = re.compile(r'\p{Emoji}', flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()

    if _HAS_UNDERTHESEA:
        text = text_normalize(text)
        if use_word_segmentation:
            text = word_tokenize(text, format="text")
    elif use_word_segmentation:
        print("Warning: underthesea chưa cài đặt, bỏ qua word segmentation. "
              "Cài bằng: uv add underthesea")
    
    return text

def compute_metrics(pred: Any) -> Dict[str, float]:
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    num_classes = pred.predictions.shape[-1]
    avg = "binary" if num_classes == 2 else "macro"

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average=avg)),
        "precision": float(precision_score(labels, preds, average=avg)),
        "recall": float(recall_score(labels, preds, average=avg)),
    }

def visualize_training_results(
    trainer: Any,
    test_preds,
    test_labels,
    output_dir: str = "./out/visualization",
    label_names: Optional[List[str]] = None,
) -> None:
    """Vẽ dashboard + confusion matrix + lưu JSON. Hỗ trợ binary & multi-class."""
    if label_names is None:
        label_names = [str(i) for i in range(len(set(test_labels)))]

    os.makedirs(output_dir, exist_ok=True)
    history = trainer.state.log_history

    # ── Tách dữ liệu từ log history ──
    train_logs = [l for l in history if "loss" in l and "eval_loss" not in l]
    eval_logs = [l for l in history if "eval_loss" in l]

    train_steps = [l["step"] for l in train_logs]
    train_loss = [l["loss"] for l in train_logs]
    train_lr = [l.get("learning_rate", None) for l in train_logs]

    eval_steps = [l["step"] for l in eval_logs]
    eval_loss = [l["eval_loss"] for l in eval_logs]
    eval_acc = [l.get("eval_accuracy", 0) for l in eval_logs]
    eval_f1 = [l.get("eval_f1", 0) for l in eval_logs]
    eval_prec = [l.get("eval_precision", 0) for l in eval_logs]
    eval_rec = [l.get("eval_recall", 0) for l in eval_logs]

    sns.set_theme(style="whitegrid", font_scale=1.05)

    # ════════════════ Figure 1: Training Dashboard (2×2) ════════════════
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Training Dashboard", fontsize=16, fontweight="bold", y=0.98)

    # (0,0) Loss curves
    ax = axes[0, 0]
    ax.plot(train_steps, train_loss, label="Train Loss", color="#1f77b4", alpha=0.7, linewidth=1.2)
    ax.plot(eval_steps, eval_loss, label="Eval Loss", color="#d62728", marker="o", markersize=4, linewidth=1.5)
    ax.set_title("Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    # Đánh dấu best eval loss
    if eval_loss:
        best_idx = int(np.argmin(eval_loss))
        ax.annotate(f"best: {eval_loss[best_idx]:.4f}",
                    xy=(eval_steps[best_idx], eval_loss[best_idx]),
                    xytext=(10, 15), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9, color="#d62728")

    # (0,1) Learning rate schedule
    ax = axes[0, 1]
    lr_vals = [v for v in train_lr if v is not None]
    if lr_vals:
        ax.plot(train_steps[:len(lr_vals)], lr_vals, color="#2ca02c", linewidth=1.5)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Step")
    ax.set_ylabel("LR")

    # (1,0) Evaluation metrics
    ax = axes[1, 0]
    metrics_cfg = [
        ("Accuracy", eval_acc, "s", "#1f77b4"),
        ("F1-Score", eval_f1, "^", "#ff7f0e"),
        ("Precision", eval_prec, "D", "#2ca02c"),
        ("Recall", eval_rec, "x", "#9467bd"),
    ]
    for name, vals, marker, color in metrics_cfg:
        ax.plot(eval_steps, vals, label=name, marker=marker, markersize=5, color=color, linewidth=1.3)
    ax.set_title("Evaluation Metrics")
    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")

    # (1,1) Summary table
    ax = axes[1, 1]
    ax.axis("off")
    report = classification_report(test_labels, test_preds, target_names=label_names, output_dict=True)
    # Build table rows
    rows = []
    for lbl in label_names:
        r = report[lbl]
        rows.append([lbl, f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1-score']:.4f}", str(int(r['support']))])
    rows.append(["", "", "", "", ""])
    rows.append(["Accuracy", "", "", f"{report['accuracy']:.4f}", str(int(report['macro avg']['support']))])
    rows.append(["Macro avg",
                 f"{report['macro avg']['precision']:.4f}",
                 f"{report['macro avg']['recall']:.4f}",
                 f"{report['macro avg']['f1-score']:.4f}",
                 str(int(report['macro avg']['support']))])

    table = ax.table(
        cellText=rows,
        colLabels=["", "Precision", "Recall", "F1", "Support"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    # Header style
    for col in range(5):
        table[0, col].set_facecolor("#4472C4")
        table[0, col].set_text_props(color="white", fontweight="bold")
    ax.set_title("Classification Report (Test Set)", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir, "training_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ════════════════ Figure 2: Confusion Matrix ════════════════
    cm = confusion_matrix(test_labels, test_preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annot = np.array([[f"{cnt}\n({pct:.1f}%)" for cnt, pct in zip(row_c, row_p)]
                      for row_c, row_p in zip(cm, cm_pct)])

    fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names,
                linewidths=0.5, ax=ax_cm, cbar_kws={"label": "Count"})
    ax_cm.set_title("Confusion Matrix (Test Set)", fontsize=14, fontweight="bold")
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    fig_cm.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_cm)

    # ════════════════ Lưu metrics dạng JSON ════════════════
    json_path = os.path.join(output_dir, "test_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Đã lưu biểu đồ và metrics tại: {output_dir}")