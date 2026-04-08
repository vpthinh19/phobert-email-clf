import logging
import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from custom_utils import preprocessing, compute_metrics, visualize_training_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Cấu hình all in one
CONFIG = {
    # Paths
    "model_path": "./models/phobert_v2",
    "trained_model_path": "./models/phobert_spam_final",
    "output_dir": "./out/outputs",
    "visualize_dir": "./out/visualization/phobert",

    # Dataset
    # Hỗ trợ 3 dạng:
    #   1) CSV file        → "dataset_path": "./data/data.csv"
    #   2) Folder cấu trúc → "dataset_path": "./data"  (chứa train/test/label1/label2...)
    #   3) HuggingFace hub → "dataset_path": "username/dataset-name"
    "dataset_path": "./dataset/vi_dataset.csv",
    # "csv" | "folder" | "huggingface"  — nếu None, tự suy từ path
    "dataset_type": "csv",

    # Dataset columns
    # Chuỗi → 1 cột, List[str] → nối nhiều cột (dùng text_separator).
    "text_column": "text",
    "text_separator": " [SEP] ",        # dùng khi text_column là list
    "label_column": "label",

    # Model
    "problem_type": "single_label_classification",
    # label_map: bắt buộc nếu label là chuỗi. Nếu label đã là số, set None.
    # Ví dụ: {"ham": 0, "spam": 1}  hoặc  {"neg": 0, "neu": 1, "pos": 2}
    # Khi None + label số → num_labels tự suy từ max(label)+1, label_names = ["0","1",...]
    "label_map": {"ham": 0, "spam": 1},

    # Preprocessing
    "use_word_segmentation": True,      # PhoBERT → True. mBERT/XLM-R → False.
    "max_length": 256,

    # Dataset split
    "test_size": 0.2,       # train 0.8 / test 0.2
    "val_ratio": 0.5,       # test 0.2 → val 0.1 + test 0.1

    # Training hyperparams
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "eval_accumulation_steps": 2,
    "eval_steps": 50,
    "learning_rate": 2e-5,
    "warmup_steps": 100,
    "weight_decay": 0.001,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
}
# =============================================================


class TrainingPipeline:
    """Facade Pipeline"""

    def __init__(self, cfg: dict = CONFIG):
        self.cfg = cfg
        self.device: torch.device = None
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.trainer: Trainer = None
        self._bf16 = False
        self._fp16 = False
        self._optim = "adamw_torch"
        self._pin_memory = False

    # ──────────────── Device & precision ────────────────
    def setup_device(self):
        """Phát hiện loại device → setup precision, optimizer, matmul"""
        self.device = self._detect_device()

        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            cap = torch.cuda.get_device_capability(self.device)
            self._bf16 = cap >= (8, 0)
            self._fp16 = not self._bf16
            self._pin_memory = True
            try:
                torch.optim.AdamW([torch.zeros(1, device=self.device)], fused=True)
                self._optim = "adamw_torch_fused"
            except RuntimeError:
                pass
            log.info("GPU: %s  |  Compute capability: %s", torch.cuda.get_device_name(self.device), cap)
        elif self.device.type == "mps":
            self._fp16 = True

        log.info("Device: %s  |  bf16=%s  fp16=%s  |  Optim: %s", self.device, self._bf16, self._fp16, self._optim)
        return self

    # ──────────────── Dataset (scan + split) ────────────────
    def load_data(self):
        """
        Load dataset → resolve label_map → split train/val/test.
        """
        c = self.cfg
        path = c["dataset_path"]
        ds_type = c.get("dataset_type") or self._infer_dataset_type(path)

        log.info("Loading dataset [%s] from %s ...", ds_type, path)
        raw_ds = self._load_raw_dataset(path, ds_type)

        # Nếu label_map chưa có → tự suy từ dữ liệu
        self._resolve_label_map(raw_ds)

        # Split
        self._raw_train, self._raw_val, self._raw_test = self._split_dataset(raw_ds)
        log.info("Split  —  Train: %d  |  Val: %d  |  Test: %d",
                 len(self._raw_train), len(self._raw_val), len(self._raw_test))
        return self

    # ──────────────── Model & tokenizer ────────────────
    def load_model(self):
        """
        Load tokenizer & model. Đọc num_labels từ label_map đã resolve.
        """
        c = self.cfg
        label_map = c.get("label_map") or {}
        num_labels = len(label_map) if label_map else 2
        log.info("Loading tokenizer & model from %s  (%d labels) ...", c["model_path"], num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(c["model_path"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            c["model_path"],
            num_labels=num_labels,
            problem_type=c["problem_type"],
        ).to(self.device)

        # Clamp max_length theo max_position_embeddings của model
        model_max = getattr(self.model.config, "max_position_embeddings", None)
        if model_max and c["max_length"] > model_max - 2:  # trừ [CLS] + [SEP]
            old = c["max_length"]
            c["max_length"] = model_max - 2
            log.warning("max_length %d vượt model max_position_embeddings=%d → clamp xuống %d",
                        old, model_max, c["max_length"])

        params = sum(p.numel() for p in self.model.parameters())
        log.info("Model loaded  |  Parameters: %s", f"{params:,}")
        return self

    # ──────────────── Tokenize dataset ────────────────
    def tokenize_data(self):
        """Tokenize raw splits đã load. Trả về DatasetDict mới với format torch."""
        c = self.cfg
        tok_fn = self._make_tokenize_fn()
        cols = ["input_ids", "attention_mask", "labels"]

        log.info("Tokenizing (max_length=%d, word_seg=%s) ...", c["max_length"], c["use_word_segmentation"])
        self.train_dataset = self._raw_train.map(tok_fn, batched=True, remove_columns=self._raw_train.column_names)
        self.val_dataset = self._raw_val.map(tok_fn, batched=True, remove_columns=self._raw_val.column_names)
        self.test_dataset = self._raw_test.map(tok_fn, batched=True, remove_columns=self._raw_test.column_names)

        for d in (self.train_dataset, self.val_dataset, self.test_dataset):
            d.set_format(type="torch", columns=cols)

        # Giải phóng raw splits
        del self._raw_train, self._raw_val, self._raw_test
        return self

    # ──────────────── Build trainer ────────────────
    def build_trainer(self):
        """Xây dựng Transformer Trainer"""
        c = self.cfg
        args = TrainingArguments(
            output_dir=c["output_dir"],
            num_train_epochs=c["num_train_epochs"],
            per_device_train_batch_size=c["per_device_train_batch_size"],
            per_device_eval_batch_size=c["per_device_eval_batch_size"],
            gradient_accumulation_steps=c["gradient_accumulation_steps"],
            eval_accumulation_steps=c["eval_accumulation_steps"],
            eval_strategy="steps",
            eval_steps=c["eval_steps"],
            learning_rate=c["learning_rate"],
            lr_scheduler_type="linear",
            warmup_steps=c["warmup_steps"],
            optim=self._optim,
            weight_decay=c["weight_decay"],
            bf16=self._bf16,
            fp16=self._fp16,
            logging_strategy="steps",
            logging_steps=c["eval_steps"],
            report_to="none",
            load_best_model_at_end=c["load_best_model_at_end"],
            metric_for_best_model=c["metric_for_best_model"],
            dataloader_pin_memory=self._pin_memory,
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
        )
        return self

    # ──────────────── Train → evaluate → visualize → save ────────────────
    def run(self):
        """Chạy toàn bộ pipeline: train ▸ test ▸ visualize ▸ save model."""
        c = self.cfg

        log.info("=" * 55)
        log.info("START TRAINING  —  epochs=%d  lr=%s  batch=%d×%d",
                 c["num_train_epochs"], c["learning_rate"],
                 c["per_device_train_batch_size"], c["gradient_accumulation_steps"])
        log.info("=" * 55)

        t0 = time.perf_counter()
        train_result = self.trainer.train()
        elapsed = time.perf_counter() - t0
        log.info("Training done in %.1fs  |  Final train loss: %.4f", elapsed, train_result.training_loss)

        log.info("Evaluating on test set ...")
        test_results = self.trainer.predict(self.test_dataset)
        m = test_results.metrics
        log.info("Test  —  Acc: %.4f  |  F1: %.4f  |  Prec: %.4f  |  Recall: %.4f",
                 m.get("test_accuracy", 0), m.get("test_f1", 0),
                 m.get("test_precision", 0), m.get("test_recall", 0))

        # Suy label_names từ label_map: {"ham": 0, "spam": 1} → ["Ham", "Spam"]
        label_map = c.get("label_map") or {}
        label_names = [k.capitalize() for k, _ in sorted(label_map.items(), key=lambda x: x[1])]
        test_preds = test_results.predictions.argmax(-1)
        test_labels = test_results.label_ids

        log.info("Saving visualizations → %s", c["visualize_dir"])
        visualize_training_results(
            self.trainer, test_preds, test_labels,
            output_dir=c["visualize_dir"],
            label_names=label_names,
        )

        self.trainer.save_model(c["trained_model_path"])
        self.tokenizer.save_pretrained(c["trained_model_path"])
        log.info("Model saved → %s", c["trained_model_path"])

        return self.trainer, test_results

    # ──────────────── Private helpers ────────────────
    @staticmethod
    def _detect_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _infer_dataset_type(path: str) -> str:
        """Suy loại dataset từ path."""
        if "/" in path and not os.path.exists(path):
            return "huggingface"
        if os.path.isdir(path):
            return "folder"
        ext = os.path.splitext(path)[1].lower()
        if ext in (".csv", ".tsv", ".json", ".jsonl"):
            return "csv"
        return "csv"

    def _load_raw_dataset(self, path: str, ds_type: str) -> DatasetDict:
        """Trả về DatasetDict (có thể chỉ có key 'train', hoặc 'train'+'test')."""
        if ds_type == "csv":
            ext = os.path.splitext(path)[1].lower()
            fmt = "json" if ext in (".json", ".jsonl") else "csv"
            ds = load_dataset(fmt, data_files=path)            # DatasetDict {"train": ...}
            return ds
        if ds_type == "folder":
            # Folder chứa sub-folder train/test, mỗi sub chứa label dirs
            # HuggingFace `imagefolder` hoặc `text` tự detect label từ tên thư mục
            # Ưu tiên text files; nếu có ảnh thì dùng imagefolder
            sample = self._peek_folder(path)
            loader = "imagefolder" if sample == "image" else "text"
            log.info("Folder detected as '%s' dataset", loader)
            return load_dataset(loader, data_dir=path)
        if ds_type == "huggingface":
            return load_dataset(path)
        raise ValueError(f"Không hỗ trợ dataset_type='{ds_type}'")

    @staticmethod
    def _peek_folder(path: str) -> str:
        """Xem file đầu tiên trong folder để quyết định loader."""
        image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        for root, _, files in os.walk(path):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in image_exts:
                    return "image"
                if ext in (".txt", ".csv", ".tsv"):
                    return "text"
        return "text"

    def _resolve_label_map(self, ds: DatasetDict):
        """
        Nếu label_map=None và label trong dataset đã là số → tự build label_map.
        Cập nhật self.cfg["label_map"] để các bước sau dùng thống nhất.
        """
        if self.cfg.get("label_map"):
            return

        # Lấy split đầu tiên có sẵn
        first_split = list(ds.keys())[0]
        label_col = self.cfg["label_column"]
        sample_labels = ds[first_split][label_col]

        # Kiểm tra label là số hay chuỗi
        unique = sorted(set(sample_labels))
        if all(isinstance(v, (int, float)) for v in unique):
            # Label đã là số → label_map: {str(id): id}
            self.cfg["label_map"] = {str(int(v)): int(v) for v in unique}
            log.info("Auto label_map (numeric): %s", self.cfg["label_map"])
        else:
            # Label là chuỗi → build map tự động theo alphabet
            unique_str = sorted(set(str(v).lower() for v in unique))
            self.cfg["label_map"] = {lbl: i for i, lbl in enumerate(unique_str)}
            log.info("Auto label_map (string): %s", self.cfg["label_map"])

    def _split_dataset(self, ds: DatasetDict):
        """Chia train/val/test từ DatasetDict, xử lý mọi trường hợp split có sẵn."""
        c = self.cfg
        has_train = "train" in ds
        has_test = "test" in ds
        has_val = "validation" in ds or "val" in ds

        if has_train and has_test and has_val:
            val_key = "validation" if "validation" in ds else "val"
            return ds["train"], ds[val_key], ds["test"]

        if has_train and has_test:
            # Có train+test, thiếu val → tách val từ train
            split = ds["train"].train_test_split(test_size=c["val_ratio"])
            return split["train"], split["test"], ds["test"]

        if has_train:
            # Chỉ có train → tách hết
            split = ds["train"].train_test_split(test_size=c["test_size"])
            tv = split["test"].train_test_split(test_size=c["val_ratio"])
            return split["train"], tv["train"], tv["test"]

        raise ValueError(f"Dataset không có split 'train'. Các split có: {list(ds.keys())}")

    def _make_tokenize_fn(self):
        tokenizer, c = self.tokenizer, self.cfg
        label_map = c["label_map"]
        num_labels = len(label_map)
        max_len = c["max_length"]
        word_seg = c["use_word_segmentation"]
        text_col = c["text_column"]
        label_col = c["label_column"]
        separator = c.get("text_separator", " [SEP] ")

        # text_column: str → 1 cột, list → nối nhiều cột
        multi_col = isinstance(text_col, list)

        def _fn(examples):
            # ── Labels ──
            raw_labels = examples[label_col]
            labels = []
            for lbl in raw_labels:
                if isinstance(lbl, int):
                    mapped = label_map.get(str(lbl), lbl)
                else:
                    mapped = label_map.get(str(lbl).lower().strip())
                if mapped is None or not (0 <= mapped < num_labels):
                    raise ValueError(
                        f"Label '{lbl}' không hợp lệ. label_map={label_map}, num_labels={num_labels}"
                    )
                labels.append(mapped)

            # ── Texts ──
            if multi_col:
                texts = [
                    separator.join(str(examples[col][i]) for col in text_col)
                    for i in range(len(raw_labels))
                ]
            else:
                texts = [str(t) for t in examples[text_col]]

            texts = [preprocessing(t, use_word_segmentation=word_seg) for t in texts]
            tok = tokenizer(texts, padding="max_length", truncation=True, max_length=max_len)
            tok["labels"] = labels
            return tok

        return _fn


def train(cfg: dict = CONFIG):
    """Entry-point facade:  setup ▸ data ▸ model ▸ tokenize ▸ build ▸ run."""
    return (
        TrainingPipeline(cfg)
        .setup_device()
        .load_data()
        .load_model()
        .tokenize_data()
        .build_trainer()
        .run()
    )


if __name__ == "__main__":
    train()