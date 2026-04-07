from underthesea import word_tokenize, text_normalize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import regex as re
from typing import List, Dict, Any, Union

class SpamDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: Any, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        # encoding trả về một BatchEncoding (hoạt động như Dict) chứa input_ids và attention_mask
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt', # pt = PyTorch tensors
        )

        # Trả về dictionary cho Trainer. 
        # Sử dụng .flatten() để chuyển từ shape [1, max_length] thành [max_length]
        return {
            'input_ids': encoding['input_ids'].flatten(),         # Shape: [256] (1D Tensor)
            'attention_mask': encoding['attention_mask'].flatten(), # Shape: [256] (1D Tensor)
            'labels': torch.tensor(label, dtype=torch.long)         # Shape: [] (0D Tensor - Scalar)
        }

def preprocessing(text: str) -> str:
    """Tiền xử lý chuỗi văn bản thô thành văn bản sạch."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    
    emoji_pattern = re.compile(r'\p{Emoji}', flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    text = text_normalize(text)
    text = word_tokenize(text, format="text")
    
    return text

def compute_metrics(pred: Any) -> Dict[str, float]:
    """
    Tính toán các chỉ số đánh giá model.
    Tham số `pred` (EvalPrediction) chứa:
    - pred.predictions: Ma trận (numpy array) các logits. Shape: [số_lượng_mẫu, 2]
    - pred.label_ids: Mảng (numpy array) các nhãn thực tế. Shape: [số_lượng_mẫu]
    """
    labels = pred.label_ids
    
    # Lấy index có giá trị lớn nhất trong mỗi cặp logits. 
    # Từ shape [số_lượng_mẫu, 2] -> [số_lượng_mẫu] (Chỉ chứa 0 hoặc 1)
    preds = pred.predictions.argmax(-1) 
    
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="binary")),
        "precision": float(precision_score(labels, preds, average="binary")),
        "recall": float(recall_score(labels, preds, average="binary")),
    }

def visualize_training_results(trainer: Any, test_results: Any, test_labels: List[int]) -> None:
    # Trích xuất dữ liệu từ log_history (List các Dict)
    history = trainer.state.log_history
    
    # Lọc dữ liệu train (sử dụng epoch làm trục X để đồng bộ)
    train_loss = [log['loss'] for log in history if 'loss' in log]
    train_epochs = [log['epoch'] for log in history if 'loss' in log]
    
    # Lọc dữ liệu eval
    eval_metrics = [log for log in history if 'eval_loss' in log]
    eval_epochs = [log['epoch'] for log in eval_metrics]
    eval_loss = [log['eval_loss'] for log in eval_metrics]
    eval_acc = [log['eval_accuracy'] for log in eval_metrics]
    eval_f1 = [log['eval_f1'] for log in eval_metrics]
    eval_prec = [log['eval_precision'] for log in eval_metrics]
    eval_recall = [log['eval_recall'] for log in eval_metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Biểu đồ Cost (Loss)
    ax1.plot(train_epochs, train_loss, label='Train Loss', color='blue', linestyle='--')
    ax1.plot(eval_epochs, eval_loss, label='Eval Loss', color='red', marker='o')
    ax1.set_title('Training & Evaluation Loss (Cost)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Biểu đồ Metrics
    ax2.plot(eval_epochs, eval_acc, label='Accuracy', marker='s')
    ax2.plot(eval_epochs, eval_f1, label='F1-Score', marker='^')
    ax2.plot(eval_epochs, eval_prec, label='Precision', marker='d')
    ax2.plot(eval_epochs, eval_recall, label='Recall', marker='x')
    ax2.set_title('Evaluation Metrics over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('./out/visualization/training_dashboard.png')
    print("Đã lưu biểu đồ training: ./out/visualization/training_dashboard.png")

    # Vẽ Confusion Matrix cho tập Test
    # test_results.predictions là logits. Shape: [số_mẫu_test, 2]
    # preds. Shape: [số_mẫu_test]
    preds = test_results.predictions.argmax(-1) 
    cm = confusion_matrix(test_labels, preds) # cm trả về numpy array 2x2
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix on Test Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('./out/visualization/confusion_matrix.png')
    print("Đã lưu Confusion Matrix: ./out/visualization/confusion_matrix.png")