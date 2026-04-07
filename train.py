import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
from custom_utils import SpamDataset, preprocessing, compute_metrics, visualize_training_results

# Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model & Tokenizer
model_path = "./models/phobert_v2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=2,
    problem_type="single_label_classification",
).to(device)

print("Loading and preprocessing data...")
# data trả về kiểu pd.DataFrame
data = pd.read_csv("./dataset/vi_dataset.csv")

# Tiền xử lý text (Nên thực hiện trước khi split)
data['text'] = data['text'].apply(preprocessing)
data['label'] = data['label'].apply(lambda x: 1 if x == 'spam' else 0)  # Chuyển nhãn thành số (0 hoặc 1)

# Split dữ liệu
# train_texts, temp_texts là List[str]
# train_labels, temp_labels là List[int] (0 hoặc 1)
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    data['text'].tolist(), data['label'].tolist(), test_size=0.2, random_state=42
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42
)
print(f"Train size: {len(train_texts)}, Validation size: {len(val_texts)}, Test size: {len(test_texts)}")

print("Preparing for training...")
# Tạo Dataset objects
train_dataset = SpamDataset(train_texts, train_labels, tokenizer)
val_dataset = SpamDataset(val_texts, val_labels, tokenizer)

# Training Arguments
trainer_args = TrainingArguments(
    output_dir="./out/checkpoints",
    num_train_epochs=5,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=24,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=torch.cuda.is_available(),
    weight_decay=0.01,
    report_to="none",
    dataloader_pin_memory=True
)

# Trainer tự động quản lý việc đóng gói dicts từ Dataset thành Batches (Tensor)
trainer = Trainer(
    model=model,
    args=trainer_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

print("Evaluating on test set...")
# test_dataset trả ra các Dict tương tự train_dataset
test_dataset = SpamDataset(test_texts, test_labels, tokenizer)

# test_results là đối tượng PredictionOutput chứa:
# - predictions: Shape [tổng_số_mẫu_test, 2]
# - label_ids: Shape [tổng_số_mẫu_test]
# - metrics: Dict chứa các chỉ số
test_results = trainer.predict(test_dataset)
print(f"Test Metrics: {test_results.metrics}")

visualize_training_results(trainer, test_results, test_labels)

trainer.save_model("./models/phobert_spam_final")
tokenizer.save_pretrained("./models/phobert_spam_final")