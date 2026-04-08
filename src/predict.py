import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from custom_utils import preprocessing
from typing import Dict, Union, List

class SpamClassifier:
    def __init__(self, model_path: str, label_map: Dict[str, int] = None, use_word_segmentation: bool = True):
        self.use_word_segmentation = use_word_segmentation
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                                   else "cpu")
        # label_map: {"ham": 0, "spam": 1}  →  id_to_label: {0: "HAM", 1: "SPAM"}
        if label_map is None:
            label_map = {"ham": 0, "spam": 1}
        self.id_to_label = {v: k.upper() for k, v in label_map.items()}

        print(f"Loading model from {model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Dict[str, Union[str, float, List[float]]]:
        # 1. Tiền xử lý -> Trả về chuỗi str
        clean_text = preprocessing(text, use_word_segmentation=self.use_word_segmentation)
        
        # 2. Tokenize
        # inputs là một Dictionary chứa 2 Tensors: 'input_ids' và 'attention_mask'
        # Do ta chỉ dự đoán 1 câu (batch_size=1), kích thước của mỗi Tensor sẽ là [1, độ_dài_chuỗi_sau_tokenize]
        # Ví dụ: Shape: [1, 15] nếu câu có 15 tokens. Tối đa là [1, 256].
        inputs = self.tokenizer(
            clean_text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=256
        ).to(self.device)
        
        # 3. Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 4. Xử lý kết quả
        # logits là điểm số thô chưa qua hàm kích hoạt. 
        # Shape: [1, 2] (vì batch_size=1 và num_labels=2)
        logits = outputs.logits 
        
        # Softmax chuyển logits thành xác suất (tổng bằng 1). Shape: [1, 2]
        # Ví dụ: tensor([[0.05, 0.95]])
        probs = torch.nn.functional.softmax(logits, dim=-1) 
        
        # torch.max trả về 2 giá trị:
        # - conf: Giá trị xác suất cao nhất. Shape: [1] (vd: tensor([0.95]))
        # - pred: Vị trí (index) của xác suất cao nhất (0 hoặc 1). Shape: [1] (vd: tensor([1]))
        conf, pred = torch.max(probs, dim=-1)
        
        label = self.id_to_label.get(pred.item(), str(pred.item()))
        
        return {
            "label": label,
            "confidence": conf.item(),
            "probabilities": probs.cpu().numpy().tolist()[0]
        }

if __name__ == "__main__":
    model = SpamClassifier("./models/phobert_spam_final")
    
    test_email = "Chúc mừng bạn! Bạn nhận được quà tặng 500k từ Shopee. Click vào link để nhận ngay: http://rác.vn"
    result = model.predict(test_email)
    
    label_names = list(model.id_to_label.values())
    print(f"\nNội dung: {test_email}")
    print(f"Kết quả: {result['label']} (Độ tin cậy: {result['confidence']:.2%})")
    print(f"Phân phối xác suất {label_names}: {result['probabilities']}")