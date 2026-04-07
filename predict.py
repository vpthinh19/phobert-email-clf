import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from custom_utils import preprocessing
from typing import Dict, Union, List

class SpamClassifier:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval() # Bật chế độ đánh giá (tắt Dropout)

    def predict(self, text: str) -> Dict[str, Union[str, float, List[float]]]:
        # 1. Tiền xử lý -> Trả về chuỗi str
        clean_text = preprocessing(text)
        
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
        
        label = "SPAM" if pred.item() == 1 else "HAM"
        
        return {
            "label": label,
            "confidence": conf.item(), # .item() chuyển từ Tensor 0D về kiểu float chuẩn của Python
            "probabilities": probs.cpu().numpy().tolist()[0] # Chuyển tensor [1, 2] thành List float [p_ham, p_spam]
        }

if __name__ == "__main__":
    model = SpamClassifier("./models/phobert_spam_final")
    
    test_email = "Chúc mừng bạn! Bạn nhận được quà tặng 500k từ Shopee. Click vào link để nhận ngay: http://rác.vn"
    result = model.predict(test_email)
    
    print(f"\nNội dung: {test_email}")
    print(f"Kết quả: {result['label']} (Độ tin cậy: {result['confidence']:.2%})")
    print(f"Phân phối xác suất [Ham, Spam]: {result['probabilities']}")