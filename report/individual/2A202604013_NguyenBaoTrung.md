# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202604013
- **Họ và tên:** Nguyễn Bảo Trung
- **SEED** (in ở ô đầu notebook): 5046

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | Có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | Có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | Có |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.987   0.798      0.882        0.919   
2. BoW + MLP (deep learning)         0.988   0.894      0.939        0.974   
3a. Encoder (MiniLM) + kNN           0.788   0.947      0.860        0.974   
3b. Encoder (MiniLM) + LogReg        0.989   0.947      0.967        0.974   
3c. Encoder (MiniLM) + MLP head      0.989   0.947      0.967        0.974   
4a. LLM nhỏ zero-shot                1.000   1.000      1.000        1.000   
4b. LLM lớn zero-shot                1.000   0.950      0.974        0.974   
4c. LLM nhỏ few-shot                 0.952   1.000      0.976        0.976   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            1.109      44581.958       0.030   
2. BoW + MLP (deep learning)         2.985      63540.470       0.205   
3a. Encoder (MiniLM) + kNN           1.116        332.554      11.796   
3b. Encoder (MiniLM) + LogReg        1.225       2809.962       6.951   
3c. Encoder (MiniLM) + MLP head      2.156       3440.014       7.333   
4a. LLM nhỏ zero-shot                0.000          1.063     913.272   
4b. LLM lớn zero-shot                0.000          0.299    3135.389   
4c. LLM nhỏ few-shot                 0.000          1.134     845.836   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.042         0.000  
2. BoW + MLP (deep learning)          0.235         0.000  
3a. Encoder (MiniLM) + kNN           13.445         0.000  
3b. Encoder (MiniLM) + LogReg         7.208         0.000  
3c. Encoder (MiniLM) + MLP head       7.580         0.000  
4a. LLM nhỏ zero-shot              1296.676         0.026  
4b. LLM lớn zero-shot              5654.445         0.062  
4c. LLM nhỏ few-shot               1249.581         0.121  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.


## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*
🎯 Đề #1: ML · Đổi ngưỡng quyết định 0.5 → 0.2 và → 0.8 cho mô hình Phần 1. Báo precision/recall/F1 từng ngưỡng. Một ngân hàng sợ chặn nhầm tin OTP thì nên chọn ngưỡng nào? Vì sao?
def challenge(decision_threshold: int) -> None:
  predict_bow_lr = lambda texts: (predict_proba(vectorize(texts, vocab), w, b) >= decision_threshold).astype(int)   # văn bản thô → nhãn
  pred_te, bench = benchmark(predict_bow_lr, texts_te, texts_te[SUB])
  m_tr = metrics(y_tr, (predict_proba(X_tr, w, b) >= decision_threshold).astype(int)); m_te = metrics(y_te, pred_te)
  print(f"Ngưỡng {decision_threshold}")
  print("TRAIN:", {k: round(v, 3) for k, v in m_tr.items()})
  print("TEST :", {k: round(v, 3) for k, v in m_te.items()})

challenge(0.2)
challenge(0.8)

Ngưỡng 0.2
TRAIN: {'accuracy': 0.866, 'precision': 0.473, 'recall': 0.484, 'f1': 0.479}
TEST : {'accuracy': 0.849, 'precision': 0.376, 'recall': 0.372, 'f1': 0.374}
Ngưỡng 0.8
TRAIN: {'accuracy': 0.874, 'precision': 0.857, 'recall': 0.013, 'f1': 0.026}
TEST : {'accuracy': 0.879, 'precision': nan, 'recall': 0.0, 'f1': nan}
/tmp/ipykernel_68331/3234187816.py:17: RuntimeWarning: invalid value encountered in scalar divide
  precision = TP / (TP + FP)
 Sợ chặn nhầm tin OTP thì phải tăng ngưỡng -> chọn 0.8 -> để tránh tin OTP bị gán nhãn spam(1)

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
BoW + MLP, BoW + LogReg đạt yêu cầu độ trễ + nhưng chỉ BoW + MLP đạt cả yêu cầu về độ trễ và chất lượng. Dùng LLM model nhỏ zẻo shot hết 0.026 cho 1k tin -> tổg 260 USD/ngàyzero-shot hết 0.026 cho 1k tin -> tổg 260 USD/ngày
Dùng BoW + MLP cho tin đơn giản , LLM few shot cho tin lưỡng lự -> đảm bảo ko bỏ sót các tin có nguye cơ
## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
|4|Hỏi cách call LLM từ API | AI đã gợi ý đoạn code |Đoạn code có sai + không hoạt động -> kiểm tra và thử lại vài vòng cho tới khi hoạt động |
|5|Hỏi về các định nghĩa chưa quen thuộc | AI giải đáp và gợi ý |Phân tích và trả lời dựa trên gợi ý |
|Notebook|Hỏi cách fix các lỗi (Vd: Please explain this error: ....) | AI giải đáp và gợi ý |Phân tích và fix lỗi |
