# Multimodal Hateful Meme Detection using Vision-Language Models

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-magenta.svg?style=for-the-badge)](https://github.com/huggingface/peft)
[![CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76B900.svg?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)

> **Nghiên cứu và triển khai hệ thống học sâu đa phương thức (Multimodal Deep Learning) nhằm phát hiện và phân loại nội dung thù ghét (Hate Speech) trong meme mạng xã hội, ứng dụng mô hình Vision-Language (Qwen2-VL), kỹ thuật tinh chỉnh tham số hiệu quả PEFT/LoRA và hàm mất mát Asymmetric Loss (ASL).**

---

## Tổng quan dự án

Meme mạng xã hội là hình thức truyền thông đa phương thức kết hợp giữa hình ảnh trực quan và văn bản lồng ghép. Thách thức lớn nhất của bài toán phát hiện Hateful Meme nằm ở chỗ:
1. **Tính đa phương thức phụ thuộc (Multimodal Dependence):** Một hình ảnh bình thường hoặc một câu chữ bình thường khi đứng riêng lẻ hoàn toàn vô hại, nhưng khi kết hợp lại có thể mang ý đồ xúc phạm, thù ghét hoặc kích động bạo lực.
2. **Ngữ cảnh và sự châm biếm:** Meme thường ẩn chứa hàm ý châm biếm (sarcasm), định kiến xã hội, tôn giáo hoặc chủng tộc đòi hỏi mô hình phải có năng lực suy luận ngôn ngữ - hình ảnh cấp cao.

Dự án tiếp cận bài toán thông qua việc khai thác các mô hình nền tảng Thị giác - Ngôn ngữ (Vision-Language Foundation Models) tiên tiến, áp dụng kỹ thuật tinh chỉnh PEFT/LoRA kết hợp hàm mất mát Asymmetric Loss và tăng cường dữ liệu khi suy luận (Test-Time Augmentation - TTA) để tối ưu độ chính xác và khả năng tổng quát hóa.

---

## Phương pháp tiếp cận & Kiến trúc kỹ thuật

Hệ thống được phát triển qua nhiều giai đoạn nghiên cứu từ mô hình cơ sở đến tối ưu nâng cao:

### 1. Tập dữ liệu nghiên cứu (Datasets)
- **Facebook Hateful Meme Challenge:** Bộ dữ liệu chuẩn quốc tế với các mẫu meme đa dạng, tập trung vào các cặp đối chiếu khó (minimally contrastive pairs).
- **Memotion Dataset 7k:** Tập dữ liệu meme thực tế phân loại theo cảm xúc và mức độ công kích.
- **Unified Merged Dataset:** Quy trình tiền xử lý và chuẩn hóa dữ liệu từ `build_merged_dataset.py` gom nhóm và tái cấu trúc thành một tập dữ liệu đồng nhất.

### 2. Mô hình nền tảng & Fine-tuning
- **Baseline Models (Frozen):** Đánh giá hiệu năng zero-shot và trích xuất đặc trưng của BLIP-base ITM, CLIP ViT-L/14, SigLIP SO400M và Qwen2-VL-2B nguyên bản.
- **Mô hình đề xuất (Proposed Fine-tuned Model):** 
  - Tinh chỉnh mô hình Vision-Language **Qwen2-VL** với kiến trúc **LoRA (Low-Rank Adaptation)** giúp giảm thiểu tham số cần huấn luyện mà vẫn giữ được tri thức tiền huấn luyện của mô hình lớn.
  - Sử dụng **BitsAndBytes (4-bit/8-bit Quantization)** để tối ưu dung lượng VRAM GPU.
  - Tích hợp **Asymmetric Loss (ASL)** giải quyết triệt để tình trạng mất cân bằng dữ liệu nghiêm trọng giữa nhãn Hateful và Non-Hateful.
  - Áp dụng **MultiScale Visual Processing** và **Test-Time Augmentation (TTA)** trong pha suy luận.

---

## Kết quả thực nghiệm & Đánh giá (Benchmark)

Các mô hình được đánh giá trên độ đo tiêu chuẩn diện tích dưới đường cong ROC (AUROC):

| STT | Mô hình | Phương pháp | FB Hateful Meme (AUROC) | Merged Dataset (AUROC) | Độ chênh lệch (Delta) |
|:---:|---|---|:---:|:---:|:---:|
| 1 | BLIP-base ITM | Zero-shot (Frozen) | 0.5286 | 0.5114 | -1.72% |
| 2 | CLIP ViT-L/14 | Feature Extractor | 0.5859 | 0.6021 | +1.62% |
| 3 | SigLIP SO400M | Feature Extractor | 0.6088 | 0.6160 | +0.72% |
| 4 | Qwen2-VL-2B | Zero-shot (Frozen) | 0.7178 | 0.6965 | -2.13% |
| 5 | **v6 MultiScale + ASL + TTA (Test)** | **Fine-tuned (LoRA)** | **0.7646** | **0.7490** | -1.56% |
| 6 | **v6 MultiScale + ASL + TTA (Dev)** | **Fine-tuned (LoRA)** | **0.8539** | **0.7840** | -6.99% |

Mô hình tinh chỉnh đạt hiệu suất vượt trội với **AUROC đạt 0.8539** trên tập kiểm thử nội bộ, cải thiện đáng kể khả năng phân biệt nội dung mỉa mai và độc hại so với các mô hình baseline truyền thống.

---

## Cấu trúc thư mục dự án

```
HatefulMemeDetection/
├── build_merged_dataset.py          # Script làm sạch, hợp nhất và gán nhãn tập dữ liệu
├── dataset_statistics.ipynb         # Khám phá và trực quan hóa phân bố dữ liệu (EDA)
├── predict_personal_images_v6.ipynb # Notebook kiểm thử suy luận trên ảnh thực tế
├── sumary/                          # Tổng hợp kết quả, biểu đồ ROC và bảng đánh giá
│   ├── merged_roc.png               # Đường cong ROC so sánh các phiên bản
│   ├── merged_fig2_benchmark.png    # Biểu đồ cột so sánh Benchmark
│   ├── unified_comparison_table.csv # Bảng tổng hợp số liệu chi tiết
│   └── summary.ipynb                # Phân tích tổng thể kết quả thí nghiệm
├── requirements.txt                 # Danh sách thư viện phụ thuộc
└── README.md                        # Tài liệu dự án
```

---

## Hướng dẫn cài đặt & Thực thi

### 1. Yêu cầu môi trường
- Python 3.10 trở lên
- Khuyến nghị GPU NVIDIA hỗ trợ CUDA với VRAM từ 12GB trở lên (RTX 3060/3080/4090 hoặc Google Colab T4/A100).

### 2. Cài đặt các thư viện phụ thuộc
```bash
git clone https://github.com/snizyboi0165/HatefulMemeDetection.git
cd HatefulMemeDetection
pip install -r requirements.txt
```

### 3. Khám phá dữ liệu & Đánh giá
- Mở `dataset_statistics.ipynb` bằng Jupyter Notebook để xem các biểu đồ phân tích phân phối độ dài văn bản và nhãn dữ liệu.
- Mở `sumary/summary.ipynb` để theo dõi tiến trình huấn luyện, biểu đồ suy giảm hàm mất mát và phân tích các trường hợp dự đoán sai (Error Analysis).

### 4. Thử nghiệm suy luận với ảnh tùy chọn
Chạy notebook `predict_personal_images_v6.ipynb` để tải mô hình và dự đoán xác suất độc hại của một bức ảnh meme bất kỳ từ máy tính.
