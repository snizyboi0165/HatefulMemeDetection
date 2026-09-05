# Phân loại Hateful Meme (Nhóm 07)

Đây là đồ án cuối kỳ môn Học Sâu (Deep Learning) - Trường Đại học Công nghiệp TP.HCM (IUH).
Dự án tập trung vào việc giải quyết bài toán phân loại Meme (Hateful Meme / Memotion) sử dụng các mô hình học sâu.

## 📌 Tổng quan dự án (Project Overview)
- **Mục tiêu:** Phân loại và nhận diện các Hateful Meme.
- **Dataset:** Sử dụng tập dữ liệu Hateful Meme và Memotion Dataset (Do kích thước lớn, dataset không được upload lên repo này).
- **Mô hình (Model):** Tích hợp các mô hình tiên tiến như Qwen2-VL, Fine-tuning với PEFT (LoRA).

## 🗂 Cấu trúc thư mục (Directory Structure)
- `build_merged_dataset.py`: Script dùng để chuẩn bị và merge các dataset.
- `dataset_statistics.ipynb`: Notebook thống kê, trực quan hóa phân phối của tập dữ liệu.
- `predict_personal_images_v6.ipynb`: Notebook hỗ trợ test dự đoán trên các hình ảnh cá nhân / thực tế.
- `sumary/`: Chứa các notebook và hình ảnh tổng hợp kết quả (biểu đồ ROC, độ chính xác, v.v.).
- `Nhom07_BaoCaoCuoiKy.pptx` & `Nhom7_BaoCaoCuoiKy.docx`: Báo cáo chi tiết và slide thuyết trình của nhóm.

## 🚀 Cài đặt (Installation)
Cài đặt các thư viện cần thiết thông qua `requirements.txt`:
```bash
pip install -r requirements.txt
```

## 📊 Một số kết quả (Results)
*(Bạn có thể thay thế các link ảnh dưới đây bằng các ảnh thực tế trong thư mục `sumary/` của bạn)*

![Biểu đồ ROC](sumary/merged_roc.png)
![Kết quả Benchmark](sumary/merged_fig2_benchmark.png)

## 👥 Thành viên nhóm (Team Members)
- Thành viên 1 - MSSV
- Thành viên 2 - MSSV
- Thành viên 3 - MSSV
