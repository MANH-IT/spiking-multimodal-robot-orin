# Legacy Vision Files - Lưu để tham khảo
========================================
Các file trong thư mục này là phiên bản cũ hoặc file debug tạm thời trong quá trình nâng cấp hệ thống Vision Robot EEEC (NCKH 2026).

### Danh sách thay thế:
- `data_collector.py` (Cũ) → Đã thay bằng `vision_system/enhanced_data_collector.py` (Mới, hỗ trợ vẽ box chuột).
- `_test_vision.py` (Debug) → Đã thay bằng `test_real_vision.py` (Test hệ thống hoàn chỉnh).
- `losses.py` (Logic cũ) → Đã được tích hợp trực tiếp vào `train_vision_snn.py` hoặc refactor lại cho phù hợp multi-anchor.

*Ngày dọn dẹp: 25/04/2026*
