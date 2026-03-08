# -*- coding: utf-8 -*-
"""
Cấu hình giao diện màn hình cảm ứng - Robot ĐH Giao thông Vận tải
"""

# Kích thước màn hình 10 inch
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 800

# Màu sắc theo logo trường (xanh - trắng)
COLORS = {
    "primary": "#006633",      # Xanh lá đậm (UTT)
    "primary_light": "#008844",
    "secondary": "#FFFFFF",    # Trắng
    "accent": "#00AA55",       # Xanh lá nhạt
    "background": "#F5F9F6",  # Nền xanh nhạt
    "text_dark": "#1A3D2E",
    "text_light": "#FFFFFF",
    "chat_user_bg": "#E8F5E9",
    "chat_robot_bg": "#FFFFFF",
    "border": "#C8E6C9",
}

# Font - lớn, dễ đọc từ 1m
FONTS = {
    "header": 18,
    "title": 16,
    "body": 14,
    "small": 12,
    "button": 14,
}

# Dữ liệu mẫu
SAMPLE_USERS = {
    "20241234": {
        "name": "Nguyễn Văn A",
        "mssv": "20241234",
        "faculty": "Khoa CNTT",
        "avatar": None,
    },
    "default": {
        "name": "Khách",
        "mssv": "---",
        "faculty": "---",
        "avatar": None,
    },
}

# Câu hỏi và trả lời mẫu
SAMPLE_QA = {
    "Học phí đóng muộn thì làm sao?": {
        "answer": "Nộp tại phòng A.101, phí phạt 50.000đ/tuần. Mang theo CMND và thẻ sinh viên. Giờ làm việc: 8h-11h30, 13h30-16h30 các ngày trong tuần.",
        "image": None,
        "room": "A.101",
    },
    "Học phí đóng muộn?": {
        "answer": "Nộp tại phòng A.101, phí phạt 50.000đ/tuần. Mang theo CMND và thẻ sinh viên.",
        "image": None,
        "room": "A.101",
    },
    "Lịch học hôm nay": {
        "answer": "Bạn có 3 môn: Toán cao cấp (8h-10h, phòng B.201), Lập trình (10h30-12h, phòng C.102), Mạng máy tính (14h-16h, phòng A.305).",
        "image": None,
        "room": "B.201",
    },
    "Phòng thi ở đâu?": {
        "answer": "Phòng thi được thông báo trên cổng thông tin sinh viên. Kiểm tra tại phòng Đào tạo A.102 hoặc website utt.edu.vn.",
        "image": None,
        "room": "A.102",
    },
    "Đăng ký môn học": {
        "answer": "Đăng ký qua cổng thông tin sinh viên hoặc đến phòng Đào tạo A.102. Thời gian: tuần 1-2 của học kỳ.",
        "image": None,
        "room": "A.102",
    },
    "Hướng dẫn thủ tục": {
        "answer": "Các thủ tục hành chính: xem tại phòng Công tác Sinh viên A.101. Hỗ trợ trực tuyến: utt.edu.vn/sinhvien.",
        "image": None,
        "room": "A.101",
    },
    "Liên hệ phòng ban": {
        "answer": "Phòng Đào tạo: A.102 | Phòng CTSV: A.101 | Phòng Kế toán: B.105. Hotline: 024.3767.xxxx.",
        "image": None,
        "room": None,
    },
}

# Câu gợi ý nhanh
QUICK_SUGGESTIONS = [
    "Học phí đóng muộn?",
    "Lịch học hôm nay",
    "Phòng thi ở đâu?",
    "Đăng ký môn học",
    "Hướng dẫn thủ tục",
    "Liên hệ phòng ban",
]
