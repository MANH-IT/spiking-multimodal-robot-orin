// map.js - Bản đồ tòa nhà 15 tầng
// Robot EEEC - Đại học Giao thông Vận tải

// ==================== DỮ LIỆU TÒA NHÀ 15 TẦNG ====================
const buildingData = {
    1: {
        name: "TẦNG 1",
        rooms: [
            { id: "101", name: "Phòng Hành chính Tổng hợp (Bộ phận Văn thư)", x: 50, y: 60, width: 160, height: 100, type: "office" },
            { id: "102", name: "Phòng Hành chính Tổng hợp", x: 250, y: 60, width: 150, height: 100, type: "office" },
            { id: "103", name: "Phòng Máy chủ", x: 450, y: 60, width: 140, height: 100, type: "server" },
            { id: "104", name: "Phòng Quản lý tòa nhà và trực PCCC", x: 640, y: 60, width: 150, height: 100, type: "office" }
        ]
    },
    2: {
        name: "TẦNG 2",
        rooms: [
            { id: "201", name: "Phòng Khánh tiết", x: 50, y: 60, width: 200, height: 120, type: "hall" },
            { id: "202", name: "Phòng Truyền thông", x: 300, y: 60, width: 180, height: 120, type: "office" },
            { id: "203", name: "Phòng Hội thảo", x: 530, y: 60, width: 200, height: 120, type: "meeting" }
        ]
    },
    3: {
        name: "TẦNG 3 - KHOA CÔNG NGHỆ THÔNG TIN VÀ KHOA ĐIỆN - ĐIỆN TỬ",
        rooms: [
            { id: "301", name: "Phòng họp", x: 30, y: 50, width: 100, height: 80, type: "meeting" },
            { id: "302", name: "Trưởng khoa Công nghệ thông tin", x: 150, y: 50, width: 120, height: 80, type: "office" },
            { id: "303", name: "Văn phòng khoa Công nghệ thông tin", x: 290, y: 50, width: 120, height: 80, type: "office" },
            { id: "304", name: "Bộ môn Điều khiển học", x: 430, y: 50, width: 100, height: 80, type: "department" },
            { id: "305", name: "Bộ môn Điều khiển học", x: 550, y: 50, width: 100, height: 80, type: "department" },
            { id: "306", name: "Bộ môn Mạng và các hệ thống thông tin", x: 670, y: 50, width: 120, height: 80, type: "department" },
            { id: "307", name: "Bộ môn Mạng và các hệ thống thông tin", x: 30, y: 160, width: 120, height: 80, type: "department" },
            { id: "308", name: "Bộ môn Khoa học máy tính", x: 170, y: 160, width: 120, height: 80, type: "department" },
            { id: "309", name: "Bộ môn Điều khiển và tự động hóa giao thông", x: 310, y: 160, width: 140, height: 80, type: "department" },
            { id: "310", name: "Bộ môn Điều khiển và tự động hóa giao thông", x: 470, y: 160, width: 100, height: 80, type: "department" },
            { id: "311", name: "Bộ môn Khoa học máy tính", x: 590, y: 160, width: 100, height: 80, type: "department" },
            { id: "312", name: "Bộ môn Công nghệ phần mềm", x: 710, y: 160, width: 80, height: 80, type: "department" }
        ]
    },
    4: {
        name: "TẦNG 4 - KHOA ĐIỆN - ĐIỆN TỬ",
        rooms: [
            { id: "401", name: "Phòng họp", x: 30, y: 50, width: 100, height: 80, type: "meeting" },
            { id: "402", name: "Bộ môn Kỹ thuật điện tử", x: 150, y: 50, width: 120, height: 80, type: "department" },
            { id: "403", name: "Trưởng khoa Điện - Điện tử", x: 290, y: 50, width: 130, height: 80, type: "office" },
            { id: "404", name: "Văn phòng khoa Điện - Điện tử", x: 440, y: 50, width: 130, height: 80, type: "office" },
            { id: "405", name: "Bộ môn Kỹ thuật thông tin", x: 590, y: 50, width: 130, height: 80, type: "department" },
            { id: "406", name: "Bộ môn Kỹ thuật điện", x: 30, y: 160, width: 110, height: 80, type: "department" },
            { id: "407", name: "Bộ môn Kỹ thuật điện", x: 160, y: 160, width: 110, height: 80, type: "department" },
            { id: "408", name: "Bộ môn Kỹ thuật viễn thông", x: 290, y: 160, width: 100, height: 80, type: "department" },
            { id: "409", name: "Bộ môn Kỹ thuật viễn thông", x: 410, y: 160, width: 100, height: 80, type: "department" },
            { id: "410", name: "Bộ môn Kỹ thuật viễn thông", x: 530, y: 160, width: 100, height: 80, type: "department" },
            { id: "411", name: "Bộ môn Kỹ thuật viễn thông", x: 650, y: 160, width: 100, height: 80, type: "department" },
            { id: "412", name: "Bộ môn Kỹ thuật điện tử", x: 770, y: 50, width: 60, height: 80, type: "department" }
        ]
    },
    5: {
        name: "TẦNG 5 - KHOA KỸ THUẬT XÂY DỰNG",
        rooms: [
            { id: "501", name: "Phòng họp", x: 30, y: 50, width: 100, height: 80, type: "meeting" },
            { id: "502", name: "Bộ môn Kỹ thuật hạ tầng đô thị", x: 150, y: 50, width: 140, height: 80, type: "department" },
            { id: "503", name: "Trưởng khoa Kỹ thuật xây dựng", x: 310, y: 50, width: 140, height: 80, type: "office" },
            { id: "504", name: "Văn phòng khoa Kỹ thuật xây dựng", x: 470, y: 50, width: 130, height: 80, type: "office" },
            { id: "505", name: "Bộ môn Vật liệu xây dựng", x: 620, y: 50, width: 110, height: 80, type: "department" },
            { id: "506", name: "Bộ môn Vật liệu xây dựng", x: 30, y: 160, width: 110, height: 80, type: "department" },
            { id: "507", name: "Bộ môn Vật liệu xây dựng", x: 160, y: 160, width: 110, height: 80, type: "department" },
            { id: "508", name: "Bộ môn Kết cấu xây dựng", x: 290, y: 160, width: 110, height: 80, type: "department" },
            { id: "509", name: "Bộ môn Kết cấu xây dựng", x: 420, y: 160, width: 110, height: 80, type: "department" },
            { id: "510", name: "Bộ môn Kết cấu xây dựng", x: 550, y: 160, width: 110, height: 80, type: "department" },
            { id: "511", name: "Bộ môn Kết cấu xây dựng", x: 680, y: 160, width: 110, height: 80, type: "department" },
            { id: "512", name: "Bộ môn Kỹ thuật hạ tầng đô thị", x: 770, y: 50, width: 60, height: 80, type: "department" }
        ]
    },
    6: {
        name: "TẦNG 6 - KHOA QUẢN LÝ XÂY DỰNG, KHOA MÔI TRƯỜNG VÀ AN TOÀN GIAO THÔNG",
        rooms: [
            { id: "601", name: "Phòng họp", x: 30, y: 50, width: 100, height: 80, type: "meeting" },
            { id: "602", name: "Trưởng khoa Quản lý xây dựng", x: 150, y: 50, width: 140, height: 80, type: "office" },
            { id: "603", name: "Văn phòng khoa Quản lý xây dựng", x: 310, y: 50, width: 140, height: 80, type: "office" },
            { id: "604", name: "Bộ môn Kinh tế xây dựng", x: 470, y: 50, width: 120, height: 80, type: "department" },
            { id: "605", name: "Bộ môn Kỹ thuật môi trường", x: 610, y: 50, width: 120, height: 80, type: "department" },
            { id: "606", name: "Bộ môn Kỹ thuật an toàn giao thông", x: 30, y: 160, width: 140, height: 80, type: "department" },
            { id: "607", name: "Văn phòng khoa Môi trường và An toàn giao thông", x: 190, y: 160, width: 150, height: 80, type: "office" },
            { id: "608", name: "Trưởng khoa Môi trường và An toàn giao thông", x: 360, y: 160, width: 140, height: 80, type: "office" },
            { id: "609", name: "Bộ môn Kinh tế xây dựng", x: 520, y: 160, width: 110, height: 80, type: "department" },
            { id: "610", name: "Bộ môn Dự án và Quản lý dự án", x: 650, y: 160, width: 120, height: 80, type: "department" },
            { id: "611", name: "Bộ môn Dự án và Quản lý dự án", x: 30, y: 270, width: 120, height: 80, type: "department" },
            { id: "612", name: "Bộ môn Kinh tế xây dựng", x: 170, y: 270, width: 120, height: 80, type: "department" }
        ]
    },
    7: {
        name: "TẦNG 7 - KHOA VẬN TẢI - KINH TẾ",
        rooms: [
            { id: "701", name: "Phòng họp", x: 30, y: 50, width: 100, height: 80, type: "meeting" },
            { id: "702", name: "Bộ môn Vận tải đường bộ và thành phố", x: 150, y: 50, width: 140, height: 80, type: "department" },
            { id: "703", name: "Bộ môn Vận tải đường bộ và thành phố", x: 310, y: 50, width: 130, height: 80, type: "department" },
            { id: "704", name: "Bộ môn Vận tải đường bộ và thành phố", x: 460, y: 50, width: 120, height: 80, type: "department" },
            { id: "705", name: "Bộ môn Kế toán - Kiểm toán", x: 600, y: 50, width: 120, height: 80, type: "department" },
            { id: "706", name: "Bộ môn Cơ sở kinh tế và quản lý", x: 30, y: 160, width: 130, height: 80, type: "department" },
            { id: "707", name: "Bộ môn Cơ sở kinh tế và quản lý", x: 180, y: 160, width: 130, height: 80, type: "department" },
            { id: "708", name: "Bộ môn Quản trị kinh doanh", x: 330, y: 160, width: 120, height: 80, type: "department" },
            { id: "709", name: "Bộ môn Quản trị kinh doanh", x: 470, y: 160, width: 120, height: 80, type: "department" },
            { id: "710", name: "Bộ môn Vận tải đường bộ và thành phố", x: 610, y: 160, width: 120, height: 80, type: "department" },
            { id: "711", name: "Bộ môn Quản trị kinh doanh", x: 30, y: 270, width: 120, height: 80, type: "department" },
            { id: "712", name: "Bộ môn Kinh tế bưu chính viễn thông", x: 170, y: 270, width: 150, height: 80, type: "department" }
        ]
    },
    8: {
        name: "TẦNG 8 - KHOA VẬN TẢI - KINH TẾ",
        rooms: [
            { id: "801", name: "Phòng họp", x: 30, y: 50, width: 100, height: 80, type: "meeting" },
            { id: "802", name: "Trưởng khoa Vận tải - Kinh tế", x: 150, y: 50, width: 140, height: 80, type: "office" },
            { id: "803", name: "Phó trưởng khoa Vận tải - Kinh tế", x: 310, y: 50, width: 140, height: 80, type: "office" },
            { id: "804", name: "Văn phòng khoa Vận tải - Kinh tế", x: 470, y: 50, width: 140, height: 80, type: "office" },
            { id: "805", name: "Bộ môn Kinh tế vận tải", x: 630, y: 50, width: 130, height: 80, type: "department" },
            { id: "806", name: "Bộ môn Quy hoạch và quản lý giao thông vận tải", x: 30, y: 160, width: 160, height: 80, type: "department" },
            { id: "807", name: "Bộ môn Kinh tế vận tải và du lịch", x: 210, y: 160, width: 150, height: 80, type: "department" },
            { id: "808", name: "Bộ môn Kinh tế vận tải và du lịch", x: 380, y: 160, width: 140, height: 80, type: "department" },
            { id: "809", name: "Bộ môn Kinh tế vận tải", x: 540, y: 160, width: 120, height: 80, type: "department" },
            { id: "810", name: "Bộ môn Vận tải đường sắt", x: 680, y: 160, width: 110, height: 80, type: "department" },
            { id: "811", name: "Bộ môn Kinh tế vận tải và du lịch", x: 30, y: 270, width: 130, height: 80, type: "department" },
            { id: "812", name: "Bộ môn Vận tải và Kinh tế đường sắt", x: 180, y: 270, width: 150, height: 80, type: "department" }
        ]
    },
    9: {
        name: "TẦNG 9 - KHOA CÔNG TRÌNH",
        rooms: [
            { id: "901", name: "Phòng họp", x: 30, y: 50, width: 100, height: 80, type: "meeting" },
            { id: "902", name: "Bộ môn Cầu hầm", x: 150, y: 50, width: 120, height: 80, type: "department" },
            { id: "903", name: "Bộ môn Cầu hầm", x: 290, y: 50, width: 120, height: 80, type: "department" },
            { id: "904", name: "Bộ môn Cầu hầm", x: 430, y: 50, width: 120, height: 80, type: "department" },
            { id: "905", name: "Bộ môn Công trình giao thông thành phố và công trình thủy", x: 570, y: 50, width: 160, height: 80, type: "department" },
            { id: "906", name: "Bộ môn Công trình giao thông thành phố và công trình thủy", x: 30, y: 160, width: 160, height: 80, type: "department" },
            { id: "907", name: "Bộ môn Công trình giao thông thành phố và công trình thủy", x: 210, y: 160, width: 140, height: 80, type: "department" },
            { id: "908", name: "Bộ môn Cầu hầm", x: 370, y: 160, width: 120, height: 80, type: "department" },
            { id: "909", name: "Bộ môn Cầu hầm", x: 510, y: 160, width: 120, height: 80, type: "department" },
            { id: "910", name: "Bộ môn Cầu hầm", x: 650, y: 160, width: 120, height: 80, type: "department" },
            { id: "911", name: "Bộ môn Cầu hầm", x: 30, y: 270, width: 130, height: 80, type: "department" },
            { id: "912", name: "Bộ môn Cầu hầm", x: 180, y: 270, width: 130, height: 80, type: "department" }
        ]
    },
    10: {
        name: "TẦNG 10 - KHOA CÔNG TRÌNH",
        rooms: [
            { id: "1001", name: "Phòng họp", x: 30, y: 50, width: 100, height: 80, type: "meeting" },
            { id: "1002", name: "Bộ môn Đường bộ", x: 150, y: 50, width: 120, height: 80, type: "department" },
            { id: "1003", name: "Bộ môn Đường bộ", x: 290, y: 50, width: 120, height: 80, type: "department" },
            { id: "1004", name: "Bộ môn Đường bộ", x: 430, y: 50, width: 120, height: 80, type: "department" },
            { id: "1005", name: "Bộ môn Đường bộ", x: 570, y: 50, width: 120, height: 80, type: "department" },
            { id: "1006", name: "Bộ môn Đường bộ", x: 30, y: 160, width: 120, height: 80, type: "department" },
            { id: "1007", name: "Bộ môn Đường ô tô và sân bay", x: 170, y: 160, width: 150, height: 80, type: "department" },
            { id: "1008", name: "Bộ môn Đường ô tô và sân bay", x: 340, y: 160, width: 140, height: 80, type: "department" },
            { id: "1009", name: "Bộ môn Đường ô tô và sân bay", x: 500, y: 160, width: 130, height: 80, type: "department" },
            { id: "1010", name: "Bộ môn Đường ô tô và sân bay", x: 650, y: 160, width: 130, height: 80, type: "department" },
            { id: "1011", name: "Bộ môn Đường ô tô và sân bay", x: 30, y: 270, width: 140, height: 80, type: "department" },
            { id: "1012", name: "Bộ môn Đường bộ", x: 190, y: 270, width: 140, height: 80, type: "department" }
        ]
    },
    11: {
        name: "TẦNG 11 - KHOA CÔNG TRÌNH",
        rooms: [
            { id: "1101", name: "Phòng họp", x: 30, y: 50, width: 100, height: 80, type: "meeting" },
            { id: "1102", name: "Bộ môn Thủy lực - Thủy văn", x: 150, y: 50, width: 140, height: 80, type: "department" },
            { id: "1103", name: "Bộ môn Công trình giao thông công chính và môi trường", x: 310, y: 50, width: 160, height: 80, type: "department" },
            { id: "1104", name: "Bộ môn Trắc địa", x: 490, y: 50, width: 120, height: 80, type: "department" },
            { id: "1105", name: "Bộ môn Trắc địa", x: 630, y: 50, width: 120, height: 80, type: "department" },
            { id: "1106", name: "Bộ môn Công trình giao thông công chính và môi trường", x: 30, y: 160, width: 150, height: 80, type: "department" },
            { id: "1107", name: "Bộ môn Kết cấu", x: 200, y: 160, width: 120, height: 80, type: "department" },
            { id: "1108", name: "Bộ môn Kết cấu", x: 340, y: 160, width: 120, height: 80, type: "department" },
            { id: "1109", name: "Bộ môn Kết cấu", x: 480, y: 160, width: 120, height: 80, type: "department" },
            { id: "1110", name: "Bộ môn Kết cấu", x: 620, y: 160, width: 120, height: 80, type: "department" },
            { id: "1111", name: "Bộ môn Kết cấu", x: 30, y: 270, width: 130, height: 80, type: "department" },
            { id: "1112", name: "Bộ môn Thủy lực - Thủy văn", x: 180, y: 270, width: 140, height: 80, type: "department" }
        ]
    },
    12: {
        name: "TẦNG 12 - KHOA CÔNG TRÌNH",
        rooms: [
            { id: "1201", name: "Phòng họp", x: 30, y: 50, width: 100, height: 80, type: "meeting" },
            { id: "1202", name: "Bộ môn Địa kỹ thuật", x: 150, y: 50, width: 130, height: 80, type: "department" },
            { id: "1203", name: "Bộ môn Đường sắt", x: 300, y: 50, width: 120, height: 80, type: "department" },
            { id: "1204", name: "Bộ môn Sức bền vật liệu", x: 440, y: 50, width: 130, height: 80, type: "department" },
            { id: "1205", name: "Bộ môn Sức bền vật liệu", x: 590, y: 50, width: 130, height: 80, type: "department" },
            { id: "1206", name: "Bộ môn Đường sắt", x: 30, y: 160, width: 120, height: 80, type: "department" },
            { id: "1207", name: "Bộ môn Đường sắt", x: 170, y: 160, width: 120, height: 80, type: "department" },
            { id: "1208", name: "Trưởng khoa Công trình", x: 310, y: 160, width: 130, height: 80, type: "office" },
            { id: "1209", name: "Phó trưởng khoa Công trình", x: 460, y: 160, width: 130, height: 80, type: "office" },
            { id: "1210", name: "Văn phòng khoa Công trình", x: 610, y: 160, width: 140, height: 80, type: "office" },
            { id: "1211", name: "Bộ môn Địa kỹ thuật", x: 30, y: 270, width: 130, height: 80, type: "department" },
            { id: "1212", name: "Bộ môn Địa kỹ thuật", x: 180, y: 270, width: 130, height: 80, type: "department" }
        ]
    },
    13: {
        name: "TẦNG 12A - KHOA CƠ KHÍ VÀ KHOA CÔNG TRÌNH",
        rooms: [
            { id: "12A1", name: "Phòng họp", x: 30, y: 50, width: 100, height: 80, type: "meeting" },
            { id: "12A2", name: "Bộ môn Cơ khí ô tô", x: 150, y: 50, width: 120, height: 80, type: "department" },
            { id: "12A3", name: "Bộ môn Cơ khí ô tô", x: 290, y: 50, width: 120, height: 80, type: "department" },
            { id: "12A4", name: "Bộ môn Cơ khí ô tô", x: 430, y: 50, width: 120, height: 80, type: "department" },
            { id: "12A5", name: "Bộ môn Cơ khí ô tô", x: 570, y: 50, width: 120, height: 80, type: "department" },
            { id: "12A6", name: "Bộ môn Tự động hóa thiết kế cầu đường", x: 30, y: 160, width: 160, height: 80, type: "department" },
            { id: "12A7", name: "Bộ môn Công nghệ giao thông", x: 210, y: 160, width: 140, height: 80, type: "department" },
            { id: "12A8", name: "Bộ môn Công nghệ giao thông", x: 370, y: 160, width: 130, height: 80, type: "department" },
            { id: "12A9", name: "Bộ môn Công nghệ giao thông", x: 520, y: 160, width: 130, height: 80, type: "department" },
            { id: "12A10", name: "Bộ môn Công nghệ giao thông", x: 670, y: 160, width: 120, height: 80, type: "department" },
            { id: "12A11", name: "Bộ môn Máy động lực", x: 30, y: 270, width: 130, height: 80, type: "department" },
            { id: "12A12", name: "Bộ môn Kỹ thuật nhiệt", x: 180, y: 270, width: 130, height: 80, type: "department" }
        ]
    },
    14: {
        name: "TẦNG 14 - KHOA CƠ KHÍ",
        rooms: [
            { id: "1401", name: "Phòng họp", x: 30, y: 50, width: 100, height: 80, type: "meeting" },
            { id: "1402", name: "Trưởng khoa Cơ khí", x: 150, y: 50, width: 140, height: 80, type: "office" },
            { id: "1403", name: "Phó trưởng khoa Cơ khí", x: 310, y: 50, width: 140, height: 80, type: "office" },
            { id: "1404", name: "Văn phòng khoa Cơ khí", x: 470, y: 50, width: 140, height: 80, type: "office" },
            { id: "1405", name: "Bộ môn Thiết kế máy", x: 630, y: 50, width: 130, height: 80, type: "department" },
            { id: "1406", name: "Bộ môn Đầu máy toa xe", x: 30, y: 160, width: 130, height: 80, type: "department" },
            { id: "1407", name: "Bộ môn Đầu máy toa xe", x: 180, y: 160, width: 130, height: 80, type: "department" },
            { id: "1408", name: "Bộ môn Máy xây dựng", x: 330, y: 160, width: 130, height: 80, type: "department" },
            { id: "1409", name: "Bộ môn Máy xây dựng", x: 480, y: 160, width: 130, height: 80, type: "department" },
            { id: "1410", name: "Bộ môn Máy xây dựng", x: 630, y: 160, width: 130, height: 80, type: "department" },
            { id: "1411", name: "Bộ môn Máy xây dựng", x: 30, y: 270, width: 130, height: 80, type: "department" },
            { id: "1412", name: "Bộ môn Cơ điện tử", x: 180, y: 270, width: 130, height: 80, type: "department" }
        ]
    },
    15: {
        name: "TẦNG 15 - KHOA LÝ LUẬN CHÍNH TRỊ",
        rooms: [
            { id: "1501", name: "Phòng họp", x: 30, y: 50, width: 100, height: 80, type: "meeting" },
            { id: "1502", name: "Nhóm nghiên cứu phương tiện giao thông thông minh", x: 150, y: 50, width: 180, height: 80, type: "research" },
            { id: "1503", name: "Nhóm nghiên cứu AI - SHM", x: 350, y: 50, width: 160, height: 80, type: "research" },
            { id: "1504", name: "Bộ môn Khoa học Mác - Lênin", x: 530, y: 50, width: 130, height: 80, type: "department" },
            { id: "1505", name: "Bộ môn Khoa học Mác - Lênin", x: 680, y: 50, width: 110, height: 80, type: "department" },
            { id: "1506", name: "Bộ môn Lịch sử Đảng Cộng sản Việt Nam", x: 30, y: 160, width: 160, height: 80, type: "department" },
            { id: "1507", name: "Văn phòng khoa Lý luận chính trị", x: 210, y: 160, width: 150, height: 80, type: "office" },
            { id: "1508", name: "Trưởng khoa Lý luận chính trị", x: 380, y: 160, width: 140, height: 80, type: "office" },
            { id: "1509", name: "Bộ môn Tư tưởng Hồ Chí Minh", x: 540, y: 160, width: 130, height: 80, type: "department" },
            { id: "1510", name: "Bộ môn Tư tưởng Hồ Chí Minh", x: 690, y: 160, width: 100, height: 80, type: "department" }
        ]
    }
};

// ==================== HÀM TÌM KIẾM PHÒNG ====================
function searchRoomById(roomId) {
    for (let floor = 1; floor <= 15; floor++) {
        const floorData = buildingData[floor];
        if (!floorData) continue;

        const room = floorData.rooms.find(r => r.id === roomId);
        if (room) {
            return { floor, room };
        }
    }
    return null;
}

function getAllRooms() {
    const allRooms = [];
    for (let floor = 1; floor <= 15; floor++) {
        const floorData = buildingData[floor];
        if (floorData) {
            floorData.rooms.forEach(room => {
                allRooms.push({
                    ...room,
                    floor: floor,
                    floorName: floorData.name
                });
            });
        }
    }
    return allRooms;
}

// Export cho các module khác
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { buildingData, searchRoomById, getAllRooms };
}