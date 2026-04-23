// Demo scenario cho hội đồng
const DEMO_SCENARIO = {
    step1: {
        trigger: "person_detected",
        robot_action: "greeting",
        message: "👋 Xin chào! Tôi thấy bạn đang đến gần. Tôi có thể giúp gì cho bạn?"
    },
    step2: {
        user_question: "Tôi muốn tìm phòng thí nghiệm",
        robot_response: "🔬 Phòng thí nghiệm nằm ở tầng 2, khu A.",
        robot_followup: "Bạn có muốn tôi dẫn đường đến đó không?"
    },
    step3: {
        user_answer: "Có, cảm ơn bạn",
        robot_action: "navigate",
        message: "📡 Hãy đi theo tôi! Phòng thí nghiệm cách đây 50 mét về phía bên phải."
    }
};
