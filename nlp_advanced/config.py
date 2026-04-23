# Cấu hình dữ liệu
DATA_ROOT = "D:/robot_eeec/data"
TRAINING_DATA = f"{DATA_ROOT}/merged_training_data.json"
UTC_KNOWLEDGE = f"{DATA_ROOT}/utc_knowledge.json"
BUILDING_DATA = f"{DATA_ROOT}/building_data.json"

# Dữ liệu dependency parsing (cho Advanced NLU)
DEP_TRAINING_DATA = "D:/robot_eeec/nlp_advanced/data/dep_training.json"

# Tham số mô hình
T = 20
MAX_SENT_LEN = 20
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 5
NUM_DEP_LABELS = 15
