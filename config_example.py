import os

# ================================
# LLM 설정
# ================================
LLM_MODEL = "qwen2.5:7b"
LLM_HOST = "http://localhost:11434"  # Ollama 서버 주소

# ================================
# Neo4j 설정
# ================================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password_here"  # 변경 필요

# ================================
# 데이터 설정
# ================================
DATA_DIR = os.path.join("data", "raw")
OUTPUT_DIR = os.path.join("results", "extracted")
SAMPLE_SIZE = 500
MAX_FILES = 10

# ================================
# 모듈 설정
# ================================
PROMPT_VERSION = "v12_clean_examples"
FUSION_SIMILARITY_THRESHOLD = 0.85
SV_ENABLED = False
SV_CONFIDENCE_THRESHOLD = 0.9
POSTPROCESS_ENABLED = True

# ================================
# OpenAI API 설정 (선택적)
# ================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")   