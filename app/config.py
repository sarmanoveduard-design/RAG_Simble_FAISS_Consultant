from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

# Базовая папка проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# Где лежат сырые txt-файлы Simble
DATA_RAW_DIR = BASE_DIR / "data" / "raw"

# Модели
CHAT_MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# Загружаем .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==== Пути к векторным базам Simble ====
VECSTORES_DIR = BASE_DIR / "vecstores"

SIMBLE_PART1_VS_DIR = VECSTORES_DIR / "simble_part1"
SIMBLE_PART2_VS_DIR = VECSTORES_DIR / "simble_part2"
SIMBLE_MERGED_VS_DIR = VECSTORES_DIR / "simble_merged"


def get_llm() -> ChatOpenAI:
    """LLM для нашего консультанта."""
    return ChatOpenAI(
        model=CHAT_MODEL_NAME,
        api_key=OPENAI_API_KEY,
        temperature=0.1,
    )


def get_embeddings() -> OpenAIEmbeddings:
    """Эмбеддинги для FAISS."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        api_key=OPENAI_API_KEY,
    )
