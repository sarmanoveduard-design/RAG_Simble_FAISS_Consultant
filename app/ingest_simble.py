# app/ingest_simble.py

from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from app.config import get_embeddings  # берём эмбеддинги из config.py


# === Пути ===

# Базовая папка проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# Где лежат сырые txt-файлы Simble
DATA_RAW_DIR = BASE_DIR / "data" / "raw"

# Куда сохраняем векторные базы (ASCII-путь, без корейских символов!)
VECTORES_DIR = Path.home() / "faiss_vecstores"
SIMBLE_PART1_DIR = VECTORES_DIR / "simble_part1"
SIMBLE_PART2_DIR = VECTORES_DIR / "simble_part2"


def load_text(path: Path) -> str:
    """Читает текстовый файл в UTF-8."""
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    return path.read_text(encoding="utf-8")


def make_docs(text: str, source_name: str) -> list[Document]:
    """Оборачиваем текст в Document, затем порежем на чанки."""
    doc = Document(page_content=text, metadata={"source": source_name})
    return [doc]


def split_into_chunks(docs: list[Document]) -> list[Document]:
    """Режем документы на чанки."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_faiss_for_file(input_path: Path, output_dir: Path) -> None:
    """Строим FAISS-векторную базу для одного файла и сохраняем локально."""
    print(f"\n=== Обработка файла: {input_path.name} ===")
    text = load_text(input_path)
    docs = make_docs(text, source_name=input_path.name)
    chunks = split_into_chunks(docs)
    print(f"👉 Получилось чанков: {len(chunks)}")

    embeddings = get_embeddings()

    # Гарантируем, что папка для индекса существует
    output_dir.mkdir(parents=True, exist_ok=True)

    vecstore = FAISS.from_documents(chunks, embeddings)
    vecstore.save_local(str(output_dir))

    print(f"✅ Векторная база сохранена в: {output_dir}")


def main() -> None:
    base1_path = DATA_RAW_DIR / "simble_base1.txt"
    base2_path = DATA_RAW_DIR / "simble_base2.txt"

    print(f"DATA_RAW_DIR = {DATA_RAW_DIR}")
    print(f"SIMBLE_PART1_DIR = {SIMBLE_PART1_DIR}")
    print(f"SIMBLE_PART2_DIR = {SIMBLE_PART2_DIR}")

    build_faiss_for_file(base1_path, SIMBLE_PART1_DIR)
    build_faiss_for_file(base2_path, SIMBLE_PART2_DIR)

    print("\n🎉 Готово: две отдельные FAISS-базы созданы и сохранены.")


if __name__ == "__main__":
    main()
