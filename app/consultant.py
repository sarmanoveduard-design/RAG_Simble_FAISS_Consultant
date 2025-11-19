"""
Простой консалт по базе Simble (FAISS + LangChain).

Что делает:
- Загружает объединённый vecstore (simble_merged).
- Делает retriever.
- Оборачивает его в RetrievalQA.
- Отвечает на один вопрос пользователя.
"""

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from app.config import (
    get_llm,
    get_embeddings,
    SIMBLE_MERGED_VS_DIR,
)


def build_qa_chain() -> RetrievalQA:
    """Создаёт RetrievalQA поверх объединённой FAISS-базы."""
    embeddings = get_embeddings()

    print(
        f"▶ Загружаю объединённый vecstore из: {SIMBLE_MERGED_VS_DIR}"
    )
    vectordb = FAISS.load_local(
        str(SIMBLE_MERGED_VS_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    llm = get_llm()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa


def ask_once() -> None:
    """Один вопрос к консультанту (для проверки ДЗ)."""
    qa = build_qa_chain()

    print("\n✅ Консультант по базе Simble готов.")
    print(
        "Введите вопрос (или оставьте пустым, чтобы "
        "использовать тестовый):"
    )
    user_q = input("> ").strip()

    if not user_q:
        user_q = (
            "Как оформить заявку на подключение услуги "
            "Simble для корпоративного клиента?"
        )

    print(f"\n❓ Вопрос: {user_q}\n")

    result = qa({"query": user_q})

    answer = result.get("result", "").strip()
    sources = result.get("source_documents", [])

    print("💬 Ответ:\n")
    print(answer)
    print("\n📚 Использованные чанки:")

    for i, doc in enumerate(sources, start=1):
        chunk_id = doc.metadata.get("chunk", "?")
        src = doc.metadata.get("source", "N/A")
        print(f"- chunk={chunk_id}, source={src}")


if __name__ == "__main__":
    ask_once()
