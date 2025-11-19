"""
Объединение двух локальных FAISS-баз Simble в одну.

По ДЗ:
- берём две готовые FAISS-базы (часть 1 и часть 2),
- объединяем их,
- сохраняем итоговый vecstore локально.
"""

from langchain_community.vectorstores import FAISS

from app.config import (
    SIMBLE_PART1_VS_DIR,
    SIMBLE_PART2_VS_DIR,
    SIMBLE_MERGED_VS_DIR,
    get_embeddings,
)


def main() -> None:
    embeddings = get_embeddings()

    print(f"▶ Загружаю базу ЧАСТЬ 1 из: {SIMBLE_PART1_VS_DIR}")
    vs_part1 = FAISS.load_local(
        str(SIMBLE_PART1_VS_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    print(f"▶ Загружаю базу ЧАСТЬ 2 из: {SIMBLE_PART2_VS_DIR}")
    vs_part2 = FAISS.load_local(
        str(SIMBLE_PART2_VS_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    print("▶ Объединяю индексы...")
    vs_part1.merge_from(vs_part2)

    print(f"▶ Сохраняю ОБЪЕДИНЁННУЮ базу в: {SIMBLE_MERGED_VS_DIR}")
    SIMBLE_MERGED_VS_DIR.mkdir(parents=True, exist_ok=True)
    vs_part1.save_local(str(SIMBLE_MERGED_VS_DIR))

    print("✅ Готово: объединённый vecstore сохранён.")


if __name__ == "__main__":
    main()
