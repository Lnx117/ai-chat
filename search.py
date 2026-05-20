import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Загружаем модель. При первом запуске она скачается (~1.2 ГБ, это нормально)
print("Загружаю модель...")
model = SentenceTransformer('intfloat/multilingual-e5-base')
print("Модель готова!")

# Читаем каталог
df = pd.read_csv('products.csv')
print(f"Загружено товаров: {len(df)}")

# Создаем текстовое представление каждого товара
# Это КРИТИЧЕСКИ ВАЖНО: чем информативнее текст, тем лучше поиск
df['text'] = df.apply(
    lambda row: f"Название: {row['name']}. Описание: {row['description']}. Цена: {row['price']} руб.",
    axis=1
)

print("Пример текста товара:")
print(df['text'].iloc[0])
print()

# Превращаем все тексты товаров в векторы
# normalize_embeddings=True делает длину вектора = 1, чтобы косинусное сходство считалось как скалярное произведение
print("Кодирую товары в векторы...")
product_embeddings = model.encode(
    df['text'].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True
)

print(f"Размерность матрицы эмбеддингов: {product_embeddings.shape}")
print(f"Каждый товар представлен вектором из {product_embeddings.shape[1]} чисел")

# Размерность наших векторов
dimension = product_embeddings.shape[1]

# Создаем индекс для поиска по косинусному сходству
# IndexFlatIP — "плоский" индекс с поиском через Inner Product (скалярное произведение)
# Для нормализованных векторов скалярное произведение = косинусное сходство
index = faiss.IndexFlatIP(dimension)

# Добавляем векторы товаров в индекс
# FAISS требует float32
index.add(product_embeddings.astype(np.float32))

print(f"В индексе {index.ntotal} векторов")


def search_products(query: str, top_k: int = 3):
    """
    Ищет top_k товаров, ближайших по смыслу к запросу query.
    Возвращает DataFrame с результатами и оценками близости.
    """
    # 1. Превращаем запрос в вектор
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype(np.float32)
    
    # 2. Ищем ближайшие векторы в индексе
    # scores — это косинусные сходства (от -1 до 1, где 1 = идеальное совпадение)
    # indices — индексы найденных товаров в исходном DataFrame
    scores, indices = index.search(query_embedding, top_k)
    
    # 3. Достаём информацию о найденных товарах
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:  # FAISS возвращает -1 если недостаточно результатов
            continue
        product = df.iloc[idx]
        results.append({
            'id': product['id'],
            'name': product['name'],
            'description': product['description'],
            'price': product['price'],
            'score': round(float(score), 4)  # косинусное сходство
        })
    
    return pd.DataFrame(results)
    
if __name__ == "__main__":
    test_queries = [
        "Ищу подарок дедушке на 23 февраля, он рыбак, бюджет небольшой",
        "Хочу защитить палатку от дождя",
        "Нужна тёплая одежда для зимней прогулки",
        "Что взять на пикник?",
        "Мышь для ноутбука",
    ]
    
    for query in test_queries:
        print("=" * 60)
        print(f"Запрос: {query}")
        print("-" * 60)
        results = search_products(query, top_k=3)
        for _, row in results.iterrows():
            print(f"  [{row['score']:.3f}] {row['name']} — {row['price']} руб.")
            print(f"         {row['description']}")
        print()