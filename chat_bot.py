import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import ollama
import os
import ssl

# Для Ubuntu/Debian
system_ca_bundle = '/etc/ssl/certs/ca-certificates.crt'
if os.path.exists(system_ca_bundle):
    os.environ['SSL_CERT_FILE'] = system_ca_bundle
    os.environ['REQUESTS_CA_BUNDLE'] = system_ca_bundle
    os.environ['CURL_CA_BUNDLE'] = system_ca_bundle

"""
Простой чат-бот магазина в стиле RAG (Retrieval-Augmented Generation):
1) Ищем подходящие товары по смыслу (эмбеддинги + FAISS).
2) Передаём найденные товары в LLM (Ollama), чтобы получить человеческий ответ.
"""

# ------------------ ПОРОГ РЕЛЕВАНТНОСТИ ------------------
# Если максимальный score ниже этого значения, считаем запрос нецелевым
# и не предлагаем товары (просто вежливо общаемся).
# Значение 0.80 подобрано экспериментально: отсекает приветствия и
# бессмысленные фразы, но пропускает реальные запросы из нашего каталога.
SCORE_THRESHOLD = 0.80

# ------------------ Поиск ------------------
def build_index(csv_path='products_b2b.csv', model_name = 'embaas/sentence-transformers-multilingual-e5-base'):
    # Загружаем модель, которая превращает текст в векторы (эмбеддинги).
    print("Загружаю модель эмбеддингов...")
    model = SentenceTransformer(model_name)

    # Читаем товары из CSV.
    df = pd.read_csv(csv_path)
    # Склеиваем поля товара в один текст: так модели проще сравнивать "смысл" запроса и товара.
    df['text'] = df.apply(
        lambda row: f"Название: {row['name']}. Описание: {row['description']}. Цена: {row['price']} руб.",
        axis=1
    )
    print(f"Товаров в индексе: {len(df)}")

    print("Кодирую товары...")
    # Для каждого товара получаем эмбеддинг.
    # normalize_embeddings=True: нормализация позволяет использовать косинусную близость через скалярное произведение.
    embeddings = model.encode(df['text'].tolist(), normalize_embeddings=True, show_progress_bar=True)

    dim = embeddings.shape[1]
    # FAISS-индекс для быстрого поиска ближайших векторов.
    # IndexFlatIP = точный поиск по Inner Product (для нормализованных векторов это близко к cosine similarity).
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    return model, index, df

def search_products(query, model, index, df, top_k=3):
    # Кодируем запрос пользователя в вектор в том же пространстве, что и товары.
    query_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    # Ищем top_k самых похожих товаров.
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        product = df.iloc[idx]
        # Сохраняем данные о найденном товаре и score похожести.
        results.append({
            'id': product['id'],
            'name': product['name'],
            'description': product['description'],
            'price': product['price'],
            'score': round(float(score), 4)
        })
    return pd.DataFrame(results)

# ------------------ Ollama ------------------
def ask_ollama(user_query, products_df=None, model_name='qwen2.5:7b'):
    """
    Если products_df передан и не пуст – генерируем рекомендацию по товарам.
    Иначе – вежливый ответ без товаров (приветствие, уточнение).
    """
    if products_df is not None and not products_df.empty:
        # Формируем текстовый список найденных товаров для передачи в LLM.
        products_text = ""
        for _, row in products_df.iterrows():
            products_text += f"- {row['name']}: {row['description']} (цена {row['price']} руб.)\n"

        # Большой промпт с ролью и правилами поведения модели.
        # Важный момент: "Не выдумывай характеристики" снижает риск галлюцинаций.
        prompt = f"""Ты — полезный консультант интернет-магазина. 
Покупатель спросил: "{user_query}"

Вот товары, которые система подобрала по смыслу (от наиболее подходящего к менее):
{products_text}

Твоя задача:
1. Вежливо и по-русски предложи эти товары.
2. Объясни, почему каждый из них может подойти.
3. Учитывай бюджет и пожелания, если они есть в вопросе.
4. Не выдумывай характеристики, которых нет в описании.
5. Если запрос слишком общий или товары слабо подходят, задай уточняющий вопрос.

Твой ответ:"""
    else:
        # Нет релевантных товаров – просто вежливый ответ.
        prompt = f"""Ты — вежливый консультант интернет-магазина.
Покупатель написал: "{user_query}"
Это не запрос на подбор товара (возможно, приветствие или общий вопрос).
Ответь кратко и по-русски: поздоровайся, представься, спроси, чем можешь помочь."""

    # Обращаемся к локальной LLM через Ollama.
    response = ollama.chat(
        model=model_name,
        messages=[
            {'role': 'system', 'content': 'Ты — эксперт по подбору товаров, отвечаешь кратко и по делу.'},
            {'role': 'user', 'content': prompt}
        ],
        options={
            # temperature: чем выше, тем "креативнее" и менее предсказуем ответ.
            'temperature': 0.7,
            # Ограничение максимальной длины ответа в токенах.
            'num_predict': 600
        }
    )
    return response['message']['content']

# ------------------ Главный цикл ------------------
if __name__ == "__main__":
    # Один раз при старте строим индекс по товарам.
    print("Инициализация поисковика...")
    model, index, df = build_index('products_b2b.csv')

    print("\n" + "="*50)
    print("Чат-бот готов! Задайте вопрос или введите 'выход'.")
    print("="*50 + "\n")

    while True:
        # Читаем вопрос пользователя в цикле.
        user_input = input("Вы: ")
        if user_input.lower() in ['выход', 'exit', 'quit']:
            print("Бот: До свидания!")
            break

        # 1) Сначала ищем наиболее релевантные товары по смыслу.
        results = search_products(user_input, model, index, df, top_k=3)

        # Проверяем, есть ли товары с достаточно высоким score.
        # Если лучший score ниже порога – не предлагаем товары, просто общаемся.
        if not results.empty and results.iloc[0]['score'] >= SCORE_THRESHOLD:
            # Есть релевантные товары – покажем их и попросим LLM сформулировать рекомендацию.
            print("\n[Найденные товары:]")
            for _, row in results.iterrows():
                print(f"  - {row['name']} (score: {row['score']})")

            print("\nБот думает...")
            try:
                # 2) LLM формулирует финальный ответ на основе найденных товаров.
                answer = ask_ollama(user_input, results)
                print(f"Бот: {answer}\n")
            except Exception as e:
                # Фоллбек: если LLM недоступна, выводим найденные товары вручную.
                print(f"Ошибка при обращении к Ollama: {e}")
                print("Бот (запасной вариант): вот что я нашёл:")
                for _, row in results.iterrows():
                    print(f"- {row['name']} за {row['price']} руб.: {row['description']}")
                print()
        else:
            # Нецелевой запрос – просто передаём его в LLM без товаров.
            print("\nБот думает...")
            try:
                answer = ask_ollama(user_input)  # без товаров
                print(f"Бот: {answer}\n")
            except Exception as e:
                print(f"Ошибка: {e}")
                print("Бот: Здравствуйте! Чем я могу вам помочь?\n")