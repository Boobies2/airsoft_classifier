
````markdown
# Category & Subcategory Classifier API

API для классификации страйкбольного снаряжения по тексту и фотографиям постов.  
Использует модель BERT для классификации категории по тексту и ResNet18 для классификации подкатегории по изображениям.

---

## Описание проекта

Этот сервис принимает батч постов с текстом и списком URL фотографий, возвращает предсказания категории и подкатегории для каждого объекта (фото) с указанием уверенности модели.

- Текстовая классификация — BERT (fine-tuned)
- Классификация изображений — ResNet18 (fine-tuned)
- Мультиклассовая обработка постов с несколькими фото
- Защита API ключом

---

## Структура проекта

- `api_server.py` — основной сервер FastAPI с эндпоинтом `/predict_batch`
- `request.py` — пример клиентского скрипта для отправки запросов на сервер
- `category_bert_model/` — директория с моделью и токенизатором BERT
- `weapon_classifier.pth` — сохранённая модель ResNet18 для подкатегорий
- `category_label_encoder.pkl` — сериализованный LabelEncoder для категорий

---

## Установка

1. Клонировать репозиторий:

```bash
git clone https://github.com/Boobies2/airsoft_classifier
cd airsoft_classifier
````

2. Создать и активировать виртуальное окружение (рекомендуется):

```bash
python3 -m venv venv
source venv/bin/activate  # Linux
venv\Scripts\activate     # Windows
```

3. Установить зависимости:

```bash
pip install -r requirements.txt
```

4. Скачать и разместить модели:

* Поместите скачанные файлы модели BERT в папку `category_bert_model/`
* Поместите файл `weapon_classifier.pth` в корень проекта
* Поместите `category_label_encoder.pkl` в корень проекта

**Примечание:** Для скачивания модели BERT с Google Drive используйте, например, [gdown](https://github.com/wkentaro/gdown):

```bash
pip install gdown
gdown https://drive.google.com/uc?id=1RiR5oNhiVa2vlGvxl6pCO5NBs-TPa-bn -O category_bert_model/model.safetensors
```

---

## Запуск сервера

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

---

## API(abc)

### Эндпоинт `/predict_batch`

* Метод: POST
* Заголовки:

  * `Authorization: Bearer abc`
  * `Content-Type: application/json`
* Тело запроса: Список постов в формате JSON:

```json
[
  {
    "post_id": "1",
    "text": "Продаю автомат и жилет",
    "photos": [
      {"photo_id": "1", "url": "https://example.com/photo1.jpg"},
      {"photo_id": "2", "url": "https://example.com/photo2.jpg"}
    ]
  }
]
```

* Ответ:

```json
[
  {
    "post_id": "1",
    "predictions": [
      {
        "object_id": "1",
        "category": "ak",
        "subcategory": "rifle",
        "confidence": 0.87,
        "photo_ids": ["1"]
      },
      {
        "object_id": "2",
        "category": "ak",
        "subcategory": "vest",
        "confidence": 0.92,
        "photo_ids": ["2"]
      }
    ]
  }
]
```

---

## Пример клиента

См. файл `request.py` — пример отправки батча постов на сервер и вывода результата.

---

## Зависимости

* fastapi
* uvicorn
* transformers
* torch
* torchvision
* pillow
* requests
* scikit-learn (для LabelEncoder)
* pydantic

---
