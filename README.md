# 🧠 EmoTwin: Эмоциональный цифровой двойник / Emotional Digital Twin

**RU:** AI-компаньон для анализа ментального здоровья с объяснимым искусственным интеллектом (XAI). Система не просто определяет эмоцию, но выявляет триггеры стресса и дает персонализированные рекомендации на основе базы знаний из 30+ категорий.  
**EN:** An AI mental health companion with Explainable AI (XAI). The system not only detects emotions but also identifies stress triggers and provides personalized recommendations based on a knowledge base of 30+ categories.

---

## ✨ Особенности / Features

- **RU:** Анализ тональности русского текста (RuBert) | Извлечение ключевых слов-триггеров (KeyBERT/YAKE) | Data-driven генерация советов (MSSQL Knowledge Base) | Приоритизация критических состояний (суицид, паника, дереализация) | REST API + Swagger UI  
- **EN:** Russian text sentiment analysis (RuBert) | Keyword trigger extraction (KeyBERT/YAKE) | Data-driven advice generation (MSSQL Knowledge Base) | Critical state prioritization (suicide, panic, derealization) | REST API + Swagger UI

---

## 🛠 Стек технологий / Tech Stack

| Компонент / Component       | Технология / Technology                     |
|-----------------------------|---------------------------------------------|
| Backend                     | FastAPI, Uvicorn                            |
| NLP / ML                    | Transformers, KeyBERT, YAKE                 |
| Database                    | Microsoft SQL Server, SQLAlchemy, PyODBC    |
| Analytics                   | Custom SQL aggregation                      |
| Config                      | python-dotenv                               |

---

## 🚀 Как запустить / How to Run

### RU
1. Установите зависимости: `pip install -r requirements.txt`  
2. Создайте БД `EmotionDB` и выполните скрипт `schema.sql` в SSMS  
3. Скопируйте `.env.example` в `.env` и заполните данные подключения к MSSQL  
4. Запустите сервер: `python main.py`  
5. Откройте документацию API: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  

### EN
1. Install dependencies: `pip install -r requirements.txt`  
2. Create `EmotionDB` database and run `schema.sql` script in SSMS  
3. Copy `.env.example` to `.env` and fill in your MSSQL connection details  
4. Start the server: `python main.py`  
5. Open API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  

---

## 📂 Структура проекта / Project Structure
