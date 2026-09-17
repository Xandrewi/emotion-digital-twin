from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import logging
import json
from transformers import pipeline
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, func
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os

# --- NLP БИБЛИОТЕКИ ДЛЯ ТРИГГЕРОВ И СОВЕТОВ ---
try:
    from keybert import KeyBERT
    import yake
except ImportError:
    raise ImportError("Установите библиотеки: pip install keybert yake")

# Загружаем переменные из .env
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. НАСТРОЙКА БАЗЫ ДАННЫХ (MSSQL)
# ==========================================

DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
SERVER = os.getenv("DB_SERVER", "localhost")
DATABASE = os.getenv("DB_NAME", "EmotionDB")
USER = os.getenv("DB_USER", "sa")
PASSWORD = os.getenv("DB_PASSWORD", "")
PORT = os.getenv("DB_PORT", "1433")

# Корректное формирование строки подключения без дублирования
if PORT and PORT != "0":
    DATABASE_URL = f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER},{PORT}/{DATABASE}?driver={DRIVER.replace(' ', '+')}"
else:
    DATABASE_URL = f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}/{DATABASE}?driver={DRIVER.replace(' ', '+')}"

try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    logger.info(f"Database engine created successfully! Target: {SERVER}")
except Exception as e:
    logger.error(f"Failed to create DB engine: {e}")
    engine = None

Base = declarative_base()

# Модель таблицы в БД (ОБНОВЛЕННАЯ)
class AnalysisHistory(Base):
    __tablename__ = "analysis_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, index=True)
    text = Column(Text, nullable=False)
    sentiment = Column(String(20), nullable=False)
    score = Column(Float, nullable=False)
    emoji = Column(String(10))
    triggers = Column(Text, nullable=True)      # JSON массив триггеров
    advice = Column(Text, nullable=True)         # Персонализированный совет
    timestamp = Column(DateTime, default=datetime.utcnow)

# Создаем/проверяем таблицы
if engine:
    try:
        Base.metadata.create_all(engine)
        logger.info("Database tables checked/created.")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None

# ==========================================
# 2. NLP ДВИЖОК: ТРИГГЕРЫ + СОВЕТЫ
# ==========================================

class EmotionalAdvisor:
    """Локальный движок для объяснения эмоций и генерации рекомендаций"""
    
    def __init__(self):
        self.kw_model = KeyBERT(model='cointegrated/rubert-tiny2')
        self.yake_extractor = yake.KeywordExtractor(lan="ru", n=3, dedupLim=0.9)

    def extract_triggers(self, text: str) -> list[str]:
        """Извлекает ключевые слова-триггеры из текста"""
        if len(text.strip()) < 5: 
            return []
        try:
            keywords = self.kw_model.extract_keywords(
                text, keyphrase_ngram_range=(1, 2), stop_words='russian', top_n=3
            )
            return [kw[0] for kw in keywords]
        except Exception as e:
            logger.warning(f"KeyBERT failed, using YAKE fallback: {e}")
            keywords = self.yake_extractor.extract_keywords(text)
            return [kw[0] for kw in keywords[:3]]

    def generate_advice(self, sentiment: str, triggers: list[str]) -> str:
        """Генерирует персонализированный совет на основе эмоции и триггеров"""
        if sentiment == 'positive':
            return "Отличный настрой! Постарайтесь сохранить это состояние. Запишите, что именно принесло вам радость сегодня."
        
        # Категоризация триггеров
        work_triggers = {"работа", "учеба", "дедлайн", "начальник", "экзамен", "проект", "коллега", "курсовая", "преподаватель"}
        health_triggers = {"боль", "усталость", "сон", "голова", "температура", "врач", "спал", "бессонница"}
        social_triggers = {"друг", "семья", "отношения", "ссора", "одиночество", "любовь", "партнер"}
        
        trigger_set = set(t.lower() for t in triggers)
        
        if trigger_set & work_triggers:
            return "Похоже на академическое или профессиональное выгорание. Попробуйте технику 'Помодоро' (25 мин работы / 5 мин отдыха). Вашему мозгу нужна микро-пауза, а не полный отказ от задач."
        elif trigger_set & health_triggers:
            return "Физическое состояние напрямую влияет на эмоции. Прежде чем решать проблемы, дайте телу базовый ресурс: стакан воды, 10 минут тишины или короткая прогулка."
        elif trigger_set & social_triggers:
            return "Социальные конфликты истощают. Попробуйте технику 'Я-сообщений': говорите о своих чувствах ('я расстроен'), а не обвиняйте. Иногда лучшая реакция — пауза в общении."
        else:
            return "Я вижу, что вам непросто. Попробуйте дыхательную практику 4-7-8: вдох на 4 счета, задержка на 7, выдох на 8. Это физиологически снижает уровень кортизола."

# Глобальный экземпляр советника
advisor = EmotionalAdvisor()

# ==========================================
# 3. МОДЕЛИ ДАННЫХ API
# ==========================================

class TextRequest(BaseModel):
    text: str
    user_id: Optional[str] = "default"

# ==========================================
# 4. FASTAPI ПРИЛОЖЕНИЕ
# ==========================================

app = FastAPI(
    title="EmoTwin: AI Mental Health Companion",
    description="API с объяснимым анализом эмоций, триггерами и персонализированными советами",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка модели тональности
logger.info("Loading sentiment analysis model...")
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="blanchefort/rubert-base-cased-sentiment", 
        device=-1
    )
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    sentiment_pipeline = None

# ==========================================
# 5. ФУНКЦИИ РАБОТЫ С БД (ОБНОВЛЕННЫЕ)
# ==========================================

def add_to_history_db(user_id: str, text: str, sentiment: str, score: float, 
                      emoji: str, triggers: list, advice: str):
    if not SessionLocal: 
        return
    db = SessionLocal()
    try:
        new_item = AnalysisHistory(
            user_id=user_id,
            text=text,
            sentiment=sentiment,
            score=score,
            emoji=emoji,
            triggers=json.dumps(triggers, ensure_ascii=False),
            advice=advice,
            timestamp=datetime.utcnow()
        )
        db.add(new_item)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"DB Error saving history: {e}")
    finally:
        db.close()

def get_history_db(user_id: str, limit: int = 20):
    if not SessionLocal: 
        return []
    db = SessionLocal()
    try:
        items = db.query(AnalysisHistory).filter(
            AnalysisHistory.user_id == user_id
        ).order_by(AnalysisHistory.timestamp.desc()).limit(limit).all()
        
        return [
            {
                "id": item.id,
                "text": item.text,
                "sentiment": item.sentiment,
                "score": item.score,
                "emoji": item.emoji,
                "triggers": json.loads(item.triggers) if item.triggers else [],
                "advice": item.advice,
                "timestamp": item.timestamp.isoformat()
            }
            for item in items
        ]
    finally:
        db.close()

# ==========================================
# 6. ЭНДПОИНТЫ
# ==========================================

@app.get("/")
def root():
    return {
        "message": "EmoTwin API v3.0 with Explainable AI is running!", 
        "status": "online"
    }

@app.post("/analyze")
async def analyze_text(request: TextRequest):
    """Анализирует текст, находит триггеры и дает персонализированный совет"""
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if sentiment_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        text = request.text[:512]
        
        # 1. Анализ тональности
        result = sentiment_pipeline(text)[0]
        label = result['label'].upper()
        sentiment = label.lower()
        score = result['score']
        
        emoji_map = {'POSITIVE': '😊', 'NEGATIVE': '😔', 'NEUTRAL': '😐'}
        emoji = emoji_map.get(label, '😐')
        
        # 2. Извлечение триггеров и генерация совета (НОВОЕ!)
        triggers = advisor.extract_triggers(text)
        advice = advisor.generate_advice(sentiment, triggers)
        
        # 3. Сохранение в MSSQL
        add_to_history_db(
            user_id=request.user_id,
            text=text,
            sentiment=sentiment,
            score=score,
            emoji=emoji,
            triggers=triggers,
            advice=advice
        )
        
        # 4. Чтение истории
        history = get_history_db(request.user_id, limit=10)
        
        return {
            "result": {
                "sentiment": sentiment,
                "score": round(score, 4),
                "emoji": emoji,
                "confidence": round(score * 100, 1),
                "triggers": triggers,              # НОВОЕ
                "personalized_advice": advice       # НОВОЕ
            },
            "history": history,
            "total_analyzed": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 20):
    history = get_history_db(user_id, limit)
    return {
        "user_id": user_id,
        "total": len(history),
        "history": history
    }

@app.delete("/history/{user_id}")
async def clear_history(user_id: str):
    if not SessionLocal:
        raise HTTPException(status_code=503, detail="DB unavailable")
    db = SessionLocal()
    try:
        db.query(AnalysisHistory).filter(AnalysisHistory.user_id == user_id).delete()
        db.commit()
        return {"message": f"History cleared for user {user_id}"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/stats/{user_id}")
async def get_user_stats(user_id: str):
    """Статистика эмоций пользователя"""
    if not SessionLocal:
        raise HTTPException(status_code=503, detail="Database connection failed")
    
    db = SessionLocal()
    try:
        total_count = db.query(AnalysisHistory).filter(
            AnalysisHistory.user_id == user_id
        ).count()

        if total_count == 0:
            return {"user_id": user_id, "message": "No data", "total_analyses": 0}

        pos = db.query(AnalysisHistory).filter(AnalysisHistory.user_id == user_id, AnalysisHistory.sentiment == 'positive').count()
        neg = db.query(AnalysisHistory).filter(AnalysisHistory.user_id == user_id, AnalysisHistory.sentiment == 'negative').count()
        neu = db.query(AnalysisHistory).filter(AnalysisHistory.user_id == user_id, AnalysisHistory.sentiment == 'neutral').count()
        avg_conf = db.query(func.avg(AnalysisHistory.score)).filter(AnalysisHistory.user_id == user_id).scalar()

        return {
            "user_id": user_id,
            "total_analyses": total_count,
            "emotions": {
                "positive": {"count": pos, "percentage": round((pos / total_count) * 100, 1)},
                "negative": {"count": neg, "percentage": round((neg / total_count) * 100, 1)},
                "neutral": {"count": neu, "percentage": round((neu / total_count) * 100, 1)}
            },
            "average_confidence": round(float(avg_conf), 4) if avg_conf else 0,
            "dominant_emotion": max({"positive": pos, "negative": neg, "neutral": neu}, key=lambda x: x[1])
        }
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# ==========================================
# 7. ЗАПУСК
# ==========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
