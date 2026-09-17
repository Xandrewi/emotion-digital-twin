-- schema.sql
-- Скрипт для создания структуры БД EmotionDB

USE EmotionDB;
GO

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'analysis_history')
BEGIN
    CREATE TABLE analysis_history (
        id INT IDENTITY(1,1) PRIMARY KEY,
        user_id NVARCHAR(100) NOT NULL,
        text NVARCHAR(MAX) NOT NULL,
        sentiment NVARCHAR(20) NOT NULL,
        score FLOAT NOT NULL,
        emoji NVARCHAR(10),
        triggers NVARCHAR(MAX) NULL,      -- JSON массив триггеров
        advice TEXT NULL,                 -- Персонализированный совет
        timestamp DATETIME DEFAULT GETDATE()
    );
    
    PRINT 'Table analysis_history created successfully.';
END
ELSE
BEGIN
    PRINT 'Table analysis_history already exists.';
END
GO