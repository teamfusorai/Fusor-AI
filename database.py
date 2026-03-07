import sqlite3
import os
from typing import Optional, Dict
from utils.logging_config import get_logger

logger = get_logger(__name__)

DB_PATH = os.getenv("SQLITE_DB_PATH", "chatbot_configs.db")

def init_db():
    """Initialize the SQLite database and create tables if they don't exist."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chatbot_configs (
                user_id TEXT,
                bot_id TEXT,
                chatbot_name TEXT,
                description TEXT,
                industry TEXT,
                color TEXT,
                logo TEXT,
                welcome_message TEXT,
                knowledge_source TEXT,
                tone TEXT,
                system_prompt TEXT,
                qr_code TEXT,
                temperature REAL,
                PRIMARY KEY (user_id, bot_id)
            )
        ''')
        conn.commit()
        
        # Add new columns gracefully if updating an existing database
        try:
            cursor.execute("ALTER TABLE chatbot_configs ADD COLUMN qr_code TEXT")
        except sqlite3.OperationalError:
            pass # Column likely exists
            
        try:
            cursor.execute("ALTER TABLE chatbot_configs ADD COLUMN temperature REAL")
        except sqlite3.OperationalError:
            pass # Column likely exists
            
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

def save_chatbot_config(user_id: str, bot_id: str, config: Dict) -> bool:
    """Save or update a chatbot configuration in the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO chatbot_configs 
            (user_id, bot_id, chatbot_name, description, industry, color, logo, welcome_message, tone, system_prompt, temperature)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, 
            bot_id,
            config.get('chatbot_name'),
            config.get('description'),
            config.get('industry'),
            config.get('color'),
            config.get('logo'),
            config.get('welcome_message'),
            config.get('tone'),
            config.get('system_prompt'),
            config.get('temperature')
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"Saved config for user_id={user_id}, bot_id={bot_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving config to DB: {e}")
        return False

def get_chatbot_config_from_db(user_id: str, bot_id: str) -> Optional[Dict]:
    """Retrieve a chatbot configuration from the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM chatbot_configs WHERE user_id = ? AND bot_id = ?', (user_id, bot_id))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return dict(row)
        return None
    except Exception as e:
        logger.error(f"Error getting config from DB: {e}")
        return None

def delete_chatbot_config(user_id: str, bot_id: str) -> bool:
    """Delete a chatbot configuration from the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM chatbot_configs WHERE user_id = ? AND bot_id = ?', (user_id, bot_id))
        
        conn.commit()
        conn.close()
        logger.info(f"Deleted config for user_id={user_id}, bot_id={bot_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting config from DB: {e}")
        return False
