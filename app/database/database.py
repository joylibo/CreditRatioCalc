from sqlmodel import create_engine
from app.config.config import DATABASE_CONFIG as DB_CONF

# 创建数据库连接
database_url = f"mysql://{DB_CONF['username']}:{DB_CONF['password']}@{DB_CONF['hostname']}:{DB_CONF['port']}/{DB_CONF['database']}"
engine = create_engine(database_url)