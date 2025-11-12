from sqlmodel import create_engine
from app.config.credit_db_conf import DATABASE_CONFIG as DB_CONF

# 构建完整的数据库URL
database_url = f"mysql+pymysql://{DB_CONF['username']}:{DB_CONF['password']}@{DB_CONF['hostname']}:{DB_CONF['port']}/{DB_CONF['database']}"
engine = create_engine(database_url)
