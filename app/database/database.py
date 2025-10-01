from sqlmodel import create_engine
from app.config.credit_db_conf import DATABASE_CONFIG as DB_CONF

# 使用连接参数而不是URL字符串
database_url = f"mysql+pymysql://{DB_CONF['hostname']}:{DB_CONF['port']}/{DB_CONF['database']}"
engine = create_engine(
    database_url,
    connect_args={
        'user': DB_CONF['username'],
        'password': DB_CONF['password']
    }
)