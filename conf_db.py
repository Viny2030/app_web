import os
from sqlalchemy import create_all, create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# 1. Cargamos las variables del archivo .env que ya configuramos
load_dotenv()

# 2. Obtenemos la URL de la base de datos (por defecto usaremos SQLite para desarrollo)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")

# 3. Creamos el motor de la base de datos
# check_same_thread=False es necesario solo para SQLite
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# 4. Creamos una fábrica de sesiones
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 5. Clase base para que nuestros modelos hereden de ella
Base = declarative_base()

# Función para obtener la base de datos de forma segura
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
