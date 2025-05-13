from sqlalchemy import Column, create_engine, Integer, PickleType, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker


__all__ = ["create_mysql_engine", "FileDataset"]


Base = declarative_base()


class FileDataset(Base):  # type: ignore
    __tablename__ = "dataset"
    id = Column(Integer, primary_key=True)
    data_filename = Column(String(128), nullable=False)
    dataset = Column(PickleType, nullable=False)


def create_mysql_engine(sqlalchemy_database_url: str) -> Session:
    kwargs = {}
    if sqlalchemy_database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}

    engine = create_engine(sqlalchemy_database_url, **kwargs)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return SessionLocal()
