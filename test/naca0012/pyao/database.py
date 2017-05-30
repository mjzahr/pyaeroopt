from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Boolean, Float

Base = declarative_base()

class AerofDb(Base):

    __tablename__ = 'aerof'

    id = Column(Integer, primary_key=True)
    
