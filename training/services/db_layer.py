#!/usr/bin/env python3
"""
数据库缓存层 - 使用SQLAlchemy
支持SQLite(开发)和PostgreSQL(生产)
"""

import logging
import os
import json
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, String, LargeBinary, Integer, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

logger = logging.getLogger(__name__)

Base = declarative_base()


class CacheEntry(Base):
    """缓存表定义"""
    __tablename__ = 'cache_entries'
    
    key = Column(String(255), primary_key=True)
    value = Column(Text, nullable=False)  # JSON字符串
    ttl_seconds = Column(Integer, default=3600)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    hits = Column(Integer, default=0)
    deleted_at = Column(DateTime, nullable=True)


class CacheStats(Base):
    """缓存统计表"""
    __tablename__ = 'cache_stats'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    total_hits = Column(Integer, default=0)
    total_misses = Column(Integer, default=0)
    total_size_bytes = Column(Integer, default=0)
    avg_ttl_seconds = Column(Integer, default=0)


class DbOperations:
    """数据库操作接口"""
    
    def __init__(self, db_url: Optional[str] = None):
        """
        初始化数据库连接
        
        Args:
            db_url: 数据库URL
                   如果为None，使用SQLite: sqlite:///./browerai_cache.db
                   PostgreSQL: postgresql://user:password@localhost/dbname
        """
        if db_url is None:
            # 默认使用SQLite
            db_url = os.getenv('DATABASE_URL', 'sqlite:///./browerai_cache.db')
        
        self.db_url = db_url
        logger.info(f"🔌 连接数据库: {db_url.split('@')[0] if '@' in db_url else db_url[:50]}")
        
        # SQLite特殊处理
        if 'sqlite' in db_url:
            self.engine = create_engine(
                db_url,
                connect_args={"check_same_thread": False},
                echo=False
            )
        else:
            # PostgreSQL连接池
            self.engine = create_engine(
                db_url,
                pool_size=10,
                max_overflow=20,
                echo=False
            )
        
        # 创建表
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        logger.info("✅ 数据库初始化完成")
    
    @contextmanager
    def get_session(self):
        """获取数据库会话"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"❌ 数据库错误: {e}")
            raise
        finally:
            session.close()
    
    def get(self, key: str) -> Optional[dict]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
        
        Returns:
            缓存的数据(dict)，如果不存在返回None
        """
        with self.get_session() as session:
            try:
                entry = session.query(CacheEntry).filter(
                    CacheEntry.key == key,
                    CacheEntry.deleted_at == None
                ).first()
                
                if not entry:
                    return None
                
                # 更新访问计数
                entry.hits += 1
                session.commit()
                
                # 解析JSON
                try:
                    return json.loads(entry.value)
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ 缓存数据格式错误: {key}")
                    return None
            
            except Exception as e:
                logger.error(f"❌ 缓存读取失败 {key}: {e}")
                return None
    
    def set(self, key: str, value: dict, ttl_seconds: int = 3600) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存数据(dict)
            ttl_seconds: 过期时间(秒)
        
        Returns:
            是否成功
        """
        with self.get_session() as session:
            try:
                # 尝试更新现有记录
                entry = session.query(CacheEntry).filter(
                    CacheEntry.key == key
                ).first()
                
                value_json = json.dumps(value)
                
                if entry:
                    entry.value = value_json
                    entry.ttl_seconds = ttl_seconds
                    entry.updated_at = datetime.utcnow()
                else:
                    entry = CacheEntry(
                        key=key,
                        value=value_json,
                        ttl_seconds=ttl_seconds
                    )
                    session.add(entry)
                
                session.commit()
                logger.debug(f"✅ 缓存保存: {key}")
                return True
            
            except Exception as e:
                logger.error(f"❌ 缓存保存失败 {key}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """删除缓存(软删除)"""
        with self.get_session() as session:
            try:
                entry = session.query(CacheEntry).filter(
                    CacheEntry.key == key,
                    CacheEntry.deleted_at == None
                ).first()
                
                if entry:
                    entry.deleted_at = datetime.utcnow()
                    session.commit()
                    return True
                return False
            except Exception as e:
                logger.error(f"❌ 缓存删除失败 {key}: {e}")
                return False
    
    def clear(self) -> int:
        """清空所有缓存"""
        with self.get_session() as session:
            try:
                count = session.query(CacheEntry).filter(
                    CacheEntry.deleted_at == None
                ).count()
                
                session.query(CacheEntry).filter(
                    CacheEntry.deleted_at == None
                ).update({CacheEntry.deleted_at: datetime.utcnow()})
                
                session.commit()
                logger.info(f"✅ 清空缓存: {count}条")
                return count
            except Exception as e:
                logger.error(f"❌ 缓存清空失败: {e}")
                return 0
    
    def get_stats(self) -> dict:
        """获取缓存统计"""
        with self.get_session() as session:
            try:
                total_entries = session.query(CacheEntry).filter(
                    CacheEntry.deleted_at == None
                ).count()
                
                total_hits = session.query(CacheEntry).filter(
                    CacheEntry.deleted_at == None
                ).with_entities(CacheEntry.hits).all()
                
                total_hits_sum = sum([h[0] for h in total_hits]) if total_hits else 0
                
                # 计算总大小
                import sys
                total_size = sum([
                    sys.getsizeof(e.value) 
                    for e in session.query(CacheEntry).filter(
                        CacheEntry.deleted_at == None
                    ).all()
                ])
                
                return {
                    'total_entries': total_entries,
                    'total_hits': total_hits_sum,
                    'total_size_bytes': total_size,
                    'avg_size_bytes': total_size // total_entries if total_entries > 0 else 0
                }
            except Exception as e:
                logger.error(f"❌ 统计获取失败: {e}")
                return {}
    
    def health_check(self) -> bool:
        """检查数据库连接"""
        with self.get_session() as session:
            try:
                from sqlalchemy import text
                session.execute(text("SELECT 1"))
                return True
            except Exception as e:
                logger.error(f"❌ 数据库健康检查失败: {e}")
                return False


# 全局实例
_db_operations = None


def get_db_operations() -> DbOperations:
    """获取全局数据库操作实例"""
    global _db_operations
    if _db_operations is None:
        _db_operations = DbOperations()
    return _db_operations


if __name__ == '__main__':
    # 测试数据库层
    logging.basicConfig(level=logging.INFO)
    
    db = DbOperations()
    
    # 测试保存
    test_data = {'framework': 'React', 'confidence': 0.95}
    db.set('test_key_1', test_data)
    
    # 测试读取
    result = db.get('test_key_1')
    print(f"✅ 读取结果: {result}")
    
    # 测试统计
    stats = db.get_stats()
    print(f"📊 缓存统计: {stats}")
    
    # 健康检查
    is_healthy = db.health_check()
    print(f"{'✅' if is_healthy else '❌'} 数据库健康")
