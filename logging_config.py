import logging
import functools
import time
from typing import Callable
from datetime import datetime

# 设置logging配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_tools.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 添加装饰器
def log_tool_usage(func: Callable):
    """记录工具使用情况的装饰器"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        tool_name = args[0].__class__.__name__ if args else "Unknown"
        
        try:
            logger.info(f"开始执行工具 {tool_name}.{func.__name__}")
            logger.info(f"输入参数: args={args[1:]}, kwargs={kwargs}")
            
            result = await func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            logger.info(f"工具 {tool_name}.{func.__name__} 执行成功")
            logger.info(f"执行时间: {execution_time:.2f}秒")
            logger.debug(f"输出结果: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"工具 {tool_name}.{func.__name__} 执行失败: {str(e)}", exc_info=True)
            raise
            
    return wrapper
