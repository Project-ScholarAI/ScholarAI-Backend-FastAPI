"""
Messaging module for ScholarAI

Handles all message queue operations including RabbitMQ connection management,
message processing, and result publishing.
"""

from .connection import RabbitMQConnection
from .handlers import WebSearchMessageHandler  
from .consumer import ScholarAIConsumer

__all__ = [
    'RabbitMQConnection',
    'WebSearchMessageHandler',
    'ScholarAIConsumer'
] 