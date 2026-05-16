from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def get_memory():
    """Initialize conversation memory for the agent."""
    history = ChatMessageHistory()
    return history