from dataclasses import dataclass, field
from typing import Literal, List

import numpy as np
from langchain_core.messages import HumanMessage

Role = Literal["user", "assistant"]

@dataclass
class ChatMessage:
    role: Role
    content: str
    images: list[np.ndarray]

@dataclass
class ChatState:
    messages: List[ChatMessage] = field(default_factory=list)

    def add_user_message(self, text: str):
        self.messages.append(ChatMessage(role='user', content=text, images=[]))

    def add_assistant_message(self, text: str, images):
        self.messages.append(ChatMessage(role='assistant', content=text, images=images))

