import os
import requests
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Optional

EURON_API_KEY = os.environ.get("EURON_API_KEY")

def generate_completion(messages, model="gpt-4.1-nano", max_tokens=1000, temperature=0.7):
    """
    Generate a response from Euron API
    """
    url = "https://api.euron.one/api/v1/euri/chat/completions"
    headers = {
       "Content-Type": "application/json",
       "Authorization": f"Bearer {EURON_API_KEY}"
    }


    # Convert LangChain messages to API format
    api_messages = []
    for message in messages:
        if hasattr(message, 'type'):
            role = message.type
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            api_messages.append({"role": role, "content": message.content})
        else:
            api_messages.append(message)

    payload = {
        "messages": api_messages,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Euron API Error: {response.status_code} {response.text}")

    return response.json()


class EuronChatModel(BaseChatModel):
    """
    LangChain compatible chat model using Euron API
    """
    model_name: str = "gpt-4.1-nano"

    def _generate(self, messages: List, stop: Optional[List[str]] = None) -> ChatResult:
        response = generate_completion(messages, model=self.model_name)

        # Extract AI message
        ai_content = response['choices'][0]['message']['content']

        # Wrap in LangChain objects
        ai_message = AIMessage(content=ai_content)
        generation = ChatGeneration(message=ai_message)

        return ChatResult(generations=[generation])

    def _llm_type(self) -> str:
        return "euron-chat"


# Optional: simple chat function
def simple_chat_completion(user_message: str):
    messages = [{"role": "user", "content": user_message}]
    response = generate_completion(messages)
    return response['choices'][0]['message']['content']
