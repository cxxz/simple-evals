import time
import os
from typing import Any
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel

from ..types import MessageList, SamplerBase

load_dotenv()

class GeminiSampler(SamplerBase):
    """
    Sample from Gemini's generative model
    """

    def __init__(
        self,
        model_name: str = "gemini-exp-1206",
        temperature: float = 1.0,
        max_tokens: int = 8192,
    ):
        project_id = os.getenv("VAI_PROJECT_ID")
        if project_id is None:
            raise ValueError("VAI_PROJECT_ID is not set")
        
        vertexai.init(project=project_id, location="us-central1")
        self.model = GenerativeModel(model_name)
        self.temperature = temperature
        self.max_output_tokens = max_tokens

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        user_prompt = "\n".join([msg["content"] for msg in message_list])
        trial = 0
        while True:
            try:
                response = self.model.generate_content(
                    user_prompt,
                    generation_config={
                        "max_output_tokens": self.max_output_tokens,
                        "temperature": self.temperature,
                    }
                )
                return response.text
            except Exception as e:
                exception_backoff = 2**trial  # exponential back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
