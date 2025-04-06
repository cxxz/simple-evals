import time
import os
from typing import Any
from dotenv import load_dotenv
from google import genai
from google.genai import types

from ..types import MessageList, SamplerBase

load_dotenv()

class GeminiSampler(SamplerBase):
    """
    Sample from Gemini's generative model
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-001",
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        provider: str = "vertex_ai",
        max_retries: int = 3,
    ):
        if provider == "vertex_ai":
            project_id = os.getenv("SE_VAI_PROJECT")
            if project_id is None:
                raise ValueError("SE_VAI_PROJECT is not set")
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location="us-central1",
            )
        elif provider == "gemini":
            api_key = os.getenv("SE_GEMINI_API_KEY")
            if api_key is None:
                raise ValueError("SE_GEMINI_API_KEY is not set")
            self.client = genai.Client(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_tokens
        self.top_p = top_p
        self.max_retries = max_retries
        
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
        contents = [
            types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_prompt)
            ]
            )
        ]
        generate_content_config = types.GenerateContentConfig(
            temperature = self.temperature,
            top_p = self.top_p,
            max_output_tokens = self.max_output_tokens,
        )
        trial = 0
        while trial < self.max_retries:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=generate_content_config,
                )
                usage = response.usage_metadata
                if usage.thoughts_token_count is None:
                    output_tokens = usage.candidates_token_count 
                else:
                    output_tokens = usage.candidates_token_count + usage.thoughts_token_count
                return response.text, output_tokens
            except Exception as e:
                exception_backoff = 2**trial  # exponential back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
