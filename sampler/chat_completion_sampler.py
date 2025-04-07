import base64
import time
import os
from typing import Any

import openai
from openai import OpenAI, AzureOpenAI

from ..types import MessageList, SamplerBase

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float = 0.5,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        timeout: int = 120,
        provider: str = "openai",
    ):
        self.timeout = timeout
        if provider == "azure":
            self.client = AzureOpenAI(timeout=timeout)
        elif provider == "openai":
            self.api_key_name = "SE_OAI_API_KEY"
            api_key = os.environ.get(self.api_key_name)
            if not api_key:
                raise ValueError(
                    f"Please set {self.api_key_name} environment variable"
                )
            self.client = OpenAI(
                api_key=api_key,
                timeout=timeout)
        elif provider == "custom":
            self.api_key_name = "SE_CUSTOM_API_KEY"
            api_key = os.environ.get(self.api_key_name)
            base_url = os.environ.get("SE_CUSTOM_API_BASE")
            if not api_key or not base_url:
                raise ValueError(
                    f"Please set {self.api_key_name} and SE_CUSTOM_API_BASE environment variables"
                )
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url, 
                timeout=timeout)
        else:
            raise ValueError(
                f"Invalid provider '{provider}'. Please use 'openai', 'azure', or 'custom'."
            )

        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.image_format = "url"

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
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )
                return response.choices[0].message.content, response.usage.completion_tokens
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 2**(trial+3)  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
