import time
import os
from typing import Any

import openai
from openai import OpenAI, AzureOpenAI

from ..types import MessageList, SamplerBase


class OChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API for o series models
    """

    def __init__(
        self,
        *,
        reasoning_effort: str | None = None,
        model: str = "o3-mini",
        max_retries: int = 3,
        timeout: int = 120,
        provider: str = "openai",
    ):
        self.timeout = timeout
        if provider == "azure":
            self.api_key_name = "SE_AZURE_API_KEY"
            api_key = os.environ.get(self.api_key_name)
            azure_endpoint = os.environ.get("SE_AZURE_ENDPOINT_URL")
            api_version = os.environ.get("SE_AZURE_API_VERSION", "2024-12-01-preview")
            if not api_key or not azure_endpoint:
                raise ValueError(
                    f"Please set {self.api_key_name} and SE_AZURE_ENDPOINT_URL environment variables"
                )
            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                timeout=timeout
                )
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
        self.image_format = "url"
        self.reasoning_effort = reasoning_effort
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
        trial = 0
        while trial < self.max_retries:
            try:
                if self.model.startswith("o1"):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        reasoning_effort=self.reasoning_effort,
                    )
                return response.choices[0].message.content, response.usage.completion_tokens
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 6**(trial+1)  # expontial back off
                print(
                    f"Getting exception: {e} so wait and retry {trial} after {exception_backoff} sec"
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
