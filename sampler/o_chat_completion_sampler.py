import time
import os
from typing import Any

import openai
from openai import OpenAI

from ..types import MessageList, SamplerBase


class OChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API for o series models
    """

    def __init__(
        self,
        *,
        reasoning_effort: str | None = None,
        model: str = "o1-mini",
        max_retries: int = 3,
    ):
        self.api_key_name = "SE_OAI_API_KEY"
        api_key=os.environ.get(self.api_key_name)
        if api_key is None:
            raise ValueError(
                f"Please set the {self.api_key_name} environment variable to use OpenAI API key."
            )
        self.client = OpenAI(api_key=api_key)
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
                exception_backoff = 2**(trial+3)  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
