import time
import os
from anthropic import AnthropicBedrock, RateLimitError

from ..types import MessageList, SamplerBase

class BedrockCompletionSampler(SamplerBase):
    """
    Sample from Claude API
    """

    def __init__(
        self,
        model: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        system_message: str | None = None,
        temperature: float = 1.0,  # default in Anthropic example
        enable_extended_thinking: bool = False,
        budget_tokens: int = 4096,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ):
        self.api_key_name = "ANTHROPIC_API_KEY"
        aws_region = os.environ.get("AWS_REGION", None)
        assert aws_region, "Please set AWS_REGION"
        self.client = AnthropicBedrock(aws_region=aws_region)
        # using api_key=os.environ.get("ANTHROPIC_API_KEY") # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.extended_thinking = enable_extended_thinking
        self.budget_tokens = budget_tokens
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        self.image_format = "base64"

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image",
            "source": {
                "type": encoding,
                "media_type": f"image/{format}",
                "data": image,
            },
        }
        return new_image

    def _handle_text(self, text):
        return {"type": "text", "text": text}

    def _pack_message(self, role, content):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        trial = 0
        thinking_params = {"type": "disabled"} if not self.extended_thinking else {
            "type": "enabled",
            "budget_tokens": self.budget_tokens
        }
        while trial < self.max_retries:
            try:
                response = self.client.messages.create(
                    model=self.model,
                    system=self.system_message,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=message_list,
                    thinking=thinking_params,
                )
                if self.extended_thinking:
                    thought_process = response.content[0].thinking
                    full_response_text = f"<think>\n\n{thought_process}</think>\n{response.content[1].text}"
                else:
                    full_response_text = response.content[0].text
                return full_response_text, response.usage.output_tokens
            except RateLimitError as e:
                exception_backoff = 2**(trial+3)  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
