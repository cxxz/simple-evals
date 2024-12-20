import base64
import time
from typing import Any

import openai
from openai import OpenAI

from multi_agent_llm import OpenAILLM, AIOT
from pydantic import BaseModel, Field

class QueryAnswer(BaseModel):
    explanation: str = Field(description="Explanation of the answer")
    answer: str = Field(description="Final Answer")


from ..types import MessageList, SamplerBase

def get_full_response_text(result):
    response_text = ""
    for i in result.thoughts:
        response_text += f"""
    ## Iteration {i.iteration}

    Brain_Thought:
    {i.brain_thought}

    LLM_Explanation:
    {i.llm_response.explanation}"

    LLM_Ans: {i.llm_response.answer}

    Is_Final: {i.is_final}
    """
        
    response_text += f"Answer: {i.llm_response.answer}"
    return response_text

class AIOTSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.client = OpenAILLM(model_name=self.model)
        self.agent = AIOT(llm=self.client, iterations=10, answer_schema=QueryAnswer)
        self.system_message = system_message
        self.temperature = temperature
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
        prompt = message_list[0]['content']
        # print(prompt)
        try:
            result = self.agent.run(prompt)
            response_text = get_full_response_text(result)
            return response_text
        except Exception as e:
            print("Error", e)
            return ""
