"""
Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
Peter Clark  and Isaac Cowhey and Oren Etzioni and Tushar Khot and Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord
https://arxiv.org/abs/1803.05457
"""

import random
import re

import blobfile as bf
import pandas

from . import common
from .common import (
    HTML_JINJA,
    ANSWER_PATTERN_MULTICHOICE,
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
    MULTILINGUAL_ANSWER_REGEXES,
    format_multichoice_question,
    normalize_extracted_answer,
    normalize_response,
)
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

SUPERGPQA_SUBFIELD_MAPPING = {
    "mb": "Biochemistry_and_Molecular_Biology",
    "hep": "Particle_and_Nuclear_Physics",
}


class SuperGPQAEval(Eval):
    def __init__(
        self,
        variant: str = "mb",
        num_examples: int | None = None,
        num_threads: int = 6,
       ):
        subfield = SUPERGPQA_SUBFIELD_MAPPING.get(variant)
        if subfield is None:
            raise ValueError(f"Invalid variant: {variant}. Must be one of {list(SUPERGPQA_SUBFIELD_MAPPING.keys())}.")
        # Load the dataset from the URL
        url = f'https://raw.githubusercontent.com/cxxz/public-files/refs/heads/main/datasets/SuperGPQA_{subfield}.csv'
        df = pandas.read_csv(url)
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self.num_threads = num_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            correct_answer = row["answer_letter"]
            prompt_messages = [
                sampler._pack_message(
                    content=row['question_with_options'], role="user"
                )
            ]
            try:
                response = sampler(prompt_messages)
            except Exception as e:
                print(f"Error in sampler: {e}")
                # Return a default result instead of None
                return SingleEvalResult(
                    html="<p>Error: Sampler failed</p>",
                    score=0.0,
                    convo=prompt_messages + [dict(content="Error: Sampler failed", role="assistant")],
                    metrics={"chars": 0, "tokens": 0},
                    correct_answer=correct_answer,
                    extracted_answer=None
                )
            
            # Check if the response is a tuple
            if isinstance(response, tuple):
                response_text, response_token_count = response
            elif isinstance(response, str):
                response_text = response
                response_token_count = 0
            else:
                response_text = None
                response_token_count = 0
                print(f"Unexpected response type: {type(response)}")
            
            if response_text is None:
                # Handle the case where the response is None
                print("Warning!!! Response is None")
                return SingleEvalResult(
                    html="<p>Error: Response is None</p>",
                    score=0.0,
                    convo=prompt_messages + [dict(content="Response is None", role="assistant")],
                    metrics={"chars": 0, "tokens": 0},
                    correct_answer=correct_answer,
                    extracted_answer=None
                )
            match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
            extracted_answer = match.group(1) if match else None
            score = 1.0 if extracted_answer == correct_answer else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            if response_token_count > 0:
                # If the response is a tuple, we have the token count
                convo[-1]["token_count"] = response_token_count
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={
                    "chars": len(response_text),
                    "tokens": response_token_count,
                    },
                correct_answer=correct_answer,
                extracted_answer=extracted_answer
            )

        results = common.map_with_progress(fn, self.examples, num_threads=self.num_threads)

        return common.aggregate_results(results)
