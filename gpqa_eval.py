"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

import random
import re

import blobfile as bf
import pandas
from datasets import load_dataset

from . import common
from .common import ANSWER_PATTERN_MULTICHOICE, HTML_JINJA, format_multichoice_question
from .types import Eval, EvalResult, MessageList, SamplerBase, SingleEvalResult


class GPQAEval(Eval):
    def __init__(
        self,
        n_repeats: int = 4,
        variant: str = "diamond",
        rng_seed: int = 17,
        num_threads: int = 6,
        num_examples: int | None = None,  # restrict to a subset of the data for debugging
        domain: str | None = None,
    ):
        dataset = load_dataset("idavidrein/gpqa", f"gpqa_{variant}")
        df = dataset["train"].to_pandas()
        if domain is not None:
            df = df[df.Subdomain == domain]
        examples = [row.to_dict() for _, row in df.iterrows()]
        rng = random.Random(rng_seed)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            examples = rng.sample(examples, num_examples)
        examples = examples * n_repeats
        examples = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]
        self.examples = examples
        self.num_threads = num_threads
        self.n_repeats = n_repeats

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            choices = [choices[i] for i in row["permutation"]]
            correct_index = choices.index(row["Correct Answer"])
            correct_answer = "ABCD"[correct_index]
            choices_dict = dict(
                A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=row["Question"]
            )
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(choices_dict), role="user"
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
