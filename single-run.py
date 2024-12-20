import argparse
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

# Import local modules
from .common import make_report, SIMPLE_GPQA_SYS_MSG, REFLECTION_SYS_MSG
from .drop_eval import DropEval
from .gpqa_eval import GPQAEval
from .math_eval import MathEval
from .mgsm_eval import MGSMEval
from .mmlu_eval import MMLUEval
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from .sampler.o1_chat_completion_sampler import O1ChatCompletionSampler
from .sampler.gemini_sampler import GeminiSampler
from .sampler.aiot_sampler import AIOTSampler
from .sampler.bedrock_sampler import ClaudeChatCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS


def setup_logging(debug: bool) -> None:
    """
    Configures the logging settings.

    Args:
        debug (bool): If True, set log level to DEBUG, else INFO.
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate language models.")
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode (default: False)",
    )
    parser.add_argument(
        "-t",
        "--test-run",
        action="store_true",
        default=False,
        help="Enable test run (default: False)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help='Specify the model to use (default: "claude-3-5-sonnet-20241022")',
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("/tmp"),
        help='Directory to save output files (default: "/tmp")',
    )
    return parser.parse_args()


def get_samplers() -> Dict[str, Any]:
    """
    Initializes and returns a dictionary of sampler instances.

    Returns:
        Dict[str, Any]: Dictionary mapping sampler names to sampler instances.
    """
    return {
        # ChatGPT models
        "o1-preview": O1ChatCompletionSampler(model="o1-preview"),
        "o1-mini": O1ChatCompletionSampler(model="o1-mini"),
        "gpt-4-turbo-2024-04-09_assistant": ChatCompletionSampler(
            model="gpt-4-turbo-2024-04-09",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
        ),
        "gpt-4-turbo-2024-04-09_chatgpt": ChatCompletionSampler(
            model="gpt-4-turbo-2024-04-09",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        ),
        "gpt-4o_assistant": ChatCompletionSampler(
            model="gpt-4o",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=4096,
        ),
        "gpt-4o_chatgpt": ChatCompletionSampler(
            model="gpt-4o",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
            max_tokens=4096,
        ),
        "gpt-4o-mini-2024-07-18": ChatCompletionSampler(
            model="gpt-4o-mini-2024-07-18",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=4096,
        ),
        # Claude models
        "claude-3-5-sonnet-20241022": ClaudeChatCompletionSampler(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            system_message=SIMPLE_GPQA_SYS_MSG,
            max_tokens=4096,
        ),
        # Gemini models
        "gemini-exp-1206": GeminiSampler(
            model_name="gemini-exp-1206",
            max_tokens=8192,
        ),
        # on-prem models
        "llama-3.1-405b-instruct": ChatCompletionSampler(
            model="Meta-Llama-3.1-405B-Instruct-AWQ-INT4",
            system_message=REFLECTION_SYS_MSG,
            max_tokens=8192,
        ),
        # on-prem models
        "aiot-llama-3.1-405b-instruct": AIOTSampler(
            model="Meta-Llama-3.1-405B-Instruct-AWQ-INT4",
            max_tokens=8192,
        ),
    }


def get_evaluator(eval_name: str, test_run: bool, equality_checker: Any) -> Any:
    """
    Returns an evaluator instance based on the evaluation name.

    Args:
        eval_name (str): Name of the evaluation.
        debug (bool): Debug flag to set the number of examples.
        equality_checker (Any): Sampler used for equality checking in MathEval.

    Returns:
        Any: An instance of the evaluator.
    """
    num_examples_map = {
        "mmlu": 1 if test_run else 2500,
        "math": 5 if test_run else 2500,
        "gpqa": 5 if test_run else None,
        "mgsm": 10 if test_run else 250,
        "drop": 10 if test_run else 2000,
    }

    match eval_name:
        case "mmlu":
            return MMLUEval(num_examples=num_examples_map["mmlu"])
        case "math":
            return MathEval(
                equality_checker=equality_checker, num_examples=num_examples_map["math"]
            )
        case "gpqa":
            return GPQAEval(
                n_repeats=1 if test_run else 1, num_examples=num_examples_map["gpqa"]
            )
        case "mgsm":
            return MGSMEval(num_examples_per_lang=num_examples_map["mgsm"])
        case "drop":
            return DropEval(
                num_examples=num_examples_map["drop"],
                train_samples_per_prompt=3,
            )
        # case "humaneval":
        #     return HumanEval(num_examples=10 if debug else None)
        case _:
            raise ValueError(f"Unrecognized eval type: {eval_name}")


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Saves a dictionary as a JSON file.

    Args:
        data (Dict[str, Any]): Data to save.
        filepath (Path): Path to the JSON file.
    """
    try:
        with filepath.open("w") as file:
            json.dump(data, file, indent=4)
        logging.info(f"Saved JSON of detailed results to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save JSON data to {filepath}: {e}")


def save_html(content: str, filepath: Path) -> None:
    """
    Saves content as an HTML file.

    Args:
        content (str): HTML content to save.
        filepath (Path): Path to the HTML file.
    """
    try:
        with filepath.open("w") as file:
            file.write(content)
        logging.debug(f"Saved HTML report to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save HTML report to {filepath}: {e}")


def main() -> List[Dict[str, Any]]:
    """
    Main function to execute the evaluation pipeline.

    Returns:
        List[Dict[str, Any]]: List of merged metrics from evaluations.
    """
    args = parse_arguments()
    setup_logging(args.debug)
    logging.info("Starting evaluation pipeline")

    samplers = get_samplers()
    equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")

    def get_evals(eval_name: str) -> Any:
        return get_evaluator(eval_name, args.test_run, equality_checker)

    # Example of multiple evaluations (currently commented out)
    # evals = {eval_name: get_evals(eval_name) for eval_name in ["mmlu", "math", "gpqa", "mgsm", "drop"]}
    # logging.debug(f"Initialized evaluators: {evals}")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.debug(f"Output directory set to: {output_dir}")

    mergekey2resultpath: Dict[str, Path] = {}

    model_name = args.model
    sampler = samplers.get(model_name)
    if not sampler:
        logging.error(f"Sampler for model '{model_name}' not found.")
        raise ValueError(f"Sampler for model '{model_name}' not found.")

    eval_name = "gpqa"  # You can modify this to accept as an argument if needed
    eval_obj = get_evals(eval_name)

    logging.info(f"Running evaluation '{eval_name}' with model '{model_name}'")
    result = eval_obj(sampler)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_stem = f"{eval_name}_{model_name}"
    test_suffix = "_test" if args.test_run else ""

    detailed_results_filename = output_dir / f"details_{file_stem}{test_suffix}_{timestamp}.json"
    detailed_results = {
        "score": result.score,
        "metrics": result.metrics,
        "convos": result.convos,
        "scores": result.scores,
        "correct_answers": result.correct_answers,
        "extracted_answers": result.extracted_answers,
    }
    save_json(detailed_results, detailed_results_filename)

    report_filename = output_dir / f"{file_stem}{test_suffix}_{timestamp}.html"
    logging.info(f"Writing report to {report_filename}")
    save_html(make_report(result), report_filename)

    metrics = {**result.metrics, "score": result.score}
    logging.info(f"Metrics: {metrics}")

    result_filename = output_dir / f"{file_stem}{test_suffix}_{timestamp}.json"
    save_json(metrics, result_filename)
    logging.info(f"Writing results to {result_filename}")

    mergekey2resultpath[file_stem] = result_filename
    merge_metrics: List[Dict[str, Any]] = []

    for eval_model_name, result_path in mergekey2resultpath.items():
        try:
            with result_path.open("r") as f:
                result_data = json.load(f)
            metric = result_data.get("f1_score", result_data.get("score"))
            eval_name_extracted = eval_model_name.split("_")[0]
            model_name_extracted = "_".join(eval_model_name.split("_")[1:])
            merge_metrics.append(
                {"eval_name": eval_name_extracted, "model_name": model_name_extracted, "metric": metric}
            )
            logging.debug(f"Processed results for {eval_model_name}: {metric}")
        except Exception as e:
            logging.error(f"Error processing {result_path}: {e}")

    if merge_metrics:
        merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
            index=["model_name"], columns="eval_name"
        )
        logging.info("\nAll results: ")
        logging.info(f"\n{merge_metrics_df.to_markdown()}")
    else:
        logging.warning("No metrics to merge.")

    return merge_metrics


if __name__ == "__main__":
    main()
