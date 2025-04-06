import argparse
import json
import logging
import datetime
import yaml
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

# Import local modules
from .common import make_report
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
from .sampler.o_chat_completion_sampler import OChatCompletionSampler
from .sampler.gemini_sampler import GeminiSampler
from .sampler.aiot_sampler import AIOTSampler
from .sampler.bedrock_sampler import BedrockCompletionSampler

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
        default="claude-3-7-sonnet",
        help='Specify the model to use (default: "claude-3-7-sonnet")',
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=8,
        help='Number of parallel processes to use (default: 8)',
    )
    parser.add_argument(
        "-s",
        "--sampler-config",
        type=Path,
        default=Path("./simple-evals/sampler.yaml"),
        help='Path to the sampler configuration file (default: "./sampler.yaml")',
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("/tmp"),
        help='Directory to save output files (default: "/tmp")',
    )
    return parser.parse_args()

def get_sampler(model_name: str, config_path: Path) -> Any:
    """
    Returns a sampler instance based on the model name.

    Args:
        model_name (str): Name of the model.

    Returns:
        Any: An instance of the sampler.
    """

    # Get the path to the sampler config file, using environment variable or default
    
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Failed to load sampler configuration from {config_path}: {e}")
        raise

    # Get model-specific configuration
    model_config = config.get(model_name)
    if not model_config:
        raise ValueError(f"Model '{model_name}' not found in configuration.")

    sampler_type = model_config.get("sampler_type")
    params = model_config.get("params", {})

    # Process system message references if present
    if "system_message" in params and params["system_message"] in config.get("system_messages", {}):
        # Replace system message reference with actual message
        sys_msg_key = params["system_message"]
        params["system_message"] = config["system_messages"][sys_msg_key]

    # Create the appropriate sampler instance
    if sampler_type == "OChatCompletionSampler":
        return OChatCompletionSampler(**params)
    elif sampler_type == "ChatCompletionSampler":
        return ChatCompletionSampler(**params)
    elif sampler_type == "BedrockCompletionSampler":
        return BedrockCompletionSampler(**params)
    elif sampler_type == "GeminiSampler":
        return GeminiSampler(**params)
    elif sampler_type == "AIOTSampler":
        return AIOTSampler(**params)

    raise ValueError(f"Unknown sampler type: {sampler_type}")

def get_evaluator(eval_name: str, test_run: bool, equality_checker: Any, num_threads: int) -> Any:
    """
    Returns an evaluator instance based on the evaluation name.

    Args:
        eval_name (str): Name of the evaluation.
        test_run (bool): Flag to indicate if this is a test run.
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
                n_repeats=1 if test_run else 1, 
                num_examples=num_examples_map["gpqa"],
                variant="extended",
                rng_seed=42,
                num_threads=num_threads,
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

    equality_checker = None
    # equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")

    # Example of multiple evaluations (currently commented out)
    # evals = {eval_name: get_evals(eval_name) for eval_name in ["mmlu", "math", "gpqa", "mgsm", "drop"]}
    # logging.debug(f"Initialized evaluators: {evals}")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.debug(f"Output directory set to: {output_dir}")

    mergekey2resultpath: Dict[str, Path] = {}

    model_name = args.model
    sampler = get_sampler(model_name, args.sampler_config)
    if not sampler:
        logging.error(f"Sampler for model '{model_name}' not found.")
        raise ValueError(f"Sampler for model '{model_name}' not found.")

    eval_name = "gpqa"  # You can modify this to accept as an argument if needed
    eval_obj = get_evaluator(eval_name, args.test_run, equality_checker, args.parallel)

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
