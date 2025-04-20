import argparse
import json
import logging
import datetime
import yaml
import sys
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
from .sampler.chat_completion_sampler import ChatCompletionSampler
from .sampler.o_chat_completion_sampler import OChatCompletionSampler
from .sampler.gemini_sampler import GeminiSampler
from .sampler.aiot_sampler import AIOTSampler
from .sampler.bedrock_sampler import BedrockCompletionSampler

SUPPORTED_BENCHMARKS = ["mmlu", "math", "gpqa", "mgsm", "drop"]

def setup_logging(debug: bool) -> None:
    """
    Configures logging settings.

    Args:
        debug (bool): If True, sets log level to DEBUG; otherwise, INFO.
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
        argparse.Namespace: Parsed command-line arguments.
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
        default="gpt-4o-mini",
        help='Specify the model to use (default: "gpt-4o-mini")',
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=8,
        help="Number of parallel processes to use (default: 8)",
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
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        default="gpqa",
        help='Name of the benchmark to run (options: "mmlu", "math", "gpqa", "mgsm", "drop"; default: "gpqa")',
    )
    return parser.parse_args()

def get_sampler(model_name: str, config_path: Path) -> Any:
    """
    Returns a sampler instance based on the model name.

    Args:
        model_name (str): Name of the model.
        config_path (Path): Path to the sampler configuration file.

    Returns:
        Any: An instance of the sampler.

    Raises:
        ValueError: If the model configuration or sampler type is not found.
    """
    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Failed to load sampler configuration from {config_path}: {e}")
        raise

    model_config = config.get(model_name)
    if not model_config:
        raise ValueError(f"Model '{model_name}' not found in configuration.")

    sampler_type = model_config.get("sampler_type")
    params = model_config.get("params", {})

    # Replace system message reference with the actual message if applicable
    system_messages = config.get("system_messages", {})
    if "system_message" in params and params["system_message"] in system_messages:
        sys_msg_key = params["system_message"]
        params["system_message"] = system_messages[sys_msg_key]

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
        test_run (bool): Flag indicating if this is a test run.
        equality_checker (Any): Sampler used for equality checking in MathEval.
        num_threads (int): Number of threads to use.

    Returns:
        Any: An instance of the evaluator.

    Raises:
        ValueError: If the evaluation type is unrecognized.
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
            return MathEval(equality_checker=equality_checker, num_examples=num_examples_map["math"])
        case "gpqa":
            return GPQAEval(
                n_repeats=1,
                num_examples=num_examples_map["gpqa"],
                variant="extended",
                #variant="diamond",
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
        case _:
            raise ValueError(f"Unrecognized evaluation type: {eval_name}")

def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Saves a dictionary as a JSON file.

    Args:
        data (Dict[str, Any]): Data to save.
        filepath (Path): Path to the JSON file.
    """
    try:
        with filepath.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        logging.info(f"Saved JSON data to {filepath}")
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
        with filepath.open("w", encoding="utf-8") as file:
            file.write(content)
        logging.debug(f"Saved HTML report to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save HTML report to {filepath}: {e}")

def main() -> List[Dict[str, Any]]:
    """
    Main function to execute the evaluation pipeline.

    Returns:
        List[Dict[str, Any]]: Merged evaluation metrics.
    """
    args = parse_arguments()
    setup_logging(args.debug)
    logging.info("Starting evaluation pipeline")

    equality_checker = None
    # Uncomment the following line if an equality checker is required
    # equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.debug(f"Output directory set to: {output_dir}")

    mergekey2resultpath: Dict[str, Path] = {}

    model_name = args.model
    sampler = get_sampler(model_name, args.sampler_config)
    if not sampler:
        logging.error(f"Sampler for model '{model_name}' not found.")
        raise ValueError(f"Sampler for model '{model_name}' not found.")

    eval_name = args.benchmark
    if eval_name not in SUPPORTED_BENCHMARKS:
        logging.error(f"Invalid benchmark name '{eval_name}'.")
        raise ValueError(f"Invalid benchmark name '{eval_name}'.")
    
    eval_obj = get_evaluator(eval_name, args.test_run, equality_checker, args.parallel)

    logging.info(f"Running evaluation '{eval_name}' with model '{model_name}'")
    result = eval_obj(sampler)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_stem = f"{eval_name}_{model_name}"
    test_suffix = "_test" if args.test_run else ""

    # Save detailed results
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

    # Save HTML report
    report_filename = output_dir / f"{file_stem}{test_suffix}_{timestamp}.html"
    logging.info(f"Writing report to {report_filename}")
    save_html(make_report(result), report_filename)

    # Save summary metrics
    metrics = {**result.metrics, "score": result.score}
    logging.info(f"Metrics: {metrics}")
    result_filename = output_dir / f"{file_stem}{test_suffix}_{timestamp}.json"
    save_json(metrics, result_filename)
    logging.info(f"Saved summary results to {result_filename}")

    mergekey2resultpath[file_stem] = result_filename
    merge_metrics: List[Dict[str, Any]] = []

    # Merge metrics from result files
    for eval_model_name, result_path in mergekey2resultpath.items():
        try:
            with result_path.open("r", encoding="utf-8") as f:
                result_data = json.load(f)
            metric = result_data.get("f1_score", result_data.get("score"))
            eval_name_extracted = eval_model_name.split("_")[0]
            model_name_extracted = "_".join(eval_model_name.split("_")[1:])
            merge_metrics.append({
                "eval_name": eval_name_extracted,
                "model_name": model_name_extracted,
                "metric": metric
            })
            logging.debug(f"Processed results for {eval_model_name}: {metric}")
        except Exception as e:
            logging.error(f"Error processing {result_path}: {e}")

    if merge_metrics:
        merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
            index="model_name", columns="eval_name", values="metric"
        )
        logging.info("All results:")
        logging.info(f"\n{merge_metrics_df.to_markdown()}")
    else:
        logging.warning("No metrics to merge.")

    return merge_metrics

if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        logging.exception("An unhandled exception occurred during the evaluation pipeline.")
        sys.exit(1)
