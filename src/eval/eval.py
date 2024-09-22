import csv
from datetime import datetime
import traceback
import math
import os
import json
from typing import Any, Callable
from dataclasses import dataclass, asdict
from litellm import completion
from tqdm import tqdm

from ..utils import ScientificNotationEncoder, completion_text, convert_units
from ..structs import Estimate, Estimator


@dataclass
class QueryEvalResult:
    question: str
    estimate: Estimate | str
    estimator_name: str
    correct_estimate: Estimate
    log: Any | None = None

@dataclass
class EvalResult:
    estimator_name: str
    score: float
    queries_incorrect: list[QueryEvalResult]
    queries_correct: list[QueryEvalResult]

def load_eval() -> tuple[list[str], list[Estimate]]:
    questions = []
    estimates = []
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'eval.csv')
    
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['lower'] and row['upper']:
                lower = float(row['lower'])
                upper = float(row['upper'])
                value = math.sqrt(lower * upper)
                estimate = Estimate(
                    lower=lower,
                    value=value,
                    upper=upper,
                    unit=row['unit']
                )
                estimates.append(estimate)
                questions.append(row['question'])
    return questions, estimates

def run_eval(estimator: Estimator, progress=True, parallel=True) -> EvalResult:
    eval_result = generate_eval_result(estimator, progress, parallel)
    save_eval_result(eval_result)
    return eval_result

def generate_eval_result(estimator: Estimator, progress=True, parallel=True) -> EvalResult:
    questions, estimates = load_eval()
    
    def process_estimate(args):
        i, correct_estimate = args
        question = questions[i]
        log = []
        try:
            estimated_estimate = estimator.fn(question)
        except Exception as e:
            estimated_estimate = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            log.append(traceback.format_exc())
        query_result = QueryEvalResult(question, estimated_estimate, estimator.name, correct_estimate, log)
        score, error_log = calculate_score(estimated_estimate, correct_estimate)
        if error_log:
            log.append(error_log)
        query_result.log = log
        return query_result, score

    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=10) as executor:
            if progress:
                results = list(tqdm(executor.map(process_estimate, enumerate(estimates)), total=len(estimates)))
            else:
                results = list(executor.map(process_estimate, enumerate(estimates)))
    else:
        if progress:
            results = [process_estimate(item) for item in tqdm(enumerate(estimates))]
        else:
            results = [process_estimate(item) for item in enumerate(estimates)]

    queries, scores = zip(*results)
    total_score = sum(scores)
    queries_correct = [query for query, score in zip(queries, scores) if score == 1]
    queries_incorrect = [query for query, score in zip(queries, scores) if score == 0]

    avg_score = total_score / len(estimates) if estimates else 0
    return EvalResult(f"{estimator.name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", avg_score, queries_incorrect, queries_correct)

def save_eval_result(eval_result: EvalResult):
    print("Saving eval result...")
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'out')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'{eval_result.estimator_name}_eval_result.json')
    print(f"Writing to file: {out_file}")
    
    with open(out_file, 'w') as f:
        json.dump(eval_result, f, indent=2, cls=ScientificNotationEncoder)
    
    # Print the first part of the file content
    with open(out_file, 'r') as f:
        print(f"JSON data (first 800 chars): {f.read(800)}...")
    
    print("Finished saving eval result.")

def calculate_score(estimated: Estimate | str, correct: Estimate) -> tuple[float, str]:
    if isinstance(estimated, str):
        # this is a sign that there was an error in the parsing code
        return 0, estimated
    conversion_factor = convert_units(estimated.unit, correct.unit)
    if isinstance(conversion_factor, str):
        if conversion_factor.lower().strip() == "same":
            return 1, ""
        elif conversion_factor.lower().strip().startswith("invalid"):
            return 0, f"Invalid units: {estimated.unit} and {correct.unit}. Conversion model returned: {conversion_factor[7:]}"
        else:
            raise ValueError(f"Unexpected response from conversion model: {conversion_factor}")
    estimated_value = estimated.value * conversion_factor
    if correct.lower < estimated_value < correct.upper:
        return 1, ""
    else:
        return 0, ""
