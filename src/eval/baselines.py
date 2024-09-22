from typing import Any, Callable
from litellm import completion

from ..structs import Estimate, Estimator
from ..utils import completion_text, parse_estimate


def run_simple_llm_estimator(model: str, question: str) -> Estimate:
    prompt = f"""
    You are an expert at estimating quantities.
    You will be given a question that asks you to estimate a quantity.
    It is important that you actually try to provide an estimate, rather than giving up, even if estimating is hard or uncertain.
    You should reason step-by-step to estimate what that quantity could be.
    You should also reason about the uncertainty of your estimate, and track a lower and an upper bound estimate.
    
    The question is:
    {question}
    
    Your reasoning:
    """
    
    messages = [{"content": prompt, "role": "user"}]
    
    response_text = completion_text(model, messages, max_tokens=1000)
    
    messages.append({"content": response_text, "role": "assistant"})
    messages.append({"content": "Now please given an output containing nothing but a JSON object with the following structure: {'lower': <float>, 'value': <float>, 'upper': <float>, 'unit': <str>}. 'value' is whatever central estimate you reasoned above, 'lower' is your lower bound estimate, 'upper' is your upper bound estimate, and 'unit' is the unit of the estimates you provided (you can use scientific 'e' notation), which might be 'kg' or 'm' or 'people' or 'spiders/m^2' or '$/h' or 'kg CO2' or whatever it needs to be.", "role": "user"})
    
    response2_text = completion_text(model, messages, max_tokens=1000)
    estimate = parse_estimate(response2_text)
    estimate.reasoning_trace = messages
    return estimate

def create_estimator(name: str, estimator_fn: Callable[[str, Any], Estimate], model: str) -> Estimator:
    def curried_estimator(question: str, *args: Any, **kwargs: Any) -> Estimate:
        return estimator_fn(model, question, *args, **kwargs)
    return Estimator(fn=curried_estimator, name=name)

def simple_llm_estimator(model: str) -> Estimator:
    return create_estimator(f"simple_llm_estimator:{model}", run_simple_llm_estimator, model)
