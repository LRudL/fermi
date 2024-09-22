from dataclasses import asdict, is_dataclass
import concurrent.futures
import json

from litellm import completion
from .structs import Estimate

def completion_text(model: str, messages: list[dict[str, str]], **kwargs) -> str:
    response = completion(model=model, messages=messages, **kwargs)
    return response.choices[0].message.content # type: ignore


def parse_estimate(text: str) -> Estimate:
    # the text is expected to be in form:
    # {'lower': <num>, 'value': <num>, 'upper': <num>, 'unit': <str>}
    # (where <num> can be a float, integer, or a scientific notation float like "3e12")

    """
    NOTE: the response may look like this:

    ```json
    {
        "lower": 5.0e13,
        "value": 1.52691e14,
        "upper": 3.0e14,
        "unit": "spiders"
    }
    ```
    So we need to extract the JSON object from the text, let's do that:
    """
    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    json_text = text[json_start:json_end]
    
    # check that the required fields are present
    if not json_text.startswith("{") or not json_text.endswith("}"):
        raise ValueError("Text is not a valid JSON object")
    
    # parse the JSON object
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError("Text is not a valid JSON object") from e
    
    # check that the required fields are present
    if "lower" not in data or "value" not in data or "upper" not in data or "unit" not in data:
        raise ValueError(f"Text is missing the fields: {', '.join(set(data.keys()) - {'lower', 'value', 'upper', 'unit'})}")
    
    numerical_fields = {"lower", "value", "upper"}
    
    for field in numerical_fields:
        if not isinstance(data[field], (int, float)):
            try:
                data[field] = float(data[field])
            except ValueError as e:
                raise ValueError(f"Field '{field}' is not a valid number") from e
    
    return Estimate(**data)
    
def convert_units(unit1: str, unit2: str) -> str | float:
    """
    Returns a factor f such that f * unit1 = unit2,
    or "invalid" if the units are not convertible.
    """
    prompt = f"""Here are two units:
x = {unit1}
y = {unit2}

Your job is determine a factor converting x -> y. See below for details.

If unit x and y mean the same thing, your response should just be the single word "same". For example, if x = "kg" and y = "kg", then you should return "same". If x = "people" and y = "humans", you should also return "same".

If there is no possible conversion between them, your response should just be the single word "invalid". For example if x = "MWh" and y = "kg", then you should return "invalid". However, any measure of energy is convertible, any measure of volume is convertible, etc.

If they are convertible, then return a mathematical expression for the conversion factor. For example, if x = "h" and y = "s", then you should return the string "3600" because [number in hours] x 3600 = [number in seconds]. If x = "s" and y = "days", you should return "1 / (24 * 60 * 60)" - note that this number is smaller than 1, because days are longer than seconds. If x = "MWh" and y = "joules", we want to multiply a factor to do the conversion Mwh -> joules, so you should return "3600 * 1e6" - note that this is greater than 1, because joules are smaller than MWh. If instead x = "joules" and y = "MWh", we are going joules -> Mwh and you should instead return "1 / (3600 * 1e6)". If x = "J" and y = "btus", you should return "1 / 1055.056". If x = "gallons" and y = "km3", you should return "3.78541 * 1e-9". Remember which way around x and y are: you're being asked how many x are in a y. If y is a bigger unit, your answer should be < 1. If y is a smaller unit, your answer should be > 1. Pay attention to which unit is the larger unit and make sure your answer makes sense in light of that. Feel free to break down the conversion into multiple multiplication / division operations; you do not need to do it all in one go.

Your response should be a string that can be directly evaluated in a Python script to get the correct conversion factor."""
    response = completion_text("claude-3-5-sonnet-20240620", [{"content": prompt, "role": "user"}], temperature=0.0)
    if response == "same":
        return 1
    elif response == "invalid":
        return "invalid"
    else:
        try:
            return eval(response)
        except Exception as e:
            return "invalid: " + str(e)

def test_convert_units():
    def run_assertion(args):
        unit1, unit2, expected = args
        result = convert_units(unit1, unit2)
        return (unit1, unit2, expected, result)

    test_cases : list[tuple[str, str, float | str]] = [
        ("kg", "kg", 1),
        ("kg", "m", "invalid"),
        ("kg", "ton", 0.001),
        ("kg", "lb", 2.20462),
        ("people", "humans", 1),
        ("KWh", "btus", 3412.14),
        ("tons CO2", "CO2", 1),
        ("CO2", "tons CO2", 1),
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(run_assertion, test_cases))
    
    # return a list of failing cases with +/- 1% error tolerance
    def is_within_tolerance(expected : float | str, actual: float | str, tolerance=0.01):
        try:
            actual = float(actual)
        except:
            pass
        if isinstance(expected, str) or isinstance(actual, str):
            return expected == actual
        else:
            return (1 - tolerance) * expected <= actual <= (1 + tolerance) * expected

    cases = [case for case in results if not is_within_tolerance(case[2], case[3])]
    for case in cases:
        print(f"Expected {case[2]} for {case[0]} to {case[1]}, but got {case[3]}")
    return len(cases) == 0

class ScientificNotationEncoder(json.JSONEncoder):
    def default(self, obj):
        if obj is None:
            return None
        if is_dataclass(obj):
            return {k: self.default(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, float):
            return format(obj, '.3e')
        elif isinstance(obj, int):
            return format(obj, '.3e')
        elif isinstance(obj, dict):
            return {k: self.default(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.default(v) for v in obj]
        elif isinstance(obj, str):
            return obj
        return super().default(obj)

    def encode(self, obj):
        def floatToJson(o):
            if isinstance(o, float):
                return format(o, '.3e')
            return o
        return super().encode(json.loads(json.dumps(obj), parse_float=floatToJson))
