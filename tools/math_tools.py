from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression using Python. Supports arithmetic,
    trigonometry, statistics, and any expression valid in Python's math module.
    Example inputs: '2 + 2', 'math.sqrt(144)', 'sum([1,2,3,4,5]) / 5'
    """
    import math
    import statistics

    allowed_names = {
        k: v for k, v in math.__dict__.items() if not k.startswith("_")
    }
    allowed_names.update({
        k: v for k, v in statistics.__dict__.items() if not k.startswith("_")
    })
    allowed_names["abs"] = abs
    allowed_names["round"] = round
    allowed_names["sum"] = sum
    allowed_names["min"] = min
    allowed_names["max"] = max
    allowed_names["len"] = len
    allowed_names["pow"] = pow
    allowed_names["int"] = int
    allowed_names["float"] = float

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def get_math_tools():
    python_repl = PythonREPLTool()
    return [calculator, python_repl]
