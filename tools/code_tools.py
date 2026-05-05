from langchain_experimental.tools import PythonREPLTool


def get_code_tools():
    python_repl = PythonREPLTool()
    return [python_repl]
