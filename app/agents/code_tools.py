
import ast

import re

def extract_python_code(message: str) -> str:
    # First, look for a fenced Python code block.
    # Example: ```python ... ```
    match = re.search(
        r"```(?:python|py)\s*\n(.*?)```",
        message,
        re.DOTALL | re.IGNORECASE,
    )

    if match:
        # Return only the code inside the fenced block.
        return match.group(1).strip()

    # For our current test, code starts at a function definition.
    # This is a temporary fallback, not a general code extractor.
    match = re.search(
        r"(?m)^(?:async\s+def|def|class)\s+\w+",
        message,
    )

    if match:
        return message[match.start():].strip()

    # Return an empty string when no code can be identified.
    return ""


def analyze_python_code(code: str) -> dict:
    # Parse Python source code without executing it.
    try:
        tree = ast.parse(code)

    except SyntaxError as error:
        # Return syntax-error details if parsing fails.
        return {
            "valid": False,
            "error": error.msg,
            "line": error.lineno,
        }

    # Find function definitions anywhere in the syntax tree.
    functions = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    # Find class definitions.
    classes = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    ]

    # Find imported modules.
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)

        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")

    # Return structured information about the source code.
    return {
        "valid": True,
        "functions": functions,
        "classes": classes,
        "imports": imports,
    }


if __name__ == "__main__":
    # Test our analyzer using a small Python example.
    sample_code = """
import os

class Calculator:
    def add(self, a, b):
        return a + b
"""

    # Display the extracted code structure.
    print(analyze_python_code(sample_code))

