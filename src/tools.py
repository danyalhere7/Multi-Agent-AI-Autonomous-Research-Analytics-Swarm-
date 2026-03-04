import json
import traceback
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool

# Using SearchResults instead of Run for URL access
search_tool = DuckDuckGoSearchResults(max_results=5)

@tool
def web_search(query: str) -> str:
    """Useful for searching the web for real-time information. Returns snippets and links."""
    try:
        return search_tool.run(query)
    except Exception as e:
        return f"Error executing search for query '{query}': {str(e)}"

@tool
def execute_python_code(code: str) -> str:
    """
    Executes Python code and returns the printed output or the error.
    Use this for data analysis with pandas, numpy, and matplotlib.
    IMPORTANT: Do not use this for destructive actions.
    """
    try:
        import sys
        from io import StringIO
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        # Create a dictionary to hold variables from the execution namespace
        exec_namespace = {
            'pd': pd,
            'np': np,
            'plt': plt
        }

        # Catch stdout
        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output

        # Execute code
        try:
            # We use a simple exec here. 
            # In a production environment, we'd use a subprocess or restricted sandbox.
            exec(code, exec_namespace)
        except Exception:
            # Try to eval if it's a single expression
            try:
                result = eval(code, exec_namespace)
                if result is not None:
                    print(result)
            except Exception:
                # If both fail, let it raise the original exception from exec
                exec(code, exec_namespace)

        # Restore stdout
        sys.stdout = old_stdout
        
        output = redirected_output.getvalue()
        if not output.strip():
            return "Code executed successfully, but no output was printed. Use print() for results."
        return output
    except Exception as e:
        # Restore stdout in case of error
        if 'old_stdout' in locals():
            sys.stdout = old_stdout
        return f"Error executing python code:\n{traceback.format_exc()}"

TOOLS = [web_search, execute_python_code]
