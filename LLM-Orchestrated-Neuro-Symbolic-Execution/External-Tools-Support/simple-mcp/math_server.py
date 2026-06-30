from mcp.server.fastmcp import FastMCP
from sympy import symbols, solve, diff, integrate, sympify
import math
import numpy as np
from typing import List

# Create MCP server
app = FastMCP(
    title="Quick Math Server",
    description="Lightweight server for core math operations",
    version="0.1.0"
)

# Allowed math functions for eval
SAFE_ENV = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "pi": math.pi,
    "e": math.e,
    "abs": abs,
    "pow": pow,
    "np": np,
    "math": math
}

@app.tool()
def evaluate(expression: str) -> dict:
    """Evaluate a math expression like '2 + 3 * sin(pi / 2)'."""
    try:
        result = eval(expression, {"__builtins__": {}}, SAFE_ENV)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def solve_equation(equation: str) -> dict:
    """Solve an equation like 'x**2 - 4 = 0'."""
    try:
        x = symbols("x")
        left, right = map(str.strip, equation.split("="))
        solutions = solve(sympify(left) - sympify(right), x)
        return {"solutions": [str(sol) for sol in solutions]}
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def differentiate(expr: str) -> dict:
    """Differentiate an expression like 'sin(x)'."""
    try:
        x = symbols("x")
        result = diff(sympify(expr), x)
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def integrate_expr(expr: str) -> dict:
    """Integrate an expression like 'x**2'."""
    try:
        x = symbols("x")
        result = integrate(sympify(expr), x)
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}

# Run server
if __name__ == "__main__":
    app.run(transport="stdio")
