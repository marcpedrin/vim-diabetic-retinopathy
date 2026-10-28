"""Evaluation package: from-scratch metrics (metrics.py), the evaluation script
(evaluate.py, run with `python -m eval.evaluate`) and the report figures
(plots.py, run with `python -m eval.plots`).

Note: Python also has a built-in function called `eval`. Because we only ever
write `from eval.metrics import ...` or `python -m eval.evaluate`, the built-in
is never hidden; but `import eval` on its own would shadow it in that file.
"""
