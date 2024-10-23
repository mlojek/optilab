"""
JSON schema of JSON with experiment results data. The file should contain
a list of dictionaries. The dictionaries should have field name with string
name of the experiment, field dim with int value representing the dimensionality
of solved problem, and field logs with list of lists of floats representing
the result logs of the optimization problems.
"""

results_json_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "dim": {"type": "integer"},
            "logs": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
            },
        },
        "required": ["name", "dim", "logs"],
    },
}
