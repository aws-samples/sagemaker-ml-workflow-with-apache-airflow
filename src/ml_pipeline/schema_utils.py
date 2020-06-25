import json
# Python Module example


def abalone_schema():
    schema = {
        "input": [
            {
                "name": "sex",
                "type": "string"
            },
            {
                "name": "length",
                "type": "double"
            },
            {
                "name": "diameter",
                "type": "double"
            },
            {
                "name": "height",
                "type": "double"
            },
            {
                "name": "whole_weight",
                "type": "double"
            },
            {
                "name": "shucked_weight",
                "type": "double"
            },
            {
                "name": "viscera_weight",
                "type": "double"
            },
            {
                "name": "shell_weight",
                "type": "double"
            },
        ],
        "output":
            {
                "name": "features",
                "type": "double",
                "struct": "vector"
        }
    }
    schema_json = json.dumps(schema)
    return (schema_json)
