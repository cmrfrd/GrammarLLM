import json

from lark import Lark
from llama_cpp import Llama

from grammar_llm.utils import generate_from_cfg

llm = Llama(model_path="./models/llama-2-13b.ggmlv3.q4_0.bin", verbose=False)
json_grammar = r"""
?start: json_schema

json_schema: "{" name ", " age ", " height ", " location "}"

name: "\"name\"" ": " ESCAPED_STRING
age: "\"age\"" ": " NUMBER
height: "\"height\"" ": " NUMBER
location: "\"location\"" ": " ESCAPED_STRING

%import common.ESCAPED_STRING
%import common.NUMBER
%import common.WS

%ignore WS
"""

### Create the JSON parser with Lark, using the LALR algorithm
json_parser = Lark(
    json_grammar,
    parser="lalr",
    lexer="basic",
    propagate_positions=False,
    maybe_placeholders=False,
    regex=True,
)
prompt = """
You are an AI assistant that takes raw text data and generates a valid json object representing the data with the following grammer:

```
{json_grammar}
```

Examples:

Input: "dave is 30 years old, 58 inches tall, and lives in Los Angeles."
Output: {"name": "dave", "age": 30, "height": 58, "location": "Los Angeles"}

Input: "jane is 25 years old, 60 inches tall, and lives in New York."
Output: {"name": "jane", "age": 25, "height": 60, "location": "New York"}

Input: "Alex is 27 years old, 72 inches tall, and lives in Miami."
Output: """

out = ""
for t in generate_from_cfg(llm, prompt, json_parser, max_tokens=50):
    print(t, end="")
    out += t

print()
print(json.dumps(json.loads(out), indent=4))
