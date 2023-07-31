import json

from lark import Lark
from llama_cpp import Llama

from grammar_llm.utils import generate_from_cfg

llm = Llama(model_path="./models/wizardlm-13b-v1.2.ggmlv3.q4_1.bin", verbose=False)
json_grammar = r"""
?start: NUMBER
%import common.NUMBER
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
Here is a valid JSON array with 3 people, each having properties for name, age, and height:

```json
[
    {
        "name": "Alice",
        "age": 28,
        "height": 165
    },
    {
        "name": "Bob",
        "age": 35,
        "height": 180
    },
    {
        "name": "Charlie",
        "age": 42,
        "height": 175
    }
]
```

Question: What is bob's age?
Answer: """

out = ""
for t in generate_from_cfg(llm, prompt, json_parser, max_tokens=5):
    print(t, end="")
    out += t
