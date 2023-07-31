from llama_cpp import Llama
from lark import Lark

from grammar_llm.utils import generate_from_cfg

llm = Llama(model_path="./models/llama-2-13b.ggmlv3.q4_0.bin", verbose=False)
prompt = """
You are an AI assistant that generates emails:

Here are some examples of valid emails:
- john.doe5678@gmail.com
- sarah.smith_90@yahoo.com
- gpt3user1234@example.com

Generate an email for Elon Musk for his domain x.com:
- """

grammar = """
?start: email
email: name "@" domain
name: CNAME
domain: CNAME "." CNAME

%import common.CNAME
"""

parser = Lark(
    grammar,
    parser="lalr",
    lexer="basic",
    propagate_positions=False,
    maybe_placeholders=False,
    regex=True,
)

out = ""
for t in generate_from_cfg(llm, prompt, parser, max_tokens=5):
    print(t, end="")
    out += t
