import regex
from llama_cpp import Llama

from grammar_llm.utils import generate_from_regular_expressions

llm = Llama(model_path="./models/llama-2-13b.ggmlv3.q4_0.bin", verbose=False)
prompt = """complete the sentence: I bought a dozen donuts. When I counted the donuts there were"""
patterns = [regex.compile("""[0-9]{2}""")]

out = ""
for t in generate_from_regular_expressions(llm, prompt, patterns, max_tokens=50):
    print(t, end="")
    out += t
