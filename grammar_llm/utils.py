from typing import Generator
import llama_cpp
import numpy as np
import regex
from lark import Lark, UnexpectedInput
from llama_cpp import Llama
from pydantic import BaseModel, Field


def is_ascii_decodable(s: str) -> bool:
    """Return whether the given string is ascii.

    Args:
        s (str): The string to check.

    Returns:
        bool: Whether the given string is ascii.
    """
    try:
        s.encode(encoding="utf-8").decode("ascii")
    except UnicodeDecodeError:
        return False
    else:
        return True


def build_token_map(llm: Llama) -> dict[int, str]:
    """Return a map of tokenid to token string for the given Llama instance.

    Args:
        llm (Llama): The Llama instance to build the token map for.

    Returns:
        dict[int, str]: A map of tokenid to token string.
    """
    token_map: dict[int, str] = {}
    for i in range(llama_cpp.llama_n_vocab(llm.ctx)):
        val = llama_cpp.llama_token_to_str(llm.ctx, i).decode("utf-8", errors="ignore")
        token_map[i] = val
    return token_map


class TokenFilter(BaseModel):
    """
    TokenFilter is a class that can be used to filter tokens by regex patterns. It is used to
    filter tokens that are partially parsable by the grammar model.
    """

    token_map: dict[int, str] = Field(..., description="A map of tokenid to token string.")
    patterns: list[regex.Pattern] = Field(
        ..., description="A list of regex patterns to filter tokens by."
    )

    def is_partial_parsable_token(self, token_id: int, partial_completion: str) -> bool:
        """Return whether the given token is partially parsable given the regex patterns.

        If the token value is empty, then it is considered parsable.

        Args:
            token_id (int): The token id to check.
            partial_completion (str): The partial completion to check against.

        Returns:
            bool: Whether the given token is partially parsable given the regex patterns.
        """
        token_val = self.token_map[token_id]
        if not is_ascii_decodable(token_val):
            return False

        for pattern in self.patterns:
            match = pattern.fullmatch(partial_completion + token_val, partial=True)
            if not match:
                break
            if match.span() == (0, 0):
                break
            return True
        return False

    def filter_partial_parsable_tokens(self, partial_completion: str) -> set[int]:
        """Return a set of token ids that are partially parsable given the regex patterns.

        Args:
            partial_completion (str): The partial completion to check against.

        Returns:
            set[int]: A set of token ids that are partially parsable given the regex patterns.
        """
        result = set(
            filter(
                lambda token_id: self.is_partial_parsable_token(token_id, partial_completion),
                self.token_map.keys(),
            )
        )
        return result

    class Config:
        arbitrary_types_allowed = True


class LogitMask(BaseModel):
    """Zeroes out non non parsable tokens."""

    partial_parsable_tokens: set[int] = Field(
        ...,
        description="A set of token ids that are partially parsable given the regex patterns.",
        min_length=1,
    )

    def __call__(self, input_ids: list[int], scores: list[float]) -> list[float]:
        """Return a list of scores with the partial parsable tokens masked.

        Args:
            input_ids (list[int]): The input ids.
            scores (list[float]): The scores.

        Returns:
            list[float]: A list of scores with the partial parsable tokens masked.
        """
        mask = np.ones_like(scores) * -1e10
        partial_parsable_tokens = np.array(list(self.partial_parsable_tokens))
        mask[partial_parsable_tokens] = 0.0
        return_scores = np.array(scores) + mask
        return list(return_scores.tolist())


def generate_from_regular_expressions(
    llm: Llama, prompt: str, patterns: list[regex.Pattern], max_tokens: int = 5, **kwargs: dict
) -> str:
    """Complete a prompt with the output matching the regex(es).

    Returns when the output fully matches the regex(es) or when max_tokens is reached.

    Args:
        llm (Llama): The Llama instance to use.
        prompt (str): The prompt to complete.
        patterns (list[regex.Pattern]): The regex patterns to use.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 5.
        kwargs: Additional arguments to pass to Llama.generate.

    Returns:
        str: The completed prompt.
    """
    partial_completion: str = ""
    prompt_plus_completion: str = prompt + partial_completion

    token_map = build_token_map(llm)
    gen_tokens = 0
    while gen_tokens < max_tokens:
        token_filter = TokenFilter(
            token_map=token_map,
            patterns=patterns,
        )
        logit_mask = LogitMask(
            partial_parsable_tokens=token_filter.filter_partial_parsable_tokens(partial_completion),
        )

        tokens = llm.tokenize(prompt_plus_completion.encode("utf-8"))
        next_token = next(
            llm.generate(
                tokens,
                logits_processor=logit_mask,
                **kwargs,
            )
        )
        output_text = llm.detokenize([next_token]).decode("utf-8", errors="ignore")
        previous_partial_completion = partial_completion
        partial_completion += output_text
        prompt_plus_completion = prompt_plus_completion + output_text

        for p in patterns:
            m = p.match(partial_completion)
            if m:
                if m.start() == 0 and m.end() < (len(partial_completion) - 5):
                    return str(m[0])
                if previous_partial_completion == partial_completion:
                    return str(m[0])
        gen_tokens += 1
    return partial_completion


class ParserState(BaseModel):
    """A class to hold the state of the parser to determine next parsable tokens."""

    parser: Lark = Field(..., description="The Lark parser.")

    def next_lex(self, input_str: str) -> list[str]:
        """Return the next lexemes given the input string.

        Args:
            input_str (str): The input string.

        Returns:
            list[str]: The next lexemes.
        """
        try:
            self.parser.parse(input_str)
        except UnexpectedInput:
            # Assuming that self.parser is always LALR
            interactive = self.parser.parse_interactive(input_str)
            interactive.exhaust_lexer()
            return list(interactive.accepts())

        return []

    def extract_terminal_regex(self, stop_token: str) -> dict[str, regex.Pattern]:
        """Return a map of terminal name to regex pattern.

        Args:
            stop_token (str): The stop token to use.

        Returns:
            dict[str, regex.Pattern]: A map of terminal name to regex pattern.
        """
        regex_map = {}
        for term in self.parser.terminals:
            if term.pattern:
                regex_map[term.name] = regex.compile(term.pattern.to_regexp())

        regex_map["$END"] = regex.compile(stop_token)
        return regex_map

    class Config:
        arbitrary_types_allowed = True


def generate_from_cfg(
    llm: Llama, prompt: str, parser: Lark, max_tokens: int = 5, **kwargs: dict
) -> Generator[str, None, None]:
    """
    Complete a prompt with a regex pattern.

    Args:
        llm (Llama): The Llama instance to use.
        prompt (str): The prompt to complete.
        parser (Lark): The Lark parser to use.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 5.
        kwargs: Additional arguments to pass to Llama.generate.

    Yields:
        Generator[str, None, None]: A generator of completed prompts.
    """
    partial_completion: str = ""
    prompt_plus_completion = prompt + partial_completion

    parser_state = ParserState(parser=parser)
    terminal_regexes = parser_state.extract_terminal_regex(
        llm.detokenize([llm._token_eos]).decode("utf-8")  # pylint: disable=protected-access
    )

    gen_tokens = 0
    while gen_tokens < max_tokens:
        valid_next_lex = parser_state.next_lex(partial_completion)
        if len(valid_next_lex) == 0 or (len(valid_next_lex) == 1 and "$END" in valid_next_lex):
            break
        r = [terminal_regexes[t] for t in valid_next_lex]
        next_token_completion = generate_from_regular_expressions(
            llm, prompt_plus_completion, r, max_tokens=max_tokens, **kwargs
        )
        yield next_token_completion
        partial_completion += next_token_completion
        prompt_plus_completion = prompt_plus_completion + next_token_completion
        gen_tokens += 1

    # return partial_completion
