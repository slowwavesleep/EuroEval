"""IFEval instruction-following constraints and metrics."""

import collections
import collections.abc as c
import json
import logging
import re
import typing as t

import langdetect
import nltk
from langdetect import DetectorFactory

from .base import Metric

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig

# make langdetect deterministic
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

nltk.download("punkt_tab", quiet=True)

Relation = t.Literal["less than", "at least"]

# all checker functions accept **kwargs to absorb extra fields from the
# dataset's kwargs dict, which contains all possible parameters for all instructions.


def check_keyword_existence(response: str, *, keywords: list[str], **_) -> bool:
    """Check that all keywords exist in the response.

    Args:
        response: The response string to check.
        keywords: List of keyword patterns to search for (case-insensitive).

    Returns:
        True if all keywords are found in the response, False otherwise.

    Raises:
        ValueError: If keywords is empty or not provided.
    """
    if not keywords:
        raise ValueError("keywords must be provided")
    for keyword in keywords:
        if not re.search(keyword, response, flags=re.IGNORECASE):
            return False
    return True


def check_keyword_frequency(
    response: str, *, keyword: str, frequency: int, relation: Relation, **_
) -> bool:
    """Check keyword appears with required frequency."""
    if not keyword:
        raise ValueError("keyword must be provided")
    actual = len(re.findall(keyword, response, flags=re.IGNORECASE))
    if relation == "less than":
        return actual < frequency
    return actual >= frequency


def check_forbidden_words(response: str, *, forbidden_words: list[str], **_) -> bool:
    """Check that forbidden words don't appear.

    Args:
        response: The response string to check.
        forbidden_words: List of words that must not appear
        (case-insensitive, whole-word match).

    Returns:
        True if none of the forbidden words are found, False otherwise.

    Raises:
        ValueError: If forbidden_words is empty or not provided.
    """
    if not forbidden_words:
        raise ValueError("forbidden_words must be provided")
    for word in forbidden_words:
        if re.search(r"\b" + word + r"\b", response, flags=re.IGNORECASE):
            return False
    return True


def check_letter_frequency(
    response: str, *, letter: str, let_frequency: int, let_relation: Relation, **_
) -> bool:
    """Check letter appears with required frequency.

    Args:
        response: The response string to check.
        letter: The single character to count (case-insensitive).
        let_frequency: The frequency threshold to compare against.
        let_relation: Comparison operator; either "less than" or "at least".

    Returns:
        True if the letter frequency satisfies the relation, False otherwise.

    Raises:
        ValueError: If letter is not a single character.
    """
    if not letter or len(letter) != 1:
        raise ValueError("letter must be a single character")
    counts = collections.Counter(response.lower())
    if let_relation == "less than":
        return counts[letter.lower()] < let_frequency
    return counts[letter.lower()] >= let_frequency


def check_number_sentences(
    response: str, *, num_sentences: int, relation: Relation, **_
) -> bool:
    """Check number of sentences.

    Args:
        response: The response string to check.
        num_sentences: The sentence count threshold to compare against.
        relation: Comparison operator; either "less than" or "at least".

    Returns:
        True if the sentence count satisfies the relation, False otherwise.
    """
    actual = len(nltk.tokenize.sent_tokenize(response))
    if relation == "less than":
        return actual < num_sentences
    return actual >= num_sentences


def check_number_paragraphs(response: str, *, num_paragraphs: int, **_) -> bool:
    """Check number of paragraphs (separated by ***).

    Args:
        response: The response string to check.
        num_paragraphs: The exact number of paragraphs expected.

    Returns:
        True if the response contains exactly num_paragraphs non-empty paragraphs,
        False otherwise.
    """
    paragraphs = re.split(r"\s?\*\*\*\s?", response)
    count = len(paragraphs)
    for i, p in enumerate(paragraphs):
        if not p.strip():
            if i == 0 or i == len(paragraphs) - 1:
                count -= 1
            else:
                return False
    return count == num_paragraphs


def check_number_words(
    response: str, *, num_words: int, relation: Relation, **_
) -> bool:
    """Check number of words.

    Args:
        response: The response string to check.
        num_words: The word count threshold to compare against.
        relation: Comparison operator; either "less than" or "at least".

    Returns:
        True if the word count satisfies the relation, False otherwise.
    """
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    actual = len(tokenizer.tokenize(response))
    if relation == "less than":
        return actual < num_words
    return actual >= num_words


def check_nth_paragraph_first_word(
    response: str,
    *,
    num_paragraphs: int,
    nth_paragraph: int,
    first_word: str | None,
    **_,
) -> bool:
    """Check paragraph count and first word of nth paragraph.

    Args:
        response: The response string to check.
        num_paragraphs: The exact number of paragraphs expected.
        nth_paragraph: The 1-based index of the paragraph whose first word to check.
        first_word: The expected first word of the nth paragraph (case-insensitive).

    Returns:
        True if the response has exactly num_paragraphs paragraphs and the nth
        paragraph starts with first_word, False otherwise.

    Raises:
        ValueError: If first_word is None.
        ValueError: If nth_paragraph is None.
    """
    if first_word is None:
        raise ValueError("first_word must be provided")

    if not nth_paragraph < num_paragraphs:
        raise ValueError("nth_paragraph must be less than num_paragraphs")

    paragraphs = re.split(r"\n\n", response)
    count = sum(1 for p in paragraphs if p.strip())

    if nth_paragraph > count:
        return False

    paragraph = paragraphs[nth_paragraph - 1].strip()
    if not paragraph:
        return False

    word = paragraph.split()[0].strip().lstrip("'\"")
    actual_first = ""
    for char in word:
        if char in ".,?!'\"":
            break
        actual_first += char.lower()

    return count == num_paragraphs and actual_first == first_word.lower()


def check_number_placeholders(response: str, *, num_placeholders: int, **_) -> bool:
    """Check minimum number of [placeholder] brackets.

    Args:
        response: The response string to check.
        num_placeholders: The minimum number of placeholder brackets expected.

    Returns:
        True if the response contains at least num_placeholders placeholders
        of the form [placeholder], False otherwise.
    """
    placeholders = re.findall(r"\[.*?\]", response)
    return len(placeholders) >= num_placeholders


def check_postscript(response: str, *, postscript_marker: str, **_) -> bool:
    """Check for postscript marker.

    Args:
        response: The response string to check.
        postscript_marker: The postscript label to look for (e.g. "P.S.", "P.P.S").

    Returns:
        True if the postscript marker is found in the response, False otherwise.
    """
    response = response.lower()
    if postscript_marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif postscript_marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + postscript_marker.lower() + r".*$"
    return bool(re.findall(pattern, response, flags=re.MULTILINE))


def check_number_bullet_lists(response: str, *, num_bullets: int, **_) -> bool:
    """Check exact number of bullet points.

    Args:
        response: The response string to check.
        num_bullets: The exact number of bullet points expected.

    Returns:
        True if the response contains exactly num_bullets bullet points,
        where bullet points are lines starting with ``*`` or ``-``, False otherwise.
    """
    bullets1 = re.findall(r"^\s*\*[^\*].*$", response, flags=re.MULTILINE)
    bullets2 = re.findall(r"^\s*-.*$", response, flags=re.MULTILINE)
    return len(bullets1) + len(bullets2) == num_bullets


def check_constrained_response(response: str, **_) -> bool:
    """Check response contains one of the constrained options.

    Args:
        response: The response string to check.

    Returns:
        True if the response contains exactly one of "My answer is yes.",
        "My answer is no.", or "My answer is maybe.", False otherwise.
    """
    options = ("My answer is yes.", "My answer is no.", "My answer is maybe.")
    return any(opt in response.strip() for opt in options)


def check_number_highlighted_sections(
    response: str, *, num_highlights: int, **_
) -> bool:
    """Check minimum highlighted *sections*.

    Args:
        response: The response string to check.
        num_highlights: The minimum number of highlighted sections expected.

    Returns:
        True if the response contains at least num_highlights non-empty highlighted
        sections, where highlights are text wrapped in single (*text*) or double
        (**text**) asterisks, False otherwise.
    """
    count = 0
    for h in re.findall(r"\*[^\n\*]*\*", response):
        if h.strip("*").strip():
            count += 1
    for h in re.findall(r"\*\*[^\n\*]*\*\*", response):
        if h.removeprefix("**").removesuffix("**").strip():
            count += 1
    return count >= num_highlights


def check_multiple_sections(
    response: str, *, section_spliter: str, num_sections: int, **_
) -> bool:
    """Check for Section X markers.

    Args:
        response: The response string to check.
        section_spliter: The section header keyword to split on (e.g. "Section").
        num_sections: The minimum number of sections expected.

    Returns:
        True if the response contains at least num_sections sections delimited
        by markers of the form "<section_spliter> <number>", False otherwise.
    """
    section_splitter = section_spliter
    pattern = r"\s?" + section_splitter + r"\s?\d+\s?"
    sections = re.split(pattern, response)
    return len(sections) - 1 >= num_sections


def check_json_format(response: str, **_) -> bool:
    """Check response is valid JSON.

    Args:
        response: The response string to check.

    Returns:
        True if the response (after stripping optional ```json``` fences) is
        valid JSON, False otherwise.
    """
    value = (
        response.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        json.loads(value)
        return True
    except ValueError:
        return False


def check_title(response: str, **_) -> bool:
    """Check for <<title>> format.

    Args:
        response: The response string to check.

    Returns:
        True if the response contains at least one non-empty title wrapped in
        double angle brackets (e.g. <<My Title>>), False otherwise.
    """
    for title in re.findall(r"<<[^\n]+>>", response):
        if title.lstrip("<").rstrip(">").strip():
            return True
    return False


def check_two_responses(response: str, **_) -> bool:
    """Check for two different responses separated by ******.

    Args:
        response: The response string to check.

    Returns:
        True if the response contains exactly two non-empty, non-identical
        sections separated by "******", False otherwise.
    """
    parts = response.split("******")
    valid = []
    for i, part in enumerate(parts):
        if not part.strip():
            if i != 0 and i != len(parts) - 1:
                return False
        else:
            valid.append(part)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


def check_repeat_prompt(response: str, *, prompt_to_repeat: str, **_) -> bool:
    """Check response starts with the prompt.

    Args:
        response: The response string to check.
        prompt_to_repeat: The exact prompt text the response must begin with
            (case-insensitive).

    Returns:
        True if the response starts with prompt_to_repeat, False otherwise.

    Raises:
        ValueError: If prompt_to_repeat is empty or not provided.
    """
    if not prompt_to_repeat:
        raise ValueError("prompt_to_repeat must be provided")
    return response.strip().lower().startswith(prompt_to_repeat.strip().lower())


def check_end_phrase(response: str, *, end_phrase: str, **_) -> bool:
    """Check response ends with exact phrase.

    Args:
        response: The response string to check.
        end_phrase: The exact phrase the response must end with (case-insensitive).

    Returns:
        True if the response ends with end_phrase, False otherwise.
    """
    return response.strip().strip('"').lower().endswith(end_phrase.strip().lower())


def check_capital_word_frequency(
    response: str, *, capital_frequency: int, capital_relation: Relation, **_
) -> bool:
    """Check frequency of ALL CAPS words.

    Args:
        response: The response string to check.
        capital_frequency: The frequency threshold to compare against.
        capital_relation: Comparison operator; either "less than" or "at least".

    Returns:
        True if the count of fully uppercased words satisfies the relation,
        False otherwise.
    """
    words = nltk.word_tokenize(response)
    count = sum(1 for w in words if w.isupper())
    if capital_relation == "less than":
        return count < capital_frequency
    return count >= capital_frequency


def check_english_capital(response: str, **_) -> bool:
    """Check response is English and all caps.

    Args:
        response: The response string to check.

    Returns:
        True if the response is entirely uppercase and detected as English,
        False otherwise. Returns True if language detection fails.
    """
    try:
        return response.isupper() and langdetect.detect(response) == "en"
    except langdetect.LangDetectException:
        return True


def check_english_lowercase(response: str, **_) -> bool:
    """Check response is English and all lowercase.

    Args:
        response: The response string to check.

    Returns:
        True if the response is entirely lowercase and detected as English,
        False otherwise. Returns True if language detection fails.
    """
    try:
        return response.islower() and langdetect.detect(response) == "en"
    except langdetect.LangDetectException:
        return True


def check_no_comma(response: str, **_) -> bool:
    """Check response has no commas.

    Args:
        response: The response string to check.

    Returns:
        True if the response contains no comma characters, False otherwise.
    """
    return "," not in response


def check_quotation(response: str, **_) -> bool:
    """Check response is wrapped in double quotes.

    Args:
        response: The response string to check.

    Returns:
        True if the response (after stripping whitespace) begins and ends with
        a double quote character, False otherwise.
    """
    response = response.strip()
    return len(response) > 1 and response[0] == '"' and response[-1] == '"'


instruction_checkers = {
    "keywords:existence": check_keyword_existence,
    "keywords:frequency": check_keyword_frequency,
    "keywords:forbidden_words": check_forbidden_words,
    "keywords:letter_frequency": check_letter_frequency,
    "length_constraints:number_sentences": check_number_sentences,
    "length_constraints:number_paragraphs": check_number_paragraphs,
    "length_constraints:number_words": check_number_words,
    "length_constraints:nth_paragraph_first_word": check_nth_paragraph_first_word,
    "detectable_content:number_placeholders": check_number_placeholders,
    "detectable_content:postscript": check_postscript,
    "detectable_format:number_bullet_lists": check_number_bullet_lists,
    "detectable_format:constrained_response": check_constrained_response,
    "detectable_format:number_highlighted_sections": check_number_highlighted_sections,
    "detectable_format:multiple_sections": check_multiple_sections,
    "detectable_format:json_format": check_json_format,
    "detectable_format:title": check_title,
    "combination:two_responses": check_two_responses,
    "combination:repeat_prompt": check_repeat_prompt,
    "startend:end_checker": check_end_phrase,
    "change_case:capital_word_frequency": check_capital_word_frequency,
    "change_case:english_capital": check_english_capital,
    "change_case:english_lowercase": check_english_lowercase,
    "punctuation:no_comma": check_no_comma,
    "startend:quotation": check_quotation,
}

SKIPPED_INSTRUCTIONS = {"language:response_language"}


def check_instruction_following(
    instruction_id_list: list[str], kwargs_list: list[dict], response: str
) -> list[bool]:
    """Check if response follows each instruction.

    Args:
        instruction_id_list: List of instruction IDs corresponding to checker keys
        in instruction_checkers.
        kwargs_list: List of parameter dicts, one per instruction,
        passed to each checker.
        response: The response string to evaluate.

    Returns:
        A list of booleans, one per supported instruction, indicating whether
        the response satisfies each instruction. Unsupported instructions are skipped.

    Raises:
        KeyError: If an instruction_id is not found in instruction_checkers and is
        not in SKIPPED_INSTRUCTIONS.
    """
    results = []
    for instruction_id, kwargs in zip(instruction_id_list, kwargs_list):
        if instruction_id in SKIPPED_INSTRUCTIONS:
            logger.warning(f"Skipping unsupported instruction: {instruction_id}")
            continue

        checker = instruction_checkers[instruction_id]
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        is_following = bool(response.strip() and checker(response, **kwargs))
        results.append(is_following)

    return results


class IFEvalInstructionAccuracy(Metric):
    """Metric for instruction-level accuracy using IFEval methodology."""

    def __init__(
        self,
        name: str = "inst_level_strict_acc",
        pretty_name: str = "Instruction-Level Strict Accuracy",
        postprocessing_fn: t.Callable[[float], tuple[float, str]] | None = None,
    ) -> None:
        """Initialize the metric."""
        super().__init__(
            name=name, pretty_name=pretty_name, postprocessing_fn=postprocessing_fn
        )

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Calculate instruction-level accuracy."""
        all_results: list[bool] = []
        for pred, ref in zip(predictions, references):
            results = check_instruction_following(
                instruction_id_list=ref["instruction_id_list"],
                kwargs_list=ref["kwargs"],
                response=str(pred),
            )
            all_results.extend(results)
        return sum(all_results) / len(all_results) if all_results else 0.0


inst_level_strict_acc_metric = IFEvalInstructionAccuracy()
