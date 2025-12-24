"""IFEval instruction-following constraints and metrics."""

import collections
import collections.abc as c
import functools
import json
import logging
import os
import random
import re
import string
import typing as t
from importlib.metadata import version

import langdetect
import nltk
from langdetect import DetectorFactory
from packaging.version import parse as parse_version

from .base import Metric

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig

# Make langdetect deterministic
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

# NLTK resource download
# Downloading 'punkt' with nltk<3.9 has a remote code vuln.
# see https://github.com/EleutherAI/lm-evaluation-harness/issues/2210
# and https://github.com/nltk/nltk/issues/3266
NLTK_MIN_VERSION = "3.9.1"
_RANK = os.environ.get("LOCAL_RANK", "0")


def _download_nltk_resources() -> None:
    """Download 'punkt' if not already installed."""
    assert (nltk_version := parse_version(version("nltk"))) >= parse_version(
        NLTK_MIN_VERSION
    ), (
        f"`nltk` version {nltk_version} is not >= {NLTK_MIN_VERSION}. "
        "Please update `nltk` before proceeding--older versions are vulnerable "
        "to a remote code execution vulnerability."
    )
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        if _RANK == "0":
            nltk.download("punkt_tab")
            print("Downloaded punkt_tab on rank 0")


_download_nltk_resources()


def _count_words(text: str) -> int:
    """Counts the number of words."""
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)
    return len(tokens)


@functools.lru_cache(maxsize=None)
def _get_sentence_tokenizer():
    return nltk.data.load("nltk:tokenizers/punkt/english.pickle")


def _count_sentences(text: str) -> int:
    """Count the number of sentences."""
    tokenizer = _get_sentence_tokenizer()
    tokenized_sentences = tokenizer.tokenize(text)
    return len(tokenized_sentences)


# ============================================================================
# Constants for instruction checkers
# ============================================================================

_COMPARISON_RELATION = ("less than", "at least")
_MAX_NUM_SENTENCES = 20
_NUM_PLACEHOLDERS = 4
_NUM_BULLETS = 5
_CONSTRAINED_RESPONSE_OPTIONS = (
    "My answer is yes.",
    "My answer is no.",
    "My answer is maybe.",
)
_ENDING_OPTIONS = ("Any other questions?", "Is there anything else I can help with?")
_NUM_HIGHLIGHTED_SECTIONS = 4
_SECTION_SPLITER = ("Section", "SECTION")
_NUM_SECTIONS = 5
_NUM_PARAGRAPHS = 5
_POSTSCRIPT_MARKER = ("P.S.", "P.P.S")
_NUM_KEYWORDS = 2
_KEYWORD_FREQUENCY = 3
_LETTER_FREQUENCY = 10
_ALL_CAPITAL_WORD_FREQUENCY = 20
_NUM_WORDS_LOWER_LIMIT = 100
_NUM_WORDS_UPPER_LIMIT = 500


# ============================================================================
# Instruction checker classes
# ============================================================================


class _Instruction:
    """Base class for instruction checkers."""

    def __init__(self, instruction_id: str) -> None:
        self.id = instruction_id

    def build_description(self, **kwargs) -> str:
        raise NotImplementedError

    def get_instruction_args(self) -> dict | None:
        raise NotImplementedError

    def check_following(self, value: str) -> bool:
        raise NotImplementedError


class _NumberOfSentences(_Instruction):
    """Check the number of sentences."""

    def build_description(self, *, num_sentences=None, relation=None) -> str:
        self._num_sentences_threshold = num_sentences
        if self._num_sentences_threshold is None or self._num_sentences_threshold < 0:
            self._num_sentences_threshold = random.randint(1, _MAX_NUM_SENTENCES)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        return f"Your response should contain {self._comparison_relation} {self._num_sentences_threshold} sentences."

    def get_instruction_args(self):
        return {
            "num_sentences": self._num_sentences_threshold,
            "relation": self._comparison_relation,
        }

    def check_following(self, value):
        num_sentences = _count_sentences(value)
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_sentences < self._num_sentences_threshold
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return num_sentences >= self._num_sentences_threshold


class _PlaceholderChecker(_Instruction):
    """Check the placeholders in template writing."""

    def build_description(self, *, num_placeholders=None) -> str:
        self._num_placeholders = num_placeholders
        if self._num_placeholders is None or self._num_placeholders < 0:
            self._num_placeholders = random.randint(1, _NUM_PLACEHOLDERS)
        return (
            f"The response must contain at least {self._num_placeholders} placeholders "
            "represented by square brackets, such as [address]."
        )

    def get_instruction_args(self):
        return {"num_placeholders": self._num_placeholders}

    def check_following(self, value):
        placeholders = re.findall(r"\[.*?\]", value)
        return len(placeholders) >= self._num_placeholders


class _BulletListChecker(_Instruction):
    """Checks the bullet list in the prompt."""

    def build_description(self, *, num_bullets=None) -> str:
        self._num_bullets = num_bullets
        if self._num_bullets is None or self._num_bullets < 0:
            self._num_bullets = random.randint(1, _NUM_BULLETS)
        return (
            f"Your answer must contain exactly {self._num_bullets} bullet points. "
            "Use the markdown bullet points such as:\n"
            "* This is point 1. \n"
            "* This is point 2"
        )

    def get_instruction_args(self):
        return {"num_bullets": self._num_bullets}

    def check_following(self, value):
        bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
        bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
        return len(bullet_lists) + len(bullet_lists_2) == self._num_bullets


class _ConstrainedResponseChecker(_Instruction):
    """Checks the constrained response."""

    def build_description(self) -> str:
        self._constrained_responses = _CONSTRAINED_RESPONSE_OPTIONS
        return (
            f"Answer with one of the following options: {self._constrained_responses}"
        )

    def get_instruction_args(self) -> None:
        return None

    def check_following(self, value) -> bool:
        value = value.strip()
        for constrained_response in self._constrained_responses:
            if constrained_response in value:
                return True
        return False


class _HighlightSectionChecker(_Instruction):
    """Checks the highlighted section."""

    def build_description(self, *, num_highlights=None) -> str:
        self._num_highlights = num_highlights
        if self._num_highlights is None or self._num_highlights < 0:
            self._num_highlights = random.randint(1, _NUM_HIGHLIGHTED_SECTIONS)
        return (
            f"Highlight at least {self._num_highlights} sections in your answer with "
            "markdown, i.e. *highlighted section*."
        )

    def get_instruction_args(self):
        return {"num_highlights": self._num_highlights}

    def check_following(self, value):
        num_highlights = 0
        highlights = re.findall(r"\*[^\n\*]*\*", value)
        double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", value)
        for highlight in highlights:
            if highlight.strip("*").strip():
                num_highlights += 1
        for highlight in double_highlights:
            if highlight.removeprefix("**").removesuffix("**").strip():
                num_highlights += 1
        return num_highlights >= self._num_highlights


class _SectionChecker(_Instruction):
    """Checks the sections."""

    def build_description(self, *, section_spliter=None, num_sections=None) -> str:
        self._section_spliter = (
            section_spliter.strip()
            if isinstance(section_spliter, str)
            else section_spliter
        )
        if self._section_spliter is None:
            self._section_spliter = random.choice(_SECTION_SPLITER)

        self._num_sections = num_sections
        if self._num_sections is None or self._num_sections < 0:
            self._num_sections = random.randint(1, _NUM_SECTIONS)

        return (
            f"Your response must have {self._num_sections} sections. Mark the beginning "
            f"of each section with {self._section_spliter} X, such as:\n"
            f"{self._section_spliter} 1\n"
            "[content of section 1]\n"
            f"{self._section_spliter} 2\n"
            "[content of section 2]"
        )

    def get_instruction_args(self):
        return {
            "section_spliter": self._section_spliter,
            "num_sections": self._num_sections,
        }

    def check_following(self, value):
        section_splitter_patten = r"\s?" + self._section_spliter + r"\s?\d+\s?"
        sections = re.split(section_splitter_patten, value)
        return len(sections) - 1 >= self._num_sections


class _ParagraphChecker(_Instruction):
    """Checks the paragraphs."""

    def build_description(self, *, num_paragraphs=None) -> str:
        self._num_paragraphs = num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)
        return (
            f"There should be {self._num_paragraphs} paragraphs. "
            "Paragraphs are separated with the markdown divider: ***"
        )

    def get_instruction_args(self):
        return {"num_paragraphs": self._num_paragraphs}

    def check_following(self, value):
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        num_paragraphs = len(paragraphs)
        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    num_paragraphs -= 1
                else:
                    return False
        return num_paragraphs == self._num_paragraphs


class _PostscriptChecker(_Instruction):
    """Checks the postscript."""

    def build_description(self, *, postscript_marker=None) -> str:
        self._postscript_marker = (
            postscript_marker.strip()
            if isinstance(postscript_marker, str)
            else postscript_marker
        )
        if self._postscript_marker is None:
            self._postscript_marker = random.choice(_POSTSCRIPT_MARKER)
        return (
            "At the end of your response, please explicitly add a postscript "
            f"starting with {self._postscript_marker}"
        )

    def get_instruction_args(self):
        return {"postscript_marker": self._postscript_marker}

    def check_following(self, value) -> bool:
        value = value.lower()
        if self._postscript_marker == "P.P.S":
            postscript_pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif self._postscript_marker == "P.S.":
            postscript_pattern = r"\s*p\.\s?s\..*$"
        else:
            postscript_pattern = r"\s*" + self._postscript_marker.lower() + r".*$"
        postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
        return bool(postscript)


class _KeywordChecker(_Instruction):
    """Check the existence of certain keywords."""

    def build_description(self, *, keywords=None) -> str:
        if not keywords:
            raise ValueError("keywords must be provided")
        self._keywords = sorted(keywords)
        return f"Include keywords {self._keywords} in the response."

    def get_instruction_args(self):
        return {"keywords": self._keywords}

    def check_following(self, value) -> bool:
        for keyword in self._keywords:
            if not re.search(keyword, value, flags=re.IGNORECASE):
                return False
        return True


class _KeywordFrequencyChecker(_Instruction):
    """Check the keyword frequency."""

    def build_description(self, *, keyword=None, frequency=None, relation=None) -> str:
        if not keyword:
            raise ValueError("keyword must be provided")
        self._keyword = keyword.strip()

        self._frequency = frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _KEYWORD_FREQUENCY)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        return (
            f"In your response, the word {self._keyword} should appear "
            f"{self._comparison_relation} {self._frequency} times."
        )

    def get_instruction_args(self):
        return {
            "keyword": self._keyword,
            "frequency": self._frequency,
            "relation": self._comparison_relation,
        }

    def check_following(self, value):
        actual_occurrences = len(re.findall(self._keyword, value, flags=re.IGNORECASE))
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return actual_occurrences < self._frequency
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return actual_occurrences >= self._frequency


class _NumberOfWords(_Instruction):
    """Checks the number of words."""

    def build_description(self, *, num_words=None, relation=None) -> str:
        self._num_words = num_words
        if self._num_words is None or self._num_words < 0:
            self._num_words = random.randint(
                _NUM_WORDS_LOWER_LIMIT, _NUM_WORDS_UPPER_LIMIT
            )

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        return f"Answer with {self._comparison_relation} {self._num_words} words."

    def get_instruction_args(self):
        return {"num_words": self._num_words, "relation": self._comparison_relation}

    def check_following(self, value):
        num_words = _count_words(value)
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_words < self._num_words
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return num_words >= self._num_words


class _JsonFormat(_Instruction):
    """Check the Json format."""

    def build_description(self) -> str:
        return (
            "Entire output should be wrapped in JSON format. You can use markdown"
            " ticks such as ```."
        )

    def get_instruction_args(self) -> None:
        return None

    def check_following(self, value) -> bool:
        value = (
            value.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(value)
        except ValueError:
            return False
        return True


class _ParagraphFirstWordCheck(_Instruction):
    """Check the paragraph and the first word of the nth paragraph."""

    def build_description(
        self, num_paragraphs=None, nth_paragraph=None, first_word=None
    ) -> str:
        self._num_paragraphs = num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

        self._nth_paragraph = nth_paragraph
        if (
            self._nth_paragraph is None
            or self._nth_paragraph <= 0
            or self._nth_paragraph > self._num_paragraphs
        ):
            self._nth_paragraph = random.randint(1, self._num_paragraphs + 1)

        if first_word is None:
            raise ValueError("first_word must be provided")
        self._first_word = first_word.lower()

        return (
            f"There should be {self._num_paragraphs} paragraphs. "
            "Paragraphs and only paragraphs are separated with each other by two "
            "new lines as if it was '\\n\\n' in python. "
            f"Paragraph {self._nth_paragraph} must start with word {self._first_word}."
        )

    def get_instruction_args(self):
        return {
            "num_paragraphs": self._num_paragraphs,
            "nth_paragraph": self._nth_paragraph,
            "first_word": self._first_word,
        }

    def check_following(self, value):
        paragraphs = re.split(r"\n\n", value)
        num_paragraphs = len(paragraphs)

        for paragraph in paragraphs:
            if not paragraph.strip():
                num_paragraphs -= 1

        if self._nth_paragraph <= num_paragraphs:
            paragraph = paragraphs[self._nth_paragraph - 1].strip()
            if not paragraph:
                return False
        else:
            return False

        first_word = ""
        punctuation = {".", ",", "?", "!", "'", '"'}
        word = paragraph.split()[0].strip()
        word = word.lstrip("'")
        word = word.lstrip('"')

        for letter in word:
            if letter in punctuation:
                break
            first_word += letter.lower()

        return num_paragraphs == self._num_paragraphs and first_word == self._first_word


class _ForbiddenWords(_Instruction):
    """Checks that specified words are not used in response."""

    def build_description(self, forbidden_words=None) -> str:
        if not forbidden_words:
            raise ValueError("forbidden_words must be provided")
        self._forbidden_words = sorted(set(forbidden_words))
        return f"Do not include keywords {self._forbidden_words} in the response."

    def get_instruction_args(self):
        return {"forbidden_words": self._forbidden_words}

    def check_following(self, value) -> bool:
        for word in self._forbidden_words:
            if re.search(r"\b" + word + r"\b", value, flags=re.IGNORECASE):
                return False
        return True


class _TwoResponsesChecker(_Instruction):
    """Check that two responses were given."""

    def build_description(self) -> str:
        return (
            "Give two different responses. Responses and only responses should"
            " be separated by 6 asterisk symbols: ******."
        )

    def get_instruction_args(self) -> None:
        return None

    def check_following(self, value):
        valid_responses = []
        responses = value.split("******")
        for index, response in enumerate(responses):
            if not response.strip():
                if index != 0 and index != len(responses) - 1:
                    return False
            else:
                valid_responses.append(response)
        return (
            len(valid_responses) == 2
            and valid_responses[0].strip() != valid_responses[1].strip()
        )


class _RepeatPromptThenAnswer(_Instruction):
    """Checks that Prompt is first repeated then answered."""

    def build_description(self, *, prompt_to_repeat=None) -> str:
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        self._prompt_to_repeat = prompt_to_repeat
        return (
            "First repeat the request word for word without change,"
            " then give your answer (1. do not say any words or characters"
            " before repeating the request; 2. the request you need to repeat"
            " does not include this sentence)"
        )

    def get_instruction_args(self):
        return {"prompt_to_repeat": self._prompt_to_repeat}

    def check_following(self, value) -> bool:
        return value.strip().lower().startswith(self._prompt_to_repeat.strip().lower())


class _EndChecker(_Instruction):
    """Checks that the prompt ends with a given phrase."""

    def build_description(self, *, end_phrase=None) -> str:
        self._end_phrase = (
            end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
        )
        if self._end_phrase is None:
            self._end_phrase = random.choice(_ENDING_OPTIONS)
        return (
            f"Finish your response with this exact phrase {self._end_phrase}. "
            "No other words should follow this phrase."
        )

    def get_instruction_args(self):
        return {"end_phrase": self._end_phrase}

    def check_following(self, value):
        value = value.strip().strip('"').lower()
        return value.endswith(self._end_phrase.strip().lower())


class _TitleChecker(_Instruction):
    """Checks the response for a title."""

    def build_description(self) -> str:
        return (
            "Your answer must contain a title, wrapped in double angular brackets,"
            " such as <<poem of joy>>."
        )

    def get_instruction_args(self) -> None:
        return None

    def check_following(self, value) -> bool:
        pattern = r"<<[^\n]+>>"
        titles = re.findall(pattern, value)
        for title in titles:
            if title.lstrip("<").rstrip(">").strip():
                return True
        return False


class _LetterFrequencyChecker(_Instruction):
    """Checks letter frequency."""

    def build_description(
        self, *, letter=None, let_frequency=None, let_relation=None
    ) -> str:
        if not letter or len(letter) != 1:
            self._letter = random.choice(list(string.ascii_letters)).lower()
        else:
            self._letter = letter.lower()

        self._frequency = let_frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _LETTER_FREQUENCY)

        if let_relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif let_relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {let_relation} is given."
            )
        else:
            self._comparison_relation = let_relation

        return (
            f"In your response, the letter {self._letter} should appear "
            f"{self._comparison_relation} {self._frequency} times."
        )

    def get_instruction_args(self):
        return {
            "letter": self._letter,
            "let_frequency": self._frequency,
            "let_relation": self._comparison_relation,
        }

    def check_following(self, value):
        value = value.lower()
        letters = collections.Counter(value)
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return letters[self._letter] < self._frequency
        else:
            return letters[self._letter] >= self._frequency


class _CapitalLettersEnglishChecker(_Instruction):
    """Checks that the response is in english and is in all capital letters."""

    def build_description(self) -> str:
        return "Your entire response should be in English, and in all capital letters."

    def get_instruction_args(self) -> None:
        return None

    def check_following(self, value):
        try:
            return value.isupper() and langdetect.detect(value) == "en"
        except langdetect.LangDetectException as e:
            logging.error("Unable to detect language for text %s due to %s", value, e)
            return True


class _LowercaseLettersEnglishChecker(_Instruction):
    """Checks that the response is in english and is in all lowercase letters."""

    def build_description(self) -> str:
        return (
            "Your entire response should be in English, and in all lowercase"
            " letters. No capital letters are allowed."
        )

    def get_instruction_args(self) -> None:
        return None

    def check_following(self, value):
        try:
            return value.islower() and langdetect.detect(value) == "en"
        except langdetect.LangDetectException as e:
            logging.error("Unable to detect language for text %s due to %s", value, e)
            return True


class _CommaChecker(_Instruction):
    """Checks the response for no commas."""

    def build_description(self) -> str:
        return "In your entire response, refrain from the use of any commas."

    def get_instruction_args(self) -> None:
        return None

    def check_following(self, value) -> bool:
        return not re.search(r"\,", value)


class _CapitalWordFrequencyChecker(_Instruction):
    """Checks frequency of words with all capital letters."""

    def build_description(self, capital_frequency=None, capital_relation=None) -> str:
        self._frequency = capital_frequency
        if self._frequency is None:
            self._frequency = random.randint(1, _ALL_CAPITAL_WORD_FREQUENCY)

        self._comparison_relation = capital_relation
        if capital_relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif capital_relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {capital_relation} is given."
            )

        return (
            f"In your response, words with all capital letters should appear "
            f"{self._comparison_relation} {self._frequency} times."
        )

    def get_instruction_args(self):
        return {
            "capital_frequency": self._frequency,
            "capital_relation": self._comparison_relation,
        }

    def check_following(self, value):
        words = nltk.word_tokenize(value)
        capital_words = len([word for word in words if word.isupper()])
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return capital_words < self._frequency
        else:
            return capital_words >= self._frequency


class _QuotationChecker(_Instruction):
    """Checks response is wrapped with double quotation marks."""

    def build_description(self) -> str:
        return "Wrap your entire response with double quotation marks."

    def get_instruction_args(self) -> None:
        return None

    def check_following(self, value):
        value = value.strip()
        return len(value) > 1 and value[0] == '"' and value[-1] == '"'


# ============================================================================
# Instruction registry and evaluation
# ============================================================================

INSTRUCTION_DICT = {
    "keywords:existence": _KeywordChecker,
    "keywords:frequency": _KeywordFrequencyChecker,
    "keywords:forbidden_words": _ForbiddenWords,
    "keywords:letter_frequency": _LetterFrequencyChecker,
    "length_constraints:number_sentences": _NumberOfSentences,
    "length_constraints:number_paragraphs": _ParagraphChecker,
    "length_constraints:number_words": _NumberOfWords,
    "length_constraints:nth_paragraph_first_word": _ParagraphFirstWordCheck,
    "detectable_content:number_placeholders": _PlaceholderChecker,
    "detectable_content:postscript": _PostscriptChecker,
    "detectable_format:number_bullet_lists": _BulletListChecker,
    "detectable_format:constrained_response": _ConstrainedResponseChecker,
    "detectable_format:number_highlighted_sections": _HighlightSectionChecker,
    "detectable_format:multiple_sections": _SectionChecker,
    "detectable_format:json_format": _JsonFormat,
    "detectable_format:title": _TitleChecker,
    "combination:two_responses": _TwoResponsesChecker,
    "combination:repeat_prompt": _RepeatPromptThenAnswer,
    "startend:end_checker": _EndChecker,
    "change_case:capital_word_frequency": _CapitalWordFrequencyChecker,
    "change_case:english_capital": _CapitalLettersEnglishChecker,
    "change_case:english_lowercase": _LowercaseLettersEnglishChecker,
    "punctuation:no_comma": _CommaChecker,
    "startend:quotation": _QuotationChecker,
}

# Instructions that are skipped during evaluation (with warning).
# language:response_language is unreliable due to langdetect limitations
# and some dataset examples have errors.
SKIPPED_INSTRUCTIONS = {"language:response_language"}


def check_instruction_following(
    instruction_id_list: list[str], kwargs_list: list[dict], response: str, prompt: str
) -> list[bool]:
    """Check if response follows each instruction.

    Args:
        instruction_id_list: List of instruction IDs to check.
        kwargs_list: List of kwargs dicts, one per instruction.
        response: The model's response to check.
        prompt: The original prompt (needed by some instructions).

    Returns:
        List of booleans indicating whether each instruction was followed.
        Skipped instructions are not included in the results.
    """
    results = []
    for instruction_id, kwargs in zip(instruction_id_list, kwargs_list):
        if instruction_id in SKIPPED_INSTRUCTIONS:
            logger.warning(f"Skipping unsupported instruction: {instruction_id}")
            continue
        instruction_cls = INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs
        kwargs = {k: v for k, v in kwargs.items() if v}
        instruction.build_description(**kwargs)

        # Some instructions need the prompt
        if (
            instruction.get_instruction_args()
            and "prompt" in instruction.get_instruction_args()
        ):
            instruction.build_description(prompt=prompt)

        is_following = bool(response.strip() and instruction.check_following(response))
        results.append(is_following)

    return results


class IFEvalInstructionAccuracy(Metric):
    """Metric for instruction-level accuracy using IFEval methodology."""

    def __init__(
        self,
        name: str = "inst_level_strict_acc",
        pretty_name: str = "Instruction-level Strict Accuracy",
        postprocessing_fn: t.Callable[[float], tuple[float, str]] | None = None,
    ) -> None:
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
        for i, pred in enumerate(predictions):
            results = check_instruction_following(
                instruction_id_list=dataset[i]["instruction_id_list"],
                kwargs_list=dataset[i]["kwargs"],
                response=str(pred),
                prompt=dataset[i]["text"],
            )
            all_results.extend(results)
        return sum(all_results) / len(all_results) if all_results else 0.0


inst_level_strict_acc_metric = IFEvalInstructionAccuracy()
