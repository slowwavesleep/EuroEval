"""Templates for the Instruction Following task."""

import typing as t

from ..data_models import PromptConfig
from ..languages import (
    BOSNIAN,
    CATALAN,
    CZECH,
    DANISH,
    DUTCH,
    ENGLISH,
    ESTONIAN,
    FINNISH,
    FRENCH,
    GERMAN,
    GREEK,
    HUNGARIAN,
    ICELANDIC,
    ITALIAN,
    LATVIAN,
    LITHUANIAN,
    NORWEGIAN,
    NORWEGIAN_BOKMÅL,
    NORWEGIAN_NYNORSK,
    POLISH,
    PORTUGUESE,
    ROMANIAN,
    SERBIAN,
    SPANISH,
    SWEDISH,
    UKRAINIAN,
)

if t.TYPE_CHECKING:
    from ..languages import Language

INSTRUCTION_FOLLOWING_TEMPLATES: dict["Language", PromptConfig] = {
    ENGLISH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="",
        default_prompt_label_mapping="auto",
    ),
}