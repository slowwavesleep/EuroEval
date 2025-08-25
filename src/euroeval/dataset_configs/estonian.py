"""All Estonian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import ET
from ..tasks import SENT, COMMON_SENSE

### Official datasets ###

ESTONIAN_VALENCE_CONFIG = DatasetConfig(
    name="estonian-valence",
    pretty_name="the Estonian sentiment classification dataset Estonian Valence",
    huggingface_id="EuroEval/estonian-valence",
    task=SENT,
    languages=[ET],
)

WINOGRANDE_ET_CONFIG = DatasetConfig(
    name="winogrande-et",
    pretty_name="the Estonian Winogrande dataset",
    huggingface_id="EuroEval/winogrande_et",
    task=COMMON_SENSE,
    languages=[ET],
    # requires custom templates as WinoGrande is different from
    # the usual multiple choice tasks
    _prompt_prefix="Sulle esitatakse lüngaga (_) tekstülesanne "
    "ja kaks vastusevarianti (a ja b).\n",
    # includes the question and the options
    _prompt_template="Tekstülesanne: {text}\n\nVastus: {label}",
    _instruction_prompt="Tekstülesanne: {text}\n\n"
    "Sinu ülesanne on valida lünka sobiv vastusevariant. "
    "Vasta ainult {labels_str}. Muud vastused ei ole lubatud.",
)
