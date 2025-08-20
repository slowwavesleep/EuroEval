"""All Estonian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import ET
from ..tasks import SENT

### Unofficial datasets ###

ESTONIAN_VALENCE_CONFIG = DatasetConfig(
    name="estonian-valence",
    pretty_name="the Estonian valence corpus reorganized for EuroEval",
    huggingface_id="EuroEval/estonian-valence-corpus-euroeval",
    task=SENT,
    languages=[ET],
    unofficial=True,
)


