"""IFEval instruction-following metrics."""

import collections.abc as c
import typing as t

from .base import Metric
from .utils import (
    InputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig


class IFEvalMetric(Metric):
    """Metric for evaluating instruction-following using IFEval methodology.

    This metric checks whether model outputs follow specific formatting and content
    instructions (e.g., word count, language, JSON format, bullet points).

    Attributes:
        name:
            The name of the metric in snake_case.
        pretty_name:
            The pretty name of the metric, used for display purposes.
        strict:
            If True, use strict evaluation. If False, use loose evaluation which
            tries variations of the response (removing first/last lines, asterisks).
        prompt_level:
            If True, compute prompt-level accuracy (did the model follow ALL
            instructions for each prompt). If False, compute instruction-level
            accuracy (fraction of individual instructions followed across all prompts).
    """

    def __init__(
        self,
        name: str,
        pretty_name: str,
        strict: bool,
        prompt_level: bool,
        postprocessing_fn: t.Callable[[float], tuple[float, str]] | None = None,
    ) -> None:
        """Initialise the IFEval metric.

        Args:
            name:
                The name of the metric in snake_case.
            pretty_name:
                The pretty name of the metric, used for display purposes.
            strict:
                If True, use strict evaluation. If False, use loose evaluation.
            prompt_level:
                If True, compute prompt-level accuracy. If False, compute
                instruction-level accuracy.
            postprocessing_fn:
                A function to apply to the metric scores after they are computed.
        """
        super().__init__(
            name=name, pretty_name=pretty_name, postprocessing_fn=postprocessing_fn
        )
        self.strict = strict
        self.prompt_level = prompt_level

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Calculate the IFEval metric score.

        Args:
            predictions:
                The model predictions (generated responses).
            references:
                Not used for IFEval - instructions come from the dataset.
            dataset:
                The dataset containing instruction metadata (instruction_id_list,
                kwargs, prompt, key).
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration.

        Returns:
            The calculated metric score, or None if the score should be ignored.
        """
        test_fn = (
            test_instruction_following_strict
            if self.strict
            else test_instruction_following_loose
        )

        if self.prompt_level:
            # Prompt-level: did the model follow ALL instructions for each prompt?
            scores: list[bool] = []
            for i, pred in enumerate(predictions):
                inp = InputExample(
                    key=dataset[i].get("key", i),
                    instruction_id_list=dataset[i]["instruction_id_list"],
                    prompt=dataset[i]["text"],
                    kwargs=dataset[i]["kwargs"],
                )
                out = test_fn(inp, str(pred))
                scores.append(out.follow_all_instructions)
            return sum(scores) / len(scores) if scores else 0.0
        else:
            # Instruction-level: fraction of individual instructions followed
            all_instruction_results: list[bool] = []
            for i, pred in enumerate(predictions):
                inp = InputExample(
                    key=dataset[i].get("key", i),
                    instruction_id_list=dataset[i]["instruction_id_list"],
                    prompt=dataset[i]["text"],
                    kwargs=dataset[i]["kwargs"],
                )
                out = test_fn(inp, str(pred))
                all_instruction_results.extend(out.follow_instruction_list)
            return (
                sum(all_instruction_results) / len(all_instruction_results)
                if all_instruction_results
                else 0.0
            )


# Pre-defined metric instances
prompt_level_strict_acc_metric = IFEvalMetric(
    name="prompt_level_strict_acc",
    pretty_name="Prompt-level Strict Accuracy",
    strict=True,
    prompt_level=True,
)

inst_level_strict_acc_metric = IFEvalMetric(
    name="inst_level_strict_acc",
    pretty_name="Instruction-level Strict Accuracy",
    strict=True,
    prompt_level=False,
)

prompt_level_loose_acc_metric = IFEvalMetric(
    name="prompt_level_loose_acc",
    pretty_name="Prompt-level Loose Accuracy",
    strict=False,
    prompt_level=True,
)

inst_level_loose_acc_metric = IFEvalMetric(
    name="inst_level_loose_acc",
    pretty_name="Instruction-level Loose Accuracy",
    strict=False,
    prompt_level=False,
)
