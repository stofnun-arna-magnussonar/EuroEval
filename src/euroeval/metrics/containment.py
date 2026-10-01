"""Metrics based on whether the reference text occurs in the model output."""

import collections.abc as c
import re
import typing as t
import unicodedata
from collections import defaultdict

from ..exceptions import InvalidBenchmark
from .base import Metric

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig


class ReferenceContainmentMetric(Metric):
    """The fraction of model outputs that contain their reference text.

    Texts are NFC-normalised and have their whitespace collapsed, but are otherwise
    compared verbatim, so casing, punctuation, quotation marks and dashes all matter.

    If a reference already occurs in its own input text, e.g. when the correction only
    deletes a trailing character, then an output merely repeating the input would
    contain the reference. Such samples therefore require an exact match instead.
    """

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Calculate the metric score.

        Args:
            predictions:
                The model predictions.
            references:
                The ground truth references.
            dataset:
                The dataset used for evaluation, used to look up the input texts.
            dataset_config:
                The dataset configuration. This is not used.
            benchmark_config:
                The benchmark configuration. This is not used.

        Returns:
            The calculated metric score, or None if the score should be ignored.

        Raises:
            InvalidBenchmark:
                If the number of predictions does not match the number of references.
        """
        if not predictions or not references:
            return None
        elif len(predictions) != len(references):
            raise InvalidBenchmark(
                f"The number of predictions ({len(predictions):,}) does not match the "
                f"number of references ({len(references):,})."
            )

        # The dataset rows are not aligned with the predictions, as cached samples are
        # moved to the end, so we look up the input texts by reference text instead
        inputs_by_reference: dict[str, list[str]] = defaultdict(list)
        if {"text", "target_text"}.issubset(dataset.column_names):
            for text, target_text in zip(dataset["text"], dataset["target_text"]):
                inputs_by_reference[normalise_text(text=target_text)].append(
                    normalise_text(text=text)
                )

        num_correct = 0
        for prediction, reference in zip(predictions, references):
            prediction = normalise_text(text=str(prediction))
            reference = normalise_text(text=str(reference))
            if any(reference in text for text in inputs_by_reference[reference]):
                num_correct += prediction == reference
            else:
                num_correct += reference in prediction
        return num_correct / len(predictions)


def normalise_text(text: str) -> str:
    """NFC-normalise a text and collapse its whitespace.

    Args:
        text:
            The text to normalise.

    Returns:
        The normalised text.
    """
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


reference_containment_metric = ReferenceContainmentMetric(
    name="reference_containment", pretty_name="Reference Containment"
)
