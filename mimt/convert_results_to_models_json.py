"""Convert EuroEval JSONL benchmark results into the ``models.json`` format.

Reads a EuroEval results file such as ``euroeval_benchmark_results.jsonl`` and emits a
JSON object keyed by model name, matching the structure of ``models.json``:

```json
{
  "<model-name>": {
    "name": {"is": "...", "en": "..."},
    "tableName": {"is": "...", "en": "..."},
    "description": {"is": "...", "en": "..."},
    "modelGroup": "<group-id>",
    "results": [
      {"benchmark": "<benchmark-id>", "score": 0.123, "metric": "<metric-id>"}
    ]
  }
}
```

Each score is an accuracy, rescaled from the range that the record declares for it to
the 0-1 range used by ``models.json``. The classification datasets do not report an
accuracy, so macro-average F1 stands in for them; see ``METRIC_PREFERENCE`` below. Use
``--metric`` to force a single named metric for every dataset instead.

Because the metric therefore varies by benchmark, every result carries the EuroEval name
of the metric it was scored on - ``accuracy``, ``macro_f1`` or ``mcc`` - so that it can
be labelled rather than left implicit. Scores on different metrics are not comparable to
each other.

Note that this is not the score EuroEval itself leads with. Its primary metric for all
of these datasets is Matthew's Correlation Coefficient, which is a correlation rather
than a proportion, so a model guessing at random scores 0 on it rather than 0.5, making
it read as much lower for the same performance.

The output is a standalone file, containing only the models in the results file, and it
never reads from or writes to ``models.json``. Model names, their canonical spelling and
their model group are taken from ``modelGroups.json``, and benchmark ordering is taken
from ``benchmarks.json``. Display names and descriptions that have been edited in a
previous output file are reused, so re-running this script keeps manual edits.

Usage::

    uv run mimt/convert_results_to_models_json.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
import typing as t
from pathlib import Path

import click

MIMT_DIR = Path(__file__).resolve().parent

# The default output file. Deliberately not `models.json`, as the output holds only the
# models in the results file.
OUTPUT_FILENAME = "euroeval_models.json"

# EuroEval dataset names that differ from their benchmark ID in `benchmarks.json`.
# Datasets not listed here use the same ID in both places.
DATASET_TO_BENCHMARK: dict[str, str] = {"ice-linguistic": "iceling-is"}

# The metrics to use, in order of preference, with the first one a record reports being
# the one used. Accuracy is preferred, but the classification datasets do not report it,
# so macro-average F1 stands in for them. Matthew's Correlation Coefficient, which
# EuroEval reports as the primary metric of all of these datasets, is deliberately last:
# it is a correlation, so a model guessing at random scores 0 rather than 0.5 on it,
# which makes it read as far lower than an accuracy for the same performance.
METRIC_PREFERENCE: tuple[str, ...] = ("test_accuracy", "test_macro_f1", "test_mcc")

# The split prefixes that EuroEval puts on the metric names it reports, stripped off to
# recover the metric's own name, e.g. `test_accuracy` -> `accuracy`.
SPLIT_PREFIXES: tuple[str, ...] = ("test_", "val_", "validation_")

# Tokens in a model ID that carry no information in a display name.
NOISE_TOKENS = frozenset({"instruct", "it", "latest", "preview", "turbo"})

# Lower-case tokens with a fixed capitalisation in display names.
ACRONYMS: dict[str, str] = {"glm": "GLM", "gpt": "GPT", "oss": "OSS"}

# Number of decimals to round the normalised scores to, matching `models.json`.
SCORE_DECIMALS = 3

EMPTY_LOCALISED: dict[str, str] = {"is": "", "en": ""}

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr
)
logger = logging.getLogger(__name__)

JsonDict = dict[str, t.Any]


class Score(t.NamedTuple):
    """A model's score on a single benchmark.

    Attributes:
        timestamp:
            The timestamp of the evaluation the score came from, used to keep the newest
            of several scores for the same model and benchmark.
        metric:
            The EuroEval name of the metric the score is on, e.g. `accuracy`.
        value:
            The score itself, normalised to the 0-1 range.
    """

    timestamp: str
    metric: str
    value: float


@click.command()
@click.option(
    "--results",
    "results_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=MIMT_DIR / "euroeval_benchmark_results.jsonl",
    show_default=True,
    help="The JSONL file with the EuroEval benchmark results.",
)
@click.option(
    "--model-groups",
    "model_groups_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=MIMT_DIR / "modelGroups.json",
    show_default=True,
    help="The `modelGroups.json` file, mapping model groups to their models.",
)
@click.option(
    "--benchmarks",
    "benchmarks_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=MIMT_DIR / "benchmarks.json",
    show_default=True,
    help="The `benchmarks.json` file, defining the known benchmarks and their order.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, allow_dash=True, path_type=Path),
    default=MIMT_DIR / OUTPUT_FILENAME,
    show_default=True,
    help="Where to write the resulting JSON. Use '-' to write it to stdout.",
)
@click.option(
    "--reference",
    "reference_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="A JSON file in this script's output format, whose display names and "
    "descriptions are reused.  [default: the output file, if it exists]",
)
@click.option(
    "--metric",
    default=None,
    help="Force a single metric for every dataset, e.g. 'test_mcc'. Defaults to the "
    "first metric of `METRIC_PREFERENCE` that each dataset reports.",
)
def main(
    results_path: Path,
    model_groups_path: Path,
    benchmarks_path: Path,
    output_path: Path,
    reference_path: Path | None,
    metric: str | None,
) -> None:
    """Convert EuroEval JSONL benchmark results into the `models.json` format.

    Args:
        results_path:
            The JSONL file with the EuroEval benchmark results.
        model_groups_path:
            The `modelGroups.json` file, mapping model groups to their models.
        benchmarks_path:
            The `benchmarks.json` file, defining the known benchmarks and their order.
        output_path:
            Where to write the resulting JSON, or `-` to write it to stdout.
        reference_path:
            A JSON file in this script's output format, whose display names and
            descriptions are reused. Defaults to the output file. Ignored if it does not
            exist.
        metric:
            A single metric to use for all datasets, or None to use the first metric of
            `METRIC_PREFERENCE` that each dataset reports.
    """
    to_stdout = str(output_path) == "-"
    if reference_path is None and not to_stdout:
        reference_path = output_path

    records = load_records(path=results_path)
    group_lookup = build_group_lookup(path=model_groups_path)
    benchmark_order = list(load_json_object(path=benchmarks_path))
    reference = (
        load_json_object(path=reference_path)
        if reference_path is not None and reference_path.exists()
        else {}
    )

    models = build_models(
        records=records,
        group_lookup=group_lookup,
        benchmark_order=benchmark_order,
        reference=reference,
        metric=metric,
    )

    output = json.dumps(models, indent=4, ensure_ascii=False) + "\n"
    if to_stdout:
        sys.stdout.write(output)
    else:
        output_path.write_text(output, encoding="utf-8")
    logger.info(
        f"Converted {len(records):,} records into {len(models):,} models, written to "
        f"{'stdout' if to_stdout else output_path}."
    )


def load_records(path: Path) -> list[JsonDict]:
    """Load the evaluation records from a JSONL file.

    Args:
        path:
            The path to the JSONL file.

    Returns:
        The records in the file, in file order.

    Raises:
        ClickException:
            If a non-empty line is not valid JSON.
    """
    records: list[JsonDict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise click.ClickException(f"{path}:{line_no}: invalid JSON: {e}") from e
    return records


def build_group_lookup(path: Path) -> dict[str, tuple[str, str]]:
    """Build a lookup from model name to its canonical spelling and model group.

    Args:
        path:
            The path to the `modelGroups.json` file.

    Returns:
        A mapping from case-folded model name to a `(canonical name, group ID)` pair.
    """
    groups = load_json_object(path=path)
    return {
        model_name.casefold(): (model_name, group_id)
        for group_id, group in groups.items()
        for model_name in group.get("models", [])
    }


def load_json_object(path: Path) -> JsonDict:
    """Load a JSON object from a file.

    Args:
        path:
            The path to the JSON file.

    Returns:
        The parsed JSON object.

    Raises:
        ClickException:
            If the file does not contain a JSON object.
    """
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise click.ClickException(f"{path}: expected a JSON object.")
    return content


def build_models(
    records: list[JsonDict],
    group_lookup: dict[str, tuple[str, str]],
    benchmark_order: list[str],
    reference: JsonDict,
    metric: str | None,
) -> JsonDict:
    """Build the `models.json` entries for all models appearing in the records.

    Args:
        records:
            The evaluation records.
        group_lookup:
            A mapping from case-folded model name to a `(canonical name, group ID)`
            pair.
        benchmark_order:
            The known benchmark IDs, in the order they should appear in the results.
        reference:
            A previous output object, whose edited names and descriptions are reused for
            models that appear in it.
        metric:
            A single metric to use for all datasets, or None to use the first metric of
            `METRIC_PREFERENCE` that each dataset reports.

    Returns:
        The models, keyed by model name and ordered by `group_lookup`.
    """
    # Keyed by model name, then benchmark ID, storing the newest score seen for the pair
    scores: dict[str, dict[str, Score]] = {}
    groups: dict[str, str] = {}
    # The metric names used for each benchmark, so that they can be reported
    metrics_used: dict[str, set[str]] = {}

    for record in records:
        model_name, group_id = resolve_model(record=record, group_lookup=group_lookup)
        benchmark_id = resolve_benchmark(record=record, benchmark_order=benchmark_order)
        extracted = extract_score(record=record, metric=metric)
        if benchmark_id is None or extracted is None:
            continue
        metric_used, value = extracted
        groups[model_name] = group_id
        metrics_used.setdefault(benchmark_id, set()).add(metric_used)
        score = Score(
            timestamp=str(record.get("evaluation_timestamp", "")),
            metric=metric_used,
            value=value,
        )
        previous = scores.setdefault(model_name, {}).get(benchmark_id)
        if previous is None or previous.timestamp <= score.timestamp:
            scores[model_name][benchmark_id] = score

    for benchmark_id in benchmark_order:
        if benchmark_id in metrics_used:
            names = ", ".join(sorted(metrics_used[benchmark_id]))
            logger.info(f"Scoring the benchmark {benchmark_id!r} on {names}.")

    models: JsonDict = {}
    for model_name in sort_model_names(model_names=scores, group_lookup=group_lookup):
        models[model_name] = build_model_entry(
            model_name=model_name,
            group_id=groups[model_name],
            benchmark_scores=scores[model_name],
            benchmark_order=benchmark_order,
            reference_entry=reference.get(model_name, {}),
        )
    return models


def resolve_model(
    record: JsonDict, group_lookup: dict[str, tuple[str, str]]
) -> tuple[str, str]:
    """Resolve the model name and model group of a record.

    The model ID in a record may carry a LiteLLM provider prefix, such as
    `together_ai/google/gemma-4-31B-it`, whereas `models.json` is keyed on the bare
    model name.

    Args:
        record:
            The evaluation record.
        group_lookup:
            A mapping from case-folded model name to a `(canonical name, group ID)`
            pair.

    Returns:
        The canonical model name and its model group ID. The group ID is an empty string
        if the model cannot be assigned to a group.
    """
    model_id = str(record["model_info"]["id"])
    segments = model_id.split("/")
    model_name = segments[-1]

    known = group_lookup.get(model_name.casefold())
    if known is not None:
        return known

    # Fall back on the organisation segment of the model ID, e.g. the `google` of
    # `together_ai/google/gemma-4-31B-it`
    group_ids = {group_id for _, group_id in group_lookup.values()}
    fallback = next(
        (seg for seg in reversed(segments[:-1]) if seg.casefold() in group_ids), ""
    )
    logger.warning(
        f"The model {model_id!r} is not listed in the model groups file; using the "
        f"group {fallback!r}. Add it to the file to set its group explicitly."
    )
    return model_name, fallback


def resolve_benchmark(record: JsonDict, benchmark_order: list[str]) -> str | None:
    """Resolve the benchmark ID that a record was evaluated on.

    Args:
        record:
            The evaluation record.
        benchmark_order:
            The known benchmark IDs.

    Returns:
        The benchmark ID, or None if the record has no dataset.
    """
    details = record.get("eval_library", {}).get("additional_details", {})
    dataset = details.get("dataset")
    if not dataset:
        logger.warning(
            f"Skipping the record {record.get('evaluation_id')!r}, as it has no "
            "dataset."
        )
        return None
    benchmark_id = DATASET_TO_BENCHMARK.get(dataset, dataset)
    if benchmark_id not in benchmark_order:
        logger.warning(
            f"The dataset {dataset!r} maps to the benchmark {benchmark_id!r}, which is "
            "not in the benchmarks file. Add it there, or map it in "
            "`DATASET_TO_BENCHMARK`."
        )
    return benchmark_id


def extract_score(record: JsonDict, metric: str | None) -> tuple[str, float] | None:
    """Extract the score of a record, normalised to the 0-1 range.

    Args:
        record:
            The evaluation record.
        metric:
            The name of the metric to extract, or None to use the first metric in
            `METRIC_PREFERENCE` that the record reports.

    Returns:
        The name of the metric used and its normalised score, or None if the record
        reports no usable metric.
    """
    results = record.get("evaluation_results") or []
    by_name = {r.get("evaluation_name"): r for r in results}
    wanted = (metric,) if metric is not None else METRIC_PREFERENCE
    result = next((by_name[name] for name in wanted if name in by_name), None)
    if result is None:
        wanted_str = " or ".join(repr(name) for name in wanted)
        logger.warning(
            f"Skipping the record {record.get('evaluation_id')!r}, as it reports "
            f"{sorted(by_name)} and none of them is {wanted_str}."
        )
        return None

    config = result.get("metric_config", {})
    if config.get("lower_is_better"):
        logger.warning(
            f"The metric {result.get('evaluation_name')!r} of the record "
            f"{record.get('evaluation_id')!r} is lower-is-better, but its score is "
            "copied as-is."
        )
    minimum, maximum = config.get("min_score", 0), config.get("max_score", 100)
    score = float(result["score_details"]["score"])
    if maximum > minimum:
        score = (score - minimum) / (maximum - minimum)
    score = round(score, SCORE_DECIMALS)
    # MCC declares a 0-100 range but actually spans -100 to 100, so a model doing worse
    # than chance rescales to a negative score. That is reported rather than clamped, as
    # clamping would hide the result.
    if not 0.0 <= score <= 1.0:
        logger.warning(
            f"The metric {result.get('evaluation_name')!r} of the record "
            f"{record.get('evaluation_id')!r} rescales to {score}, which is outside "
            "the 0-1 range."
        )
    return metric_name(evaluation_name=str(result["evaluation_name"])), score


def metric_name(evaluation_name: str) -> str:
    """Recover a metric's own name from the name a record reports it under.

    Args:
        evaluation_name:
            The name in the record, which carries a split prefix, e.g. `test_accuracy`.

    Returns:
        The metric's EuroEval name, e.g. `accuracy`.
    """
    for prefix in SPLIT_PREFIXES:
        if evaluation_name.startswith(prefix):
            return evaluation_name.removeprefix(prefix)
    return evaluation_name


def sort_model_names(
    model_names: t.Iterable[str], group_lookup: dict[str, tuple[str, str]]
) -> list[str]:
    """Sort model names by the order they appear in the model groups file.

    Args:
        model_names:
            The model names to sort.
        group_lookup:
            A mapping from case-folded model name to a `(canonical name, group ID)`
            pair.

    Returns:
        The sorted model names, with the ones missing from the model groups file last,
        in alphabetical order.
    """
    order = {name: idx for idx, (name, _) in enumerate(group_lookup.values())}
    return sorted(model_names, key=lambda name: (order.get(name, len(order)), name))


def build_model_entry(
    model_name: str,
    group_id: str,
    benchmark_scores: dict[str, Score],
    benchmark_order: list[str],
    reference_entry: JsonDict,
) -> JsonDict:
    """Build the `models.json` entry for a single model.

    Args:
        model_name:
            The canonical model name.
        group_id:
            The ID of the model's model group.
        benchmark_scores:
            The model's scores, keyed by benchmark ID.
        benchmark_order:
            The known benchmark IDs, in the order they should appear in the results.
        reference_entry:
            The model's entry in a previous output object, or an empty dict if it has
            none. Names and descriptions in it take precedence.

    Returns:
        The `models.json` entry for the model.
    """
    order = {benchmark_id: idx for idx, benchmark_id in enumerate(benchmark_order)}
    results = [
        {
            "benchmark": benchmark_id,
            "score": benchmark_scores[benchmark_id].value,
            "metric": benchmark_scores[benchmark_id].metric,
        }
        for benchmark_id in sorted(
            benchmark_scores, key=lambda bid: (order.get(bid, len(order)), bid)
        )
    ]
    table_name = pretty_model_name(model_name=model_name)
    return {
        "name": reference_entry.get("name", {"is": model_name, "en": model_name}),
        "tableName": reference_entry.get(
            "tableName", {"is": table_name, "en": table_name}
        ),
        "description": reference_entry.get("description", dict(EMPTY_LOCALISED)),
        "modelGroup": group_id,
        "results": results,
    }


def pretty_model_name(model_name: str) -> str:
    """Build a human-readable display name from a model name.

    Release dates and boilerplate such as `Instruct` are dropped, dash-separated version
    numbers are joined with a dot, and known acronyms are capitalised. E.g.
    `claude-haiku-4-5` becomes `Claude Haiku 4.5`.

    Args:
        model_name:
            The model name, without any provider prefix.

    Returns:
        The display name, falling back on the model name if nothing is left of it.
    """
    tokens: list[str] = []
    for token in re.split(r"[-_]", model_name):
        if not token or token.casefold() in NOISE_TOKENS:
            continue
        # A date stamp, such as `20250514`, `2025` or `06`. Everything after it is part
        # of the same date, so the name ends here.
        if token.isdigit() and (len(token) >= 4 or token.startswith("0")):
            break
        tokens.append(format_name_token(token=token))
    return " ".join(join_version_tokens(tokens=tokens)) or model_name


def format_name_token(token: str) -> str:
    """Capitalise a single token of a display name.

    Args:
        token:
            The token to capitalise.

    Returns:
        The capitalised token.
    """
    if token.casefold() in ACRONYMS:
        return ACRONYMS[token.casefold()]
    # A parameter count, such as `120b` or `2.4t`
    if re.fullmatch(r"\d+(\.\d+)?[bmtk]", token, flags=re.IGNORECASE):
        return token[:-1] + token[-1].upper()
    # Tokens with digits, such as `3.5`, `E4B` and `3n`, already have their intended
    # capitalisation, as do tokens that are not plain lower-case, such as `DeepSeek`
    if any(char.isdigit() for char in token) or not token.islower():
        return token
    return token.capitalize()


def join_version_tokens(tokens: list[str]) -> list[str]:
    """Join consecutive short numeric tokens into a dotted version number.

    E.g. the `4` and `5` of `claude-haiku-4-5` become `4.5`.

    Args:
        tokens:
            The tokens of a display name.

    Returns:
        The tokens, with consecutive short numeric ones joined.
    """
    joined: list[str] = []
    for token in tokens:
        is_short_number = token.isdigit() and len(token) <= 2
        if is_short_number and joined and joined[-1].isdigit() and len(joined[-1]) <= 2:
            joined[-1] = f"{joined[-1]}.{token}"
        else:
            joined.append(token)
    return joined


if __name__ == "__main__":
    main()
