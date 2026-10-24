# Spelling Standardisation

## 📚 Overview

Spelling standardisation is the task of bringing a sentence in line with the official
spelling rules of a language. The model is presented with a sentence that may or may not
break a spelling or punctuation rule, such as the use of capital letters, compound
words, hyphens or commas, and has to write out the sentence in its standardised form.

This task is only evaluated zero-shot, and only instruction-tuned and reasoning models
can be evaluated on it. When evaluating generative models, we allow the model to generate
256 tokens on this task.

## 📊 Metrics

The primary metric used to evaluate the performance of a model on the spelling
standardisation task is reference containment, being the percentage of model outputs
that contain the reference sentence. Whitespace is normalised, but the comparison is
otherwise verbatim, so casing, punctuation, quotation marks and dashes all have to be
correct. This allows the model to wrap its answer in quotation marks or a short
introduction without being penalised. If a reference sentence already occurs in its own
input sentence, as when the correction only removes a trailing full stop, a model that
merely copies its input would contain the reference, so such samples require an exact
match instead.

The secondary metric is exact match, being the percentage of model outputs that are
identical to the reference sentence. This additionally measures whether the model
follows the instruction to output the sentence and nothing else.

## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package), you
can benchmark your favorite model on the spelling standardisation task like so:

```bash
euroeval --model <model-id> --task spelling-standardisation
```
