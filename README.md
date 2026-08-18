# self-redaction

Record matching can improve PII redaction when a service already knows which customer is speaking. The service can match selected fields from that customer's record, combine those spans with a general PII detector, and redact the text before it reaches a model, analytics system, or log.

This repository provides a deterministic synthetic check of that design. It compares the same record matcher with two general detectors:

- transparent regular expressions adapted from DataFog at a pinned commit;
- [Microsoft Presidio](https://microsoft.github.io/presidio/) 2.2.364 with the pinned spaCy `en_core_web_sm` 3.8.0 model.

Each general detector runs alone and in combination with record matching. A separate control deliberately supplies the wrong customer record.

The generator uses reserved example domains, fictional 555-01xx phone numbers, test payment cards, and SSN area numbers 900 through 999, which the Social Security Administration excludes from assignment ([SSA](https://www.ssa.gov/employer/randomizationfaqs.html)).

This is a sanity check, not a performance benchmark. The data are synthetic. The result establishes only that record matching and general detection have complementary error surfaces. Record matching is strongest where the organization already knows the answer. General detection is necessary where it does not.

## Prior work

The idea is not new. A 2008 clinical de-identification system matched patient names from the structured record linked to each note back into the note text, alongside dictionaries, regular expressions, and context rules ([Neamatullah et al., 2008](https://doi.org/10.1186/1472-6947-8-32)). A later system created patient-specific runtime dictionaries and combined them with learned and rule-based components ([Dehghan et al., 2015](https://doi.org/10.1016/j.jbi.2015.06.029)).

This repository applies the same basic design to customer-support text and makes the comparison reproducible. It does not claim a new de-identification method or measure how often private systems use one.

## Reproduce the results

Install Python 3.11, 3.12, or 3.13 and [uv 0.12.5](https://docs.astral.sh/uv/), then run:

```bash
uv sync --locked --all-extras --all-groups
make analysis
```

The command regenerates the profiles, chats, labels, predictions, redacted text, aggregate CSV files, and TeX tables in `build/analysis/`.

Run the complete local checks with:

```bash
make ci
```

`make ci` requires a TeX distribution with `latexmk`. The standard Docker check covers formatting, linting, tests, and analysis without requiring TeX on the host:

```bash
make ci-docker
```

Other useful targets are `make paper`, `make package`, and `make clean`.

## Run the command

The full comparison, including Presidio, is:

```bash
uv run --extra benchmark self-redaction --presidio --output-dir build/analysis
```

Omit `--presidio` for the lightweight regex and record-matching check.

The output directory contains:

- `synthetic_data.json`, with every generated profile, chat, and gold span;
- `chat_results.csv`, with predictions and redacted text for every method and chat;
- `summary.csv`, with masking and strict entity metrics;
- `stratified_recall.csv`, split by known or novel source and entity type;
- `record_resolution_summary.csv`, with correct-record and wrong-record results;
- generated TeX inputs used by the paper.

## Interpretation

Record matching cannot find a new phone number, a third party's name, or any fact absent from the resolved record. General detection remains necessary. Record matching can still cover account IDs, order IDs, addresses, and other values that a general detector must infer from text.

Correct customer resolution is a security precondition. The matcher should retrieve only fields authorized for the current interaction, run inside the trusted redaction boundary, retain match provenance, and discard its temporary lookup values afterward. The customer record should not be copied into a model prompt.

Masking direct identifiers is not anonymization. This project does not measure re-identification risk, fairness, production latency, legal compliance, multilingual text, or speech-recognition output.

## License and provenance

The project is released under the MIT License. `THIRD_PARTY_NOTICES.md` records the pinned DataFog source and its MIT notice. Presidio is MIT licensed, spaCy is MIT licensed, and the English spaCy model is MIT licensed.
