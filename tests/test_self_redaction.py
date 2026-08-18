from __future__ import annotations

import csv
import json

import pytest

import self_redaction as experiment


def summary_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    return {(str(row["suite"]), str(row["method"])): row for row in rows}


def test_generator_contract() -> None:
    profiles, chats = experiment.generate_data(16)

    assert len(profiles) == 16
    assert len(chats) == 128
    assert sum(len(chat.gold) for chat in chats) == 400
    assert len({profile.full_name for profile in profiles}) == len(profiles)
    assert len({experiment.make_profile(i).full_name for i in range(64)}) == 64
    assert {chat.suite for chat in chats} == {"canonical", "stress"}
    assert len({chat.chat_id for chat in chats}) == len(chats)
    experiment.validate_generated_data(profiles, chats)

    for chat in chats:
        for span in chat.gold:
            assert span.source in {"known", "novel"}
            assert chat.text[span.start : span.end]


def test_headline_results_are_gated() -> None:
    profiles, chats = experiment.generate_data(16)
    profiles_by_id = {profile.customer_id: profile for profile in profiles}
    summary, _ = experiment.evaluate_methods(
        chats, experiment.build_standard_methods(profiles_by_id)
    )
    rows = summary_lookup(summary)

    assert rows[("canonical", "regex")]["mention_recall"] == pytest.approx(7 / 12)
    assert rows[("canonical", "record")]["known_mention_recall"] == 1
    assert rows[("canonical", "record")]["novel_mention_recall"] == 0
    assert rows[("canonical", "regex_record")]["mention_recall"] == pytest.approx(11 / 12)

    assert rows[("stress", "record")]["known_mention_recall"] == pytest.approx(3 / 7)
    assert rows[("stress", "regex_record")]["novel_mention_recall"] == pytest.approx(4 / 6)
    assert rows[("stress", "regex_record")]["mention_recall"] == pytest.approx(7 / 13)
    assert rows[("all", "regex_record")]["mention_recall"] == pytest.approx(18 / 25)


def test_presidio_and_record_matching_have_complementary_errors() -> None:
    profiles, chats = experiment.generate_data(4)
    profiles_by_id = {profile.customer_id: profile for profile in profiles}
    methods = experiment.build_standard_methods(
        profiles_by_id,
        {
            "regex": experiment.regex_detect,
            "presidio": experiment.build_presidio_detector(),
        },
    )
    summary, _ = experiment.evaluate_methods(chats, methods)
    rows = summary_lookup(summary)

    assert rows[("all", "presidio")]["mention_recall"] == pytest.approx(59 / 100)
    assert rows[("all", "presidio_record")]["mention_recall"] == pytest.approx(79 / 100)
    assert rows[("all", "presidio_record")]["known_mention_recall"] == pytest.approx(45 / 56)
    assert rows[("all", "presidio_record")]["novel_mention_recall"] == pytest.approx(34 / 44)


def test_wrong_record_control_removes_record_matching_gain() -> None:
    profiles, chats = experiment.generate_data(16)
    canonical = [chat for chat in chats if chat.suite == "canonical"]
    summary, _ = experiment.evaluate_methods(
        canonical, experiment.build_resolution_methods(profiles)
    )
    rows = summary_lookup(summary)

    assert rows[("canonical", "record_correct")]["known_mention_recall"] == 1
    assert rows[("canonical", "record_wrong")]["known_mention_recall"] == 0
    assert rows[("canonical", "regex_record_wrong")]["mention_recall"] == pytest.approx(7 / 12)


def test_scoring_separates_masking_from_strict_entity_detection() -> None:
    builder = experiment.TextBuilder()
    builder.add("Send it to ")
    builder.add("10 Cedar Street, Redmond, WA 98052", label="ADDRESS", source="novel")
    chat = builder.build("one", "CUST-1", "test", "partial address")
    zip_start = chat.text.index("98052")
    prediction = [experiment.Span(zip_start, zip_start + 5, "ZIP")]

    score = experiment.score_chat(chat, "regex", prediction)

    assert score["overlap_chars"] == 5
    assert score["mentions_fully_redacted"] == 0
    assert score["strict_true_positive_entities"] == 0
    assert score["wrong_label_spans"] == 1
    assert score["false_positive_spans"] == 0


def test_redaction_merges_overlapping_spans() -> None:
    text = "Email avery@example.com now"
    start = text.index("avery")
    spans = [
        experiment.Span(start, start + 5, "NAME"),
        experiment.Span(start, start + len("avery@example.com"), "EMAIL"),
    ]

    assert experiment.redact_text(text, spans) == "Email [REDACTED] now"


def test_conservative_self_detector_avoids_low_entropy_values() -> None:
    profile = experiment.make_profile(0)
    first, last = profile.full_name.split(" ", 1)
    text = f"Ask {first}; surname {last}; ZIP 98052; last four 0100."

    assert experiment.self_detect(profile, text) == []


def test_cli_writes_complete_reproducibility_artifacts(tmp_path) -> None:
    experiment.main(["--profiles", "4", "--output-dir", str(tmp_path)])

    expected = {
        "chat_results.csv",
        "metrics_table.tex",
        "record_resolution_chat_results.csv",
        "record_resolution_summary.csv",
        "resolution_table.tex",
        "results.tex",
        "strata_table.tex",
        "stratified_recall.csv",
        "summary.csv",
        "synthetic_data.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected

    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == 9

    data = json.loads((tmp_path / "synthetic_data.json").read_text(encoding="utf-8"))
    assert len(data["profiles"]) == 4
    assert len(data["chats"]) == 32
    assert "[REDACTED]" in (tmp_path / "chat_results.csv").read_text(encoding="utf-8")


def test_profile_count_must_support_wrong_record_control() -> None:
    with pytest.raises(SystemExit):
        experiment.parse_args(["--profiles", "1"])
