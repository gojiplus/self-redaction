#!/usr/bin/env python3
"""Reproducible synthetic check of record-conditioned PII redaction.

The experiment compares general PII detectors, customer-record matching, and
their unions on two deterministic corpora:

* ``canonical`` uses common formats covered by the detectors.
* ``stress`` uses plausible but unsupported formats, third-party PII, and
  non-PII numeric distractors.

The program also reruns the canonical corpus with an intentionally wrong
customer record. That control measures the dependence of record matching on
upstream entity resolution.

This is a sanity check, not a performance benchmark or production estimate.
All names, identifiers, contact details, and addresses are synthetic.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int
    label: str
    source: str = "prediction"

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Invalid span [{self.start}, {self.end}).")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Profile:
    customer_id: str
    full_name: str
    email: str
    phone: str
    address: str
    dob: str
    order_id: str
    account_id: str


@dataclass(frozen=True)
class ChatRecord:
    chat_id: str
    customer_id: str
    suite: str
    scenario: str
    text: str
    gold: tuple[Span, ...]


FIRST_NAMES = [
    "Avery",
    "Maya",
    "Noah",
    "Priya",
    "Ethan",
    "Leila",
    "Owen",
    "Nina",
    "Lucas",
    "Sofia",
    "Ravi",
    "Elena",
    "Miles",
    "Anika",
    "Jonah",
    "Mira",
]
LAST_NAMES = [
    "Patel",
    "Morgan",
    "Kim",
    "Garcia",
    "Shah",
    "Bennett",
    "Nguyen",
    "Davis",
    "Mehta",
    "Wilson",
    "Chen",
    "Rivera",
    "Kapoor",
    "Foster",
    "Singh",
    "Brooks",
]
PARTNER_FIRST_NAMES = [
    "Daria",
    "Hugo",
    "Inez",
    "Kofi",
    "Luz",
    "Mateo",
    "Nadia",
    "Tariq",
]
PARTNER_LAST_NAMES = [
    "Arden",
    "Bose",
    "Costa",
    "Diaz",
    "Ellis",
    "Faruq",
    "Ghosh",
    "Ito",
]
STREETS = [
    "Cedar",
    "Willow",
    "Maple",
    "Juniper",
    "Birch",
    "Aspen",
    "Pine",
    "Lake",
    "Hill",
    "Cherry",
    "Walnut",
    "Sunset",
    "Meadow",
    "River",
    "Spruce",
    "Oak",
]
CITIES = [
    ("Redmond", "WA", "98052"),
    ("Austin", "TX", "78704"),
    ("Portland", "OR", "97205"),
    ("Denver", "CO", "80206"),
]
SUFFIXES = [
    ("Street", "St"),
    ("Avenue", "Ave"),
    ("Road", "Rd"),
    ("Boulevard", "Blvd"),
]
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
MONTH_ABBREVIATIONS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def make_profile(i: int) -> Profile:
    first = FIRST_NAMES[i % len(FIRST_NAMES)]
    last_index = (i * 5 + i // len(FIRST_NAMES) + 3) % len(LAST_NAMES)
    last = LAST_NAMES[last_index]
    email = f"{first.lower()}.{last.lower()}.{i:03d}@example.com"
    # 555-0100 through 555-0199 are reserved for fictional use in the NANP.
    phone = f"{[206, 425, 512, 303][i % 4]}-555-01{i % 100:02d}"
    number = 100 + (i * 37) % 8800
    street = STREETS[(i * 7) % len(STREETS)]
    suffix_long, _ = SUFFIXES[i % len(SUFFIXES)]
    city, state, zip_code = CITIES[i % len(CITIES)]
    year = 1960 + (i % 40)
    month = 1 + ((i * 3) % 12)
    day = 1 + ((i * 7) % 27)
    return Profile(
        customer_id=f"CUST-{100000 + i}",
        full_name=f"{first} {last}",
        email=email,
        phone=phone,
        address=f"{number} {street} {suffix_long}, {city}, {state} {zip_code}",
        dob=date(year, month, day).isoformat(),
        order_id=f"ORD-{410000 + i * 17:06d}",
        account_id=f"ACCT-{73000000 + i * 29:08d}",
    )


class TextBuilder:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.gold: list[Span] = []
        self.length = 0

    def add(
        self,
        value: str,
        *,
        label: str | None = None,
        source: str | None = None,
    ) -> None:
        start = self.length
        self.parts.append(value)
        self.length += len(value)
        if label is not None:
            if source not in {"known", "novel"}:
                raise ValueError("Gold spans require source='known' or source='novel'.")
            self.gold.append(Span(start, self.length, label, source))

    def build(
        self,
        chat_id: str,
        customer_id: str,
        suite: str,
        scenario: str,
    ) -> ChatRecord:
        return ChatRecord(
            chat_id=chat_id,
            customer_id=customer_id,
            suite=suite,
            scenario=scenario,
            text="".join(self.parts),
            gold=tuple(self.gold),
        )


def phone_variant(phone: str, i: int) -> str:
    digits = re.sub(r"\D", "", phone)
    area, exchange, line = digits[:3], digits[3:6], digits[6:]
    variants = [
        f"({area}) {exchange}-{line}",
        f"{area}.{exchange}.{line}",
        f"+1 {area} {exchange} {line}",
        f"{area}{exchange}{line}",
    ]
    return variants[i % len(variants)]


def dob_variant(iso_date: str, i: int) -> str:
    year, month, day = map(int, iso_date.split("-"))
    variants = [
        f"{month:02d}/{day:02d}/{year}",
        f"{month}/{day}/{year}",
        f"{MONTHS[month - 1]} {day}, {year}",
        f"{year:04d}-{month:02d}-{day:02d}",
    ]
    return variants[i % len(variants)]


def address_variant(address: str, i: int) -> str:
    result = address
    for long_form, short_form in SUFFIXES:
        if long_form in result:
            result = result.replace(long_form, short_form if i % 2 == 0 else long_form)
            break
    if i % 3 == 0:
        result = result.replace(",", "")
    return result


def stress_address_variant(address: str) -> str:
    match = re.fullmatch(
        r"(?P<number>\d+) (?P<street>[A-Za-z]+) "
        r"(?P<suffix>Street|Avenue|Road|Boulevard), "
        r"(?P<city>[A-Za-z]+), (?P<state>[A-Z]{2}) (?P<zip>\d{5})",
        address,
    )
    if not match:
        raise ValueError(f"Unexpected synthetic address: {address}")
    values = match.groupdict()
    suffix = dict(SUFFIXES)[values["suffix"]]
    return (
        f"{values['number']} {values['street']} {suffix}., "
        f"{values['city']} {values['state']} {values['zip']}"
    )


def generate_canonical_chats(profiles: Sequence[Profile]) -> list[ChatRecord]:
    chats: list[ChatRecord] = []
    for i, profile in enumerate(profiles):
        b = TextBuilder()
        b.add("Hi, this is ")
        b.add(profile.full_name, label="NAME", source="known")
        b.add(". I am asking about order ")
        order_digits = profile.order_id.split("-", 1)[1]
        b.add(f"ORD {order_digits}", label="ORDER_ID", source="known")
        b.add(". Please call me at ")
        b.add(phone_variant(profile.phone, i), label="PHONE", source="known")
        b.add(". Case CASE-942731 was opened on 08/17/2026.")
        chats.append(
            b.build(
                f"canonical-{i:03d}-1",
                profile.customer_id,
                "canonical",
                "known identifiers",
            )
        )

        b = TextBuilder()
        b.add("My email is ")
        b.add(profile.email, label="EMAIL", source="known")
        b.add(". The shipping address is ")
        b.add(address_variant(profile.address, i), label="ADDRESS", source="known")
        b.add(". My date of birth is ")
        b.add(dob_variant(profile.dob, i), label="DOB", source="known")
        b.add(". The item SKU is 98105 and the price is $123.45.")
        chats.append(
            b.build(
                f"canonical-{i:03d}-2",
                profile.customer_id,
                "canonical",
                "known profile fields",
            )
        )

        novel_email = f"alternate.{i:03d}@example.net"
        novel_phone = f"(646) 555-01{i % 100:02d}"
        novel_ssn = f"{900 + (i % 100):03d}-{20 + (i % 70):02d}-{1000 + i:04d}"
        b = TextBuilder()
        b.add("Account ")
        b.add(profile.account_id, label="ACCOUNT_ID", source="known")
        b.add(". Use my new email ")
        b.add(novel_email, label="EMAIL", source="novel")
        b.add(" and my partner's phone ")
        b.add(novel_phone, label="PHONE", source="novel")
        b.add(". For verification, the SSN is ")
        b.add(novel_ssn, label="SSN", source="novel")
        b.add(".")
        chats.append(
            b.build(
                f"canonical-{i:03d}-3",
                profile.customer_id,
                "canonical",
                "new structured identifiers",
            )
        )

        _, _, novel_zip = CITIES[(i + 1) % len(CITIES)]
        b = TextBuilder()
        b.add("Charge test card ")
        b.add("4111 1111 1111 1111", label="CREDIT_CARD", source="novel")
        b.add(" and send the replacement to ")
        b.add(
            f"{9000 + i} Harbor Lane, Seattle, WA {novel_zip}",
            label="ADDRESS",
            source="novel",
        )
        b.add(". Tracking code 4255550199 is not a phone number.")
        chats.append(
            b.build(
                f"canonical-{i:03d}-4",
                profile.customer_id,
                "canonical",
                "new payment and address",
            )
        )
    return chats


def generate_stress_chats(profiles: Sequence[Profile]) -> list[ChatRecord]:
    """Create a declared adversarial suite after inspection of the baseline."""
    chats: list[ChatRecord] = []
    for i, profile in enumerate(profiles):
        first, last = profile.full_name.split(" ", 1)
        phone_digits = re.sub(r"\D", "", profile.phone)

        b = TextBuilder()
        b.add("The account is under ")
        b.add(f"{last}, {first}", label="NAME", source="known")
        b.add(". It concerns order #")
        b.add(profile.order_id.split("-", 1)[1], label="ORDER_ID", source="known")
        b.add(". My callback number is ")
        b.add(
            f"{phone_digits[:3]} / {phone_digits[3:6]} / {phone_digits[6:]}",
            label="PHONE",
            source="known",
        )
        b.add(".")
        chats.append(
            b.build(
                f"stress-{i:03d}-1",
                profile.customer_id,
                "stress",
                "unsupported known formats",
            )
        )

        year, month, day = map(int, profile.dob.split("-"))
        b = TextBuilder()
        b.add("EMAIL: ")
        b.add(profile.email.upper(), label="EMAIL", source="known")
        b.add(". Deliver to ")
        b.add(stress_address_variant(profile.address), label="ADDRESS", source="known")
        b.add(". DOB: ")
        b.add(
            f"{day} {MONTH_ABBREVIATIONS[month - 1]} {year}",
            label="DOB",
            source="known",
        )
        b.add(".")
        chats.append(
            b.build(
                f"stress-{i:03d}-2",
                profile.customer_id,
                "stress",
                "mixed supported and unsupported formats",
            )
        )

        partner_name = (
            f"{PARTNER_FIRST_NAMES[i % len(PARTNER_FIRST_NAMES)]} "
            f"{PARTNER_LAST_NAMES[(i * 3) % len(PARTNER_LAST_NAMES)]}"
        )
        novel_email = f"handoff.{i:03d}@example.org"
        novel_phone = f"646-555-01{(i + 40) % 100:02d}"
        novel_ssn = f"{900 + (i % 100):03d}-{30 + (i % 60):02d}-{2000 + i:04d}"
        b = TextBuilder()
        b.add("Please coordinate with ")
        b.add(partner_name, label="NAME", source="novel")
        b.add(" at ")
        b.add(novel_email, label="EMAIL", source="novel")
        b.add(" or ")
        b.add(novel_phone, label="PHONE", source="novel")
        b.add(". Their SSN is ")
        b.add(novel_ssn, label="SSN", source="novel")
        b.add(" and their address is ")
        b.add(
            f"{7000 + i} Harbor Lane, Seattle, WA 98109",
            label="ADDRESS",
            source="novel",
        )
        b.add(".")
        chats.append(
            b.build(
                f"stress-{i:03d}-3",
                profile.customer_id,
                "stress",
                "third-party identifiers",
            )
        )

        b = TextBuilder()
        b.add("Account ")
        b.add(profile.account_id, label="ACCOUNT_ID", source="known")
        b.add(" may use test card ")
        b.add("4111-1111-1111-1111", label="CREDIT_CARD", source="novel")
        b.add(". Quote 98105, case CASE-942731, date 08/17/2026, price $123.45.")
        chats.append(
            b.build(
                f"stress-{i:03d}-4",
                profile.customer_id,
                "stress",
                "known account plus distractors",
            )
        )
    return chats


def generate_data(n_profiles: int = 64) -> tuple[list[Profile], list[ChatRecord]]:
    profiles = [make_profile(i) for i in range(n_profiles)]
    chats = generate_canonical_chats(profiles) + generate_stress_chats(profiles)
    return profiles, chats


# Adapted from DataFog RegexAnnotator, MIT License, commit
# 1dc8cc57ca82bd45ad2e60ac9529fd922937f25a. See THIRD_PARTY_NOTICES.md.
GENERIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(
        r"""
        (?<![A-Za-z0-9._%+\-@])
        (?![A-Za-z_]{2,20}=)
        [A-Za-z0-9!#$%&*+\-/=^_`{|}~]
        [A-Za-z0-9!#$%&'*+\-/=?^_`{|}~.]*
        @
        (?:\.?[A-Za-z0-9-]+\.)+
        [A-Za-z]{2,}
        (?=$|[^A-Za-z])
        """,
        re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    ),
    "PHONE": re.compile(
        r"""
        (?<![A-Za-z0-9])
        (?:
            (?:(?:\+?1)[\-.\s]?)?
            (?:\(\d{3}\)|\d{3})
            [\-.\s]?
            \d{3}
            [\-.\s]?
            \d{4}
            |
            \+\d{1,3}
            [\s\-.]?
            \d{1,4}
            (?:[\s\-.]?\d{2,4}){2,3}
        )
        (?![-A-Za-z0-9])
        """,
        re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    ),
    "SSN": re.compile(
        r"""
        (?<!\d)
        (?:
            (?!000|666)\d{3}-(?!00)\d{2}-(?!0000)\d{4}
            |
            (?!000|666)\d{3}(?!00)\d{2}(?!0000)\d{4}
        )
        (?!\d)
        """,
        re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    ),
    "CREDIT_CARD": re.compile(
        r"""
        \b
        (?:
            4\d{12}(?:\d{3})?
            |
            5[1-5]\d{14}
            |
            3[47]\d{13}
            |
            (?:(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})
            |
            (?:3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5})
        )
        \b
        """,
        re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    ),
    "IP_ADDRESS": re.compile(
        r"""
        \b
        (?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.)
        (?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.)
        (?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.)
        (?:25[0-5]|2[0-4]\d|1?\d?\d)
        \b
        """,
        re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    ),
    "DOB": re.compile(
        r"""
        \b
        (?:
            (?:0?[1-9]|1[0-2])
            [/-]
            (?:0?[1-9]|[12][0-9]|3[01])
            [/-]
            (?:\d{2}|\d{4})
            |
            (?:\d{4})-(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12][0-9]|3[01])
            |
            (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|
               Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
            \s+(?:0?[1-9]|[12][0-9]|3[01]),\s+(?:19|20)\d{2}
            |
            year\s+(?:19|20)\d{2}
        )
        \b
        """,
        re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    ),
    "ZIP": re.compile(r"\b\d{5}(?:-\d{4})?\b", re.IGNORECASE | re.MULTILINE),
}

PRESIDIO_ENTITY_LABELS = {
    "CREDIT_CARD": "CREDIT_CARD",
    "DATE_TIME": "DOB",
    "EMAIL_ADDRESS": "EMAIL",
    "IP_ADDRESS": "IP_ADDRESS",
    "LOCATION": "ADDRESS",
    "PERSON": "NAME",
    "PHONE_NUMBER": "PHONE",
    "US_SSN": "SSN",
}

METHOD_TITLES = {
    "regex": "Regex",
    "record": "Record",
    "regex_record": "Regex + record",
    "presidio": "Presidio",
    "presidio_record": "Presidio + record",
    "record_correct": "Record, correct customer",
    "record_wrong": "Record, wrong customer",
    "presidio_record_correct": "Presidio + record, correct customer",
    "presidio_record_wrong": "Presidio + record, wrong customer",
    "regex_record_correct": "Regex + record, correct customer",
    "regex_record_wrong": "Regex + record, wrong customer",
}


def dedupe_spans(spans: Iterable[Span]) -> list[Span]:
    unique = {(span.start, span.end, span.label): span for span in spans}
    return sorted(unique.values(), key=lambda span: (span.start, span.end, span.label))


def regex_detect(text: str) -> list[Span]:
    return dedupe_spans(
        Span(match.start(), match.end(), label, "regex")
        for label, pattern in GENERIC_PATTERNS.items()
        for match in pattern.finditer(text)
    )


TextDetector = Callable[[str], Sequence[Span]]


def build_presidio_detector() -> TextDetector:
    try:
        import tldextract
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except ImportError as error:
        raise RuntimeError(
            "Presidio dependencies are missing. Run `uv sync --locked --all-extras --all-groups`."
        ) from error

    tldextract.extract = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=())

    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=configuration).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])

    def detect(text: str) -> list[Span]:
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=list(PRESIDIO_ENTITY_LABELS),
            score_threshold=0.0,
        )
        return dedupe_spans(
            Span(
                result.start,
                result.end,
                PRESIDIO_ENTITY_LABELS[result.entity_type],
                "presidio",
            )
            for result in results
        )

    return detect


def boundary_literal(value: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(value)}(?!\w)", re.IGNORECASE)


def address_pattern(address: str) -> re.Pattern[str]:
    match = re.fullmatch(
        r"(?P<number>\d+)\s+(?P<street>[A-Za-z]+)\s+"
        r"(?P<suffix>Street|Avenue|Road|Boulevard),\s*"
        r"(?P<city>[A-Za-z]+),\s*(?P<state>[A-Z]{2})\s+(?P<zip>\d{5})",
        address,
    )
    if not match:
        return boundary_literal(address)
    values = match.groupdict()
    suffix_long = values["suffix"]
    suffix_short = dict(SUFFIXES)[suffix_long]
    return re.compile(
        rf"(?<!\w){re.escape(values['number'])}\s+{re.escape(values['street'])}\s+"
        rf"(?:{re.escape(suffix_long)}|{re.escape(suffix_short)}\.?)"
        rf"\s*,?\s*{re.escape(values['city'])}\s*,?\s*{re.escape(values['state'])}"
        rf"\s+{re.escape(values['zip'])}(?!\w)",
        re.IGNORECASE,
    )


def dob_patterns(iso_date: str) -> list[re.Pattern[str]]:
    year, month, day = map(int, iso_date.split("-"))
    values = {
        f"{month:02d}/{day:02d}/{year}",
        f"{month}/{day}/{year}",
        f"{MONTHS[month - 1]} {day}, {year}",
        f"{year:04d}-{month:02d}-{day:02d}",
    }
    return [boundary_literal(value) for value in values]


def self_detect(profile: Profile, text: str) -> list[Span]:
    spans: list[Span] = []

    def add_matches(pattern: re.Pattern[str], label: str) -> None:
        spans.extend(Span(match.start(), match.end(), label) for match in pattern.finditer(text))

    add_matches(boundary_literal(profile.full_name), "NAME")
    add_matches(boundary_literal(profile.email), "EMAIL")
    add_matches(address_pattern(profile.address), "ADDRESS")
    for pattern in dob_patterns(profile.dob):
        add_matches(pattern, "DOB")

    order_digits = profile.order_id.split("-", 1)[1]
    add_matches(
        re.compile(
            rf"(?<!\w)(?:ORD\s*[- ]?\s*|order\s+){re.escape(order_digits)}(?!\w)",
            re.IGNORECASE,
        ),
        "ORDER_ID",
    )

    account_digits = profile.account_id.split("-", 1)[1]
    add_matches(
        re.compile(
            rf"(?<!\w)(?:ACCT\s*[- ]?\s*|account\s+){re.escape(account_digits)}(?!\w)",
            re.IGNORECASE,
        ),
        "ACCOUNT_ID",
    )

    profile_phone_digits = re.sub(r"\D", "", profile.phone)[-10:]
    for match in GENERIC_PATTERNS["PHONE"].finditer(text):
        candidate_digits = re.sub(r"\D", "", match.group())[-10:]
        if candidate_digits == profile_phone_digits:
            spans.append(Span(match.start(), match.end(), "PHONE"))

    return dedupe_spans(spans)


def union_spans(*groups: Sequence[Span]) -> list[Span]:
    return dedupe_spans(span for group in groups for span in group)


def masking_ranges(spans: Sequence[Span]) -> list[tuple[int, int]]:
    ranges: list[list[int]] = []
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if ranges and span.start <= ranges[-1][1]:
            ranges[-1][1] = max(ranges[-1][1], span.end)
        else:
            ranges.append([span.start, span.end])
    return [(start, end) for start, end in ranges]


def redact_text(text: str, spans: Sequence[Span]) -> str:
    redacted = text
    for start, end in reversed(masking_ranges(spans)):
        redacted = f"{redacted[:start]}[REDACTED]{redacted[end:]}"
    return redacted


def mask_for(length: int, spans: Sequence[Span]) -> list[bool]:
    mask = [False] * length
    for span in spans:
        if span.end > length:
            raise ValueError(f"Span {span} exceeds text length {length}.")
        for position in range(span.start, span.end):
            mask[position] = True
    return mask


def overlap(left: Span, right: Span) -> bool:
    return max(left.start, right.start) < min(left.end, right.end)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f_score(precision: float, recall: float, beta: float = 1.0) -> float:
    beta_squared = beta**2
    denominator = beta_squared * precision + recall
    return safe_divide((1 + beta_squared) * precision * recall, denominator)


def score_chat(chat: ChatRecord, method: str, predicted: Sequence[Span]) -> dict[str, object]:
    gold_mask = mask_for(len(chat.text), chat.gold)
    pred_mask = mask_for(len(chat.text), predicted)
    gold_chars = sum(gold_mask)
    pred_chars = sum(pred_mask)
    overlap_chars = sum(gold and pred for gold, pred in zip(gold_mask, pred_mask, strict=True))
    safe_chars = len(chat.text) - gold_chars
    mention_hits = [
        all(pred_mask[position] for position in range(span.start, span.end)) for span in chat.gold
    ]
    gold_entities = {(span.start, span.end, span.label) for span in chat.gold}
    pred_entities = {(span.start, span.end, span.label) for span in predicted}
    strict_true_positives = len(gold_entities & pred_entities)
    false_positive_spans = sum(
        not any(overlap(span, gold) for gold in chat.gold) for span in predicted
    )
    wrong_label_spans = sum(
        any(overlap(span, gold) for gold in chat.gold)
        and not any(overlap(span, gold) and span.label == gold.label for gold in chat.gold)
        for span in predicted
    )
    return {
        "chat_id": chat.chat_id,
        "customer_id": chat.customer_id,
        "suite": chat.suite,
        "scenario": chat.scenario,
        "method": method,
        "text": chat.text,
        "redacted_text": redact_text(chat.text, predicted),
        "gold_mentions": len(chat.gold),
        "mentions_fully_redacted": sum(mention_hits),
        "known_mentions": sum(span.source == "known" for span in chat.gold),
        "known_mentions_redacted": sum(
            span.source == "known" and hit
            for span, hit in zip(chat.gold, mention_hits, strict=True)
        ),
        "novel_mentions": sum(span.source == "novel" for span in chat.gold),
        "novel_mentions_redacted": sum(
            span.source == "novel" and hit
            for span, hit in zip(chat.gold, mention_hits, strict=True)
        ),
        "gold_chars": gold_chars,
        "predicted_chars": pred_chars,
        "overlap_chars": overlap_chars,
        "safe_chars": safe_chars,
        "strict_true_positive_entities": strict_true_positives,
        "predicted_entities": len(pred_entities),
        "gold_entities": len(gold_entities),
        "false_positive_spans": false_positive_spans,
        "wrong_label_spans": wrong_label_spans,
        "gold": json.dumps([asdict(span) for span in chat.gold], sort_keys=True),
        "predictions": json.dumps([asdict(span) for span in predicted], sort_keys=True),
    }


def aggregate_scores(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    totals: dict[str, float] = defaultdict(float)
    sum_fields = [
        "gold_mentions",
        "mentions_fully_redacted",
        "known_mentions",
        "known_mentions_redacted",
        "novel_mentions",
        "novel_mentions_redacted",
        "gold_chars",
        "predicted_chars",
        "overlap_chars",
        "safe_chars",
        "strict_true_positive_entities",
        "predicted_entities",
        "gold_entities",
        "false_positive_spans",
        "wrong_label_spans",
    ]
    for row in rows:
        for field in sum_fields:
            totals[field] += float(row[field])

    char_precision = safe_divide(totals["overlap_chars"], totals["predicted_chars"])
    char_recall = safe_divide(totals["overlap_chars"], totals["gold_chars"])
    strict_precision = safe_divide(
        totals["strict_true_positive_entities"], totals["predicted_entities"]
    )
    strict_recall = safe_divide(totals["strict_true_positive_entities"], totals["gold_entities"])
    return {
        "mention_recall": safe_divide(totals["mentions_fully_redacted"], totals["gold_mentions"]),
        "known_mention_recall": safe_divide(
            totals["known_mentions_redacted"], totals["known_mentions"]
        ),
        "novel_mention_recall": safe_divide(
            totals["novel_mentions_redacted"], totals["novel_mentions"]
        ),
        "character_precision": char_precision,
        "character_recall": char_recall,
        "character_f1": f_score(char_precision, char_recall),
        "character_f2": f_score(char_precision, char_recall, beta=2),
        "false_positive_character_rate": safe_divide(
            totals["predicted_chars"] - totals["overlap_chars"], totals["safe_chars"]
        ),
        "strict_entity_precision": strict_precision,
        "strict_entity_recall": strict_recall,
        "strict_entity_f1": f_score(strict_precision, strict_recall),
        "predicted_spans": int(totals["predicted_entities"]),
        "false_positive_spans": int(totals["false_positive_spans"]),
        "wrong_label_spans": int(totals["wrong_label_spans"]),
        "chats": len(rows),
        "gold_mentions": int(totals["gold_mentions"]),
        "gold_characters": int(totals["gold_chars"]),
        "safe_characters": int(totals["safe_chars"]),
    }


PredictionFunction = Callable[[ChatRecord], Sequence[Span]]


def evaluate_methods(
    chats: Sequence[ChatRecord],
    methods: dict[str, PredictionFunction],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    details = [
        score_chat(chat, method, detector(chat))
        for chat in chats
        for method, detector in methods.items()
    ]
    summary: list[dict[str, object]] = []
    suites = sorted({chat.suite for chat in chats})
    for suite in [*suites, "all"]:
        for method in methods:
            selected = [
                row
                for row in details
                if row["method"] == method and (suite == "all" or row["suite"] == suite)
            ]
            summary.append({"suite": suite, "method": method, **aggregate_scores(selected)})
    return summary, details


def build_standard_methods(
    profiles_by_id: dict[str, Profile],
    general_detectors: dict[str, TextDetector] | None = None,
) -> dict[str, PredictionFunction]:
    if general_detectors is None:
        general_detectors = {"regex": regex_detect}

    def record(chat: ChatRecord) -> Sequence[Span]:
        return self_detect(profiles_by_id[chat.customer_id], chat.text)

    methods: dict[str, PredictionFunction] = {}
    for name, detector in general_detectors.items():

        def general(chat: ChatRecord, detector: TextDetector = detector) -> Sequence[Span]:
            return detector(chat.text)

        methods[name] = general
        if name == "regex":
            methods["record"] = record

        def combined(
            chat: ChatRecord,
            detector: TextDetector = detector,
        ) -> Sequence[Span]:
            return union_spans(detector(chat.text), record(chat))

        methods[f"{name}_record"] = combined

    return methods


def build_resolution_methods(
    profiles: Sequence[Profile],
    general_detector: TextDetector = regex_detect,
    general_name: str = "regex",
) -> dict[str, PredictionFunction]:
    correct = {profile.customer_id: profile for profile in profiles}
    wrong = {
        profile.customer_id: profiles[(index + 1) % len(profiles)]
        for index, profile in enumerate(profiles)
    }

    def general(chat: ChatRecord) -> Sequence[Span]:
        return general_detector(chat.text)

    def self_correct(chat: ChatRecord) -> Sequence[Span]:
        return self_detect(correct[chat.customer_id], chat.text)

    def self_wrong(chat: ChatRecord) -> Sequence[Span]:
        return self_detect(wrong[chat.customer_id], chat.text)

    def combined_correct(chat: ChatRecord) -> Sequence[Span]:
        return union_spans(general(chat), self_correct(chat))

    def combined_wrong(chat: ChatRecord) -> Sequence[Span]:
        return union_spans(general(chat), self_wrong(chat))

    return {
        general_name: general,
        "record_correct": self_correct,
        "record_wrong": self_wrong,
        f"{general_name}_record_correct": combined_correct,
        f"{general_name}_record_wrong": combined_wrong,
    }


def stratified_recall(
    chats: Sequence[ChatRecord],
    details: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    chats_by_id = {chat.chat_id: chat for chat in chats}
    output: dict[tuple[str, str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in details:
        chat = chats_by_id[str(row["chat_id"])]
        predictions = [Span(**item) for item in json.loads(str(row["predictions"]))]
        pred_mask = mask_for(len(chat.text), predictions)
        for gold in chat.gold:
            hit = all(pred_mask[position] for position in range(gold.start, gold.end))
            for suite in (chat.suite, "all"):
                key = (suite, str(row["method"]), gold.source, gold.label)
                output[key][0] += 1
                output[key][1] += int(hit)
    return [
        {
            "suite": suite,
            "method": method,
            "source": source,
            "label": label,
            "mentions": counts[0],
            "mentions_fully_redacted": counts[1],
            "mention_recall": safe_divide(counts[1], counts[0]),
        }
        for (suite, method, source, label), counts in sorted(output.items())
    ]


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_pct(value: object, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def tex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    return "".join(replacements.get(character, character) for character in value)


def method_title(method: str) -> str:
    return METHOD_TITLES.get(method, method.replace("_", " ").title())


def tex_command_part(value: str) -> str:
    return "".join(part.title() for part in value.split("_"))


def write_tex_artifacts(
    output_dir: Path,
    summary: Sequence[dict[str, object]],
    resolution_summary: Sequence[dict[str, object]],
    strata: Sequence[dict[str, object]],
) -> None:
    lookup = {(str(row["suite"]), str(row["method"])): row for row in summary}
    methods = [str(row["method"]) for row in summary if row["suite"] == "canonical"]
    macros = ["% Generated by self_redaction.py. Do not edit."]
    for suite in ("canonical", "stress", "all"):
        for method in methods:
            row = lookup[(suite, method)]
            prefix = f"{suite.title()}{tex_command_part(method)}"
            for suffix, field in (
                ("MentionRecall", "mention_recall"),
                ("KnownRecall", "known_mention_recall"),
                ("NovelRecall", "novel_mention_recall"),
                ("CharacterFtwo", "character_f2"),
                ("StrictEntityFone", "strict_entity_f1"),
            ):
                formatted = format_pct(row[field]).replace("%", r"\%")
                macros.append(rf"\newcommand{{\{prefix}{suffix}}}{{{formatted}}}")
    macros.extend(
        [
            rf"\newcommand{{\ProfileCount}}{{{int(lookup[('all', methods[0])]['chats']) // 8}}}",
            rf"\newcommand{{\ChatCount}}{{{int(lookup[('all', methods[0])]['chats'])}}}",
            rf"\newcommand{{\MentionCount}}{{{int(lookup[('all', methods[0])]['gold_mentions'])}}}",
        ]
    )
    (output_dir / "results.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    table_lines = [
        "% Generated by self_redaction.py. Do not edit.",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Suite & Method & Mention R & Known R & Novel R & Character F$_2$ & "
        r"Strict F$_1$ & FP spans \\",
        r"\midrule",
    ]
    for suite in ("canonical", "stress", "all"):
        for method in methods:
            row = lookup[(suite, method)]
            table_lines.append(
                " & ".join(
                    [
                        tex_escape(suite.title()),
                        tex_escape(method_title(method)),
                        format_pct(row["mention_recall"]),
                        format_pct(row["known_mention_recall"]),
                        format_pct(row["novel_mention_recall"]),
                        format_pct(row["character_f2"]),
                        format_pct(row["strict_entity_f1"]),
                        str(row["false_positive_spans"]),
                    ]
                ).replace("%", r"\%")
                + r" \\"
            )
        if suite != "all":
            table_lines.append(r"\addlinespace")
    table_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "metrics_table.tex").write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    resolution_lookup = {str(row["method"]): row for row in resolution_summary}
    resolution_lines = [
        "% Generated by self_redaction.py. Do not edit.",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & All mentions & Known mentions & Character F$_2$ \\",
        r"\midrule",
    ]
    for method in resolution_lookup:
        row = resolution_lookup[method]
        resolution_lines.append(
            " & ".join(
                [
                    tex_escape(method_title(method)),
                    format_pct(row["mention_recall"]),
                    format_pct(row["known_mention_recall"]),
                    format_pct(row["character_f2"]),
                ]
            ).replace("%", r"\%")
            + r" \\"
        )
    resolution_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "resolution_table.tex").write_text(
        "\n".join(resolution_lines) + "\n", encoding="utf-8"
    )

    combined_method = "presidio_record" if "presidio_record" in methods else "regex_record"
    combined_strata = [
        row for row in strata if row["suite"] == "all" and row["method"] == combined_method
    ]
    strata_lines = [
        "% Generated by self_redaction.py. Do not edit.",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Source & Entity & Mentions & Fully redacted \\",
        r"\midrule",
    ]
    for row in combined_strata:
        strata_lines.append(
            " & ".join(
                [
                    tex_escape(str(row["source"]).title()),
                    tex_escape(str(row["label"]).replace("_", " ").title()),
                    str(row["mentions"]),
                    format_pct(row["mention_recall"]),
                ]
            ).replace("%", r"\%")
            + r" \\"
        )
    strata_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "strata_table.tex").write_text("\n".join(strata_lines) + "\n", encoding="utf-8")


def validate_generated_data(profiles: Sequence[Profile], chats: Sequence[ChatRecord]) -> None:
    profile_ids = [profile.customer_id for profile in profiles]
    chat_ids = [chat.chat_id for chat in chats]
    if len(profile_ids) != len(set(profile_ids)):
        raise ValueError("Customer IDs are not unique.")
    if len(chat_ids) != len(set(chat_ids)):
        raise ValueError("Chat IDs are not unique.")
    if any(chat.customer_id not in set(profile_ids) for chat in chats):
        raise ValueError("A chat references a missing customer profile.")
    for chat in chats:
        for span in chat.gold:
            if span.end > len(chat.text):
                raise ValueError(f"Gold span exceeds text in {chat.chat_id}.")
            if not chat.text[span.start : span.end]:
                raise ValueError(f"Empty gold span in {chat.chat_id}.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=int, default=64, help="Number of synthetic customers")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "build" / "analysis",
        help="Directory for generated CSV, JSON, and TeX outputs",
    )
    parser.add_argument(
        "--presidio",
        action="store_true",
        help="Include the pinned Presidio and spaCy general-detector comparison",
    )
    args = parser.parse_args(argv)
    if args.profiles < 2:
        parser.error("--profiles must be at least 2 for the wrong-record control")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    profiles, chats = generate_data(args.profiles)
    validate_generated_data(profiles, chats)
    profiles_by_id = {profile.customer_id: profile for profile in profiles}

    general_detectors: dict[str, TextDetector] = {"regex": regex_detect}
    if args.presidio:
        general_detectors["presidio"] = build_presidio_detector()
    summary, details = evaluate_methods(
        chats,
        build_standard_methods(profiles_by_id, general_detectors),
    )
    canonical = [chat for chat in chats if chat.suite == "canonical"]
    resolution_name = "presidio" if args.presidio else "regex"
    resolution_summary, resolution_details = evaluate_methods(
        canonical,
        build_resolution_methods(
            profiles,
            general_detectors[resolution_name],
            resolution_name,
        ),
    )
    resolution_summary = [row for row in resolution_summary if row["suite"] == "canonical"]
    strata = stratified_recall(chats, details)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "chat_results.csv", details)
    write_csv(args.output_dir / "stratified_recall.csv", strata)
    write_csv(args.output_dir / "record_resolution_summary.csv", resolution_summary)
    write_csv(args.output_dir / "record_resolution_chat_results.csv", resolution_details)
    (args.output_dir / "synthetic_data.json").write_text(
        json.dumps(
            {
                "profiles": [asdict(profile) for profile in profiles],
                "chats": [asdict(chat) for chat in chats],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_tex_artifacts(args.output_dir, summary, resolution_summary, strata)

    print(f"Synthetic customers: {len(profiles)}")
    print(f"Synthetic chats:     {len(chats)}")
    print(f"Gold PII mentions:   {sum(len(chat.gold) for chat in chats)}")
    print()
    print(
        f"{'suite':<10} {'method':<24} {'mention':>9} {'known':>9} "
        f"{'novel':>9} {'char F2':>9} {'strict F1':>10} {'FP spans':>9}"
    )
    for row in summary:
        print(
            f"{str(row['suite']):<10} {method_title(str(row['method'])):<24} "
            f"{format_pct(row['mention_recall']):>9} "
            f"{format_pct(row['known_mention_recall']):>9} "
            f"{format_pct(row['novel_mention_recall']):>9} "
            f"{format_pct(row['character_f2']):>9} "
            f"{format_pct(row['strict_entity_f1']):>10} "
            f"{int(row['false_positive_spans']):>9}"
        )
    print()
    print(f"Wrote analysis artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
