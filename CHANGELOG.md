# Changelog

## 0.1.1 - 2026-08-18

- Package the implementation under `src/self_redaction`.
- Add bounded Levenshtein matching for names and partial street lines.
- Add deterministic mistyped and incomplete record fields to the stress suite.
- Replace the tracked release-note file with GitHub-generated release notes.
- Move the DataFog license notice into the conventional `LICENSES` directory.

## 0.1.0 - 2026-08-18

- Add a deterministic synthetic corpus with canonical and stress suites.
- Compare record matching with pinned DataFog regexes and Microsoft Presidio.
- Report general detectors alone and combined with record matching.
- Add a wrong-record control, span-level outputs, and generated result tables.
- Pin the Python environment, spaCy model, and uv toolchain.
- Add local, Docker, and GitHub Actions checks.
