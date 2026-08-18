# self-redaction 0.1.0

This release tests a small idea: when a conversation is already tied to a customer, use selected fields from that customer record as one input to redaction.

The comparison includes pinned DataFog regular expressions and Microsoft Presidio with spaCy. Each general detector is evaluated alone and with the same record matcher. The repository generates every profile, chat, label, prediction, redacted output, table, and paper result from one deterministic command.

This is a sanity check, not a performance benchmark. The data are synthetic. The result establishes only that record matching and general detection have complementary error surfaces. Record matching is strongest where the organization already knows the answer. General detection is necessary where it does not.

Correct customer resolution is a precondition. This release deliberately substitutes the wrong customer record and reports the resulting loss in coverage.
