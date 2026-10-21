# Evidence cleanup independent review

Reviewer: Codex `gpt-6-astra`, reasoning effort `xhigh`, 2026-10-04.
Scope: [plan](../plans/evidence-cleanup.md), current Python/Swift result readers,
writers, archive transport and safety tests. No experiments run.

Initial findings, all fixed:

1. Preserve NaN/Infinity failed diagnostic records in the archive; sanitize only
   the human-readable projection.
2. Retain complete curated aggregate rows in eligibility and sweep summaries.
3. Route the manifest-crosscheck result writer through archive publication.
4. Export immutable bytes from one captured index snapshot during concurrent publication.

Final independent reviewer result: **clean; no remaining actionable findings**.

The reviewer independently verified 15/15 Python evidence safety tests, all
673 archived records, passing C4/multi/ANE eligibility on Linux and
`git diff --check`. Swift readers and safety tests were reviewed without
additional findings. The Mac Swift package independently passed 11/11 tests.
