# Handoff: shrink committed results on `parakeet-ios`

## Completion status

Steps 1–4 completed on 2026-10-04; independent Astra xhigh review clean.
See [evidence-cleanup-report.md](evidence-cleanup-report.md) for the inventory,
archive verification, tests and byte totals. Step 5 was not authorized and was not performed.

## Context

The `parakeet-ios` branch of this **public** repository has about 2.13M lines
(31 MB) of generated results under `finetune/parakeet-ternary/ios/results/`,
against about 32k lines of code and docs. The results are machine-readable
gate evidence: per-clip, per-decode-loop JSON, pretty-printed.

| Path | Size | What it holds |
| --- | ---: | --- |
| `results/gates/v7/` | 17 MB | per-arm gate 4/5 records, about 13k lines each |
| `results/eligibility/pipelines/` | 3.4 MB | revision-10 pipeline eligibility records |
| `results/eligibility/pipelines/quarantine/` | 2.2 MB | invalid pre-review records |
| `results/builds/` | 3.0 MB | build manifests |
| `results/pipegates/` | 2.2 MB | 50 per-pipeline gate summaries, about 1,800 lines each |

`results/pipegates/` holds the 87k lines the user noticed. The quarantined
records predate a clean review and must never count as evidence. Nothing in
`results/` is secret, but the volume hides the code in diffs.

## Goal

1. Git keeps only compact, human-readable results.
2. Full records live outside Git, on `/mnt/hd`.
3. Every committed summary stays verifiable against those full records.
4. Nothing that reads the records breaks.

## Steps

1. **Inventory.** List every file under `ios/results/` with its size, which
   work package (WP) produced it, and which code reads it.
   - These files reference committed result paths: `mil/eligibility.py`,
     `mil/gates.py`, `mil/gates7.py`, `mil/build.py`, `mil/refcache.py`,
     `pipegate.py`, `mlxarm/record.py`, and the Swift files
     `bench/Sources/BenchCore/{Pipeline,Data}.swift` and
     `bench/Sources/parakeet-bench/ParakeetBenchCLI.swift`.
   - Re-grep to confirm nothing else does.
2. **Archive the full records.** Copy them to
   `/mnt/hd/wilderness-labs-stt/parakeet-ios/results-archive/<commit>/`
   (check that `/mnt/hd` is mounted) and write a `SHA256SUMS` file there.
   - Delete `quarantine/` outright. Keep its SHA-256 list in the archive
     only as a record that it existed.
3. **Keep in Git.** Each kept file should be small, ideally under 100 KB:
   - eligibility tables (`table.txt`, `summary.json`);
   - `wp3_*` / `wp6a_table.txt` / `wp5` / `wp7` sweep tables and sweep
     manifests;
   - per-arm `summary.json` files;
   - stress and probe summaries;
   - a new `results/INDEX.json`, mapping every archived record to its
     SHA-256, size and archive path.

   Committed summaries must cite the SHA-256 of the full record they
   summarize.
4. **Repoint the code.**
   - Readers resolve full records from a configurable root. On Linux that is
     `/mnt/hd/.../results-archive` or the live results directory; on the Mac
     it is the artifacts dir. The setting comes from an environment variable
     or the untracked `mil/local.json`, as in the existing `artifacts.py`
     pattern.
   - They verify each record against `INDEX.json` or the hash embedded in the
     summary.
   - The eligibility check (`mil/eligibility.py` and the Swift runner) must
     still refuse missing, stale or mismatched records. Writers keep writing
     full records outside Git and only the summary into Git.
5. **History** (needs the user's explicit OK; not yet given). Rewrite
   `parakeet-ios` so the bulk never enters history, for example with
   `git filter-repo --path-glob ...` on a fresh clone, then force-push. Do not
   touch `main` or any other branch.
   - The Mac checkout `/Users/ajbarry/workspace/github.com/wilderness-labs-stt`
     must then be reset to the new branch head.
   - Without the OK, do steps 1-4 as normal commits and stop.

## Constraints

- **Repo rules:**
  - Public repo: commit only by explicit path, never `git add -A`.
  - No models, audio or credentials.
  - End commit messages with the `Co-Authored-By` line used on this branch.
- **Review rule** (DESIGN.md "Stages"): the code changes in step 4 must pass a
  Codex `gpt-6-astra` xhigh review of code plus this plan, iterated until
  clean, **before** any gate, sweep or other experiment runs again.
  - Unit and safety tests are allowed: `tests/*selftest*`, Swift tests,
    `tests/macguard_tests.sh`.
- **Shared machines:**
  - Do not disturb `parakeet-main.service` (GPU training) on Linux.
  - On the Mac, use `ios/macguard` for any non-trivial job, keep at least
    30 GB free, and use no sudo.
- **Do not edit `DESIGN.md`.** Note required design text in the report
  instead.

## Done when

- `git diff --stat main..parakeet-ios -- finetune/parakeet-ternary/ios/results`
  is a few MB at most.
- Every committed summary resolves to a hash-verified archived record.
- The existing self-tests pass, plus a new one: an eligibility check on an
  archived record passes, and it refuses a missing or tampered record.
- `mil/eligibility.py check mp2 C4 multi ane` still passes on Linux and on
  the Mac.
- The report lists bytes before and after, the files kept, and whether
  history was rewritten.
