# Shared-history CI receipt (`ce2e6579`)

Publishable receipt from workflow **Shared-history receipts**, push on `main`.

This is a **mechanism test (hash embedder), not a semantic-quality benchmark**.
Any number quoted outside this folder must carry the label «mechanism test
(hash embedder), not a semantic-quality benchmark».

## Provenance

- Run: https://github.com/gianlucamazza/emotional-memory/actions/runs/40084829068
- Head SHA: `ce2e6579853ef272b03ba5577624a42fdcf1ada6`
- Artifact: `shared-history-receipts` (id `12556796823`)

CI artifacts expire. This directory keeps the durable subset of that run.

The manifest's `dataset_sha256` hashes the parsed dataset, not the file bytes,
so it differs from the file's sha256 (`f748e37f…`).

## Files from `SHA256SUMS`

Committed, byte-identical to the artifact:

- `arm1_replay_writeback.md`
- `arm1_replay_writeback.protocol.json`
- `arm1_replay_writeback.stdout.log`
- `arm2_replay_readonly.md`
- `arm2_replay_readonly.protocol.json`
- `arm2_replay_readonly.stdout.log`
- `arm3_timeline_writeback.md`
- `arm3_timeline_writeback.protocol.json`
- `arm3_timeline_writeback.stdout.log`
- `arm4_timeline_readonly.md`
- `arm4_timeline_readonly.protocol.json`
- `arm4_timeline_readonly.stdout.log`

Also committed from the same run, but not listed in `SHA256SUMS`:

- `SHA256SUMS`
- `receipt.json`
- `timings.txt`

Omitted for size:

- `arm1_replay_writeback.json`
- `arm2_replay_readonly.json`
- `arm3_timeline_writeback.json`
- `arm4_timeline_readonly.json`

Omitted empty stderr logs:

- `arm1_replay_writeback.stderr.log`
- `arm2_replay_readonly.stderr.log`
- `arm3_timeline_writeback.stderr.log`
- `arm4_timeline_readonly.stderr.log`

## Verify

While the artifact is still available, download `shared-history-receipts` from
the run above and compare committed files to that original artifact. Then, in
this directory:

```bash
sha256sum -c --ignore-missing SHA256SUMS
```

`--ignore-missing` is required here: the large `arm*.json` reports and the
empty stderr logs are listed in `SHA256SUMS` but are excluded from this
directory. Hashes verify only against that original artifact. A later
workflow run is not expected to reproduce them, because each `protocol.json`
records the Python version and platform.
