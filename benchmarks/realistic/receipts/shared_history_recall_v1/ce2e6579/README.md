# Shared-history CI receipt (`ce2e6579`)

Publishable receipt from workflow **Shared-history receipts**, push on `main`.

This is a **mechanism test (hash embedder), not a semantic-quality benchmark**.

## Provenance

- Run: https://github.com/gianlucamazza/emotional-memory/actions/runs/40084829068
- Head SHA: `ce2e6579853ef272b03ba5577624a42fdcf1ada6`
- Artifact: `shared-history-receipts` (id `12556796823`)

CI artifacts expire. This directory keeps the durable subset of that run.

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
the run above. After it expires, re-run the **Shared-history receipts**
workflow on this SHA. Then:

```bash
sha256sum -c SHA256SUMS
```

Committed files must match their `SHA256SUMS` lines. Omitted files appear only
in the downloaded artifact or a fresh workflow run.
