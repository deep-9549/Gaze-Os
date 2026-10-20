# Test fixtures

`replay_baseline.json` holds the replay metrics of the last accepted pipeline on
the synthetic replay session built in `tests/test_profiles_recording.py`. CI fails
when mean error or jitter gets more than 5% worse (PRD 23).

When the pipeline improves on purpose, regenerate the baseline by running the
snippet in that test's `replay_metrics` on the same session and commit the new
numbers with the change that caused them. Once real recorded sessions exist
(`tools/collect.py`), add them under `tests/fixtures/sessions/` (landmarks only,
with participant consent) and extend the test to replay each one.
