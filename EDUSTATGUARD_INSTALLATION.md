# EduStatGuard v0.1.0 — ready-to-apply package

Target repository: `DrNour/postediting-ai-assessment-app`

Target base commit: `806ef688f4179fdd045184fe566ef6ab8c9ec7a0`

Original analytics-page blob: `dcbca3631961acbc85f2fc123fbfec8876d66cbc`

## Included changes

- `modules/edustatguard.py`: independent `allow` / `warn` / `block` validation engine.
- `pages/6_Research_Analytics.py`: integration for paired tests, correlations, and categorical tests.
- `tests/test_edustatguard.py`: 16 unit tests covering identifiers, small samples, repeated participants, deterministic mappings, sparse tables, p-value display, and Arabic-direction detection.
- `docs/edustatguard_policy.md`: versioned rule catalogue for the software paper.
- `modules/rtl_layout.py` and `app.py`: Arabic-aware RTL, right alignment, and justified paragraph rendering without changing English interface direction.

## Verification

From the repository root, run:

```bash
python -m unittest discover -s tests -v
python -m compileall -q modules pages/6_Research_Analytics.py
```

Expected result: 16 tests pass.

## Safe application

Apply these files on a new branch based on the target commit. Do not replace the analytics page if its upstream blob has changed; merge the EduStatGuard import, renderer, and three guarded analysis sections into the newer page instead.
