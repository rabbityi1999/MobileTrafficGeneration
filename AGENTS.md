# Working in this repository

This is the public code repository for Work 4 cross-city zero-label mobile traffic prediction research.

- Preserve scientific negative results and distinguish source spatial development from independent cross-city validation.
- Target traffic labels must not enter training, normalization fitting, adaptation, retrieval or model selection for K=0 claims.
- Run `python tools/run_checks.py` for changes to the exported baseline code; add relevant checks for new implementations.
- Do not add credentials, private raw data, model weights, private manuscripts or unrelated work.
- The user has authorized ordinary commits and pushes to `origin/main` after meaningful validated code changes. Stage only intended files, verify the remote commit, and report failures. Never force-push or reset another person's work.
- The initial `work4/SOURCE_MANIFEST.json` records source snapshot hashes. When an exported file changes, update its source provenance and manifest together instead of claiming an old hash matches.
