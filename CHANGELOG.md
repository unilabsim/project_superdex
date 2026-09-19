# Changelog – Project SuperDex

All notable changes to this repository will be documented here.

## [Unreleased]

- Standardized precision names on `fp32` and `fp64`. The public
  `PRECISION_NAME` value now reports the canonical `fp32` or `fp64` name.
- Added `SceneBatchExecutorV3` ABI 3 with selective per-actor boundary-condition
  writes while retaining the ABI 2 executor. Invalid payloads are rejected before
  mutation, selected actors replace their complete clearable boundary set, and a
  native write failure closes the executor.

## [2026-08-24]

- Initial release.
