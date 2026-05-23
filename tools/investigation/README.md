# Investigation Archive

This directory contains one-off scripts, scratch inputs, and intermediate result files from the asymmetric and tokamak investigation work.

## Purpose

These files were useful during diagnosis. They preserve the reasoning trail. They are outside the supported build, test, and benchmark workflow.

## Contents

- `scripts/`
  Ad hoc Python, shell, and C++ probes used during debugging.
- `inputs/`
  Scratch and historical input files that are not part of the curated example or test-data sets.
- `results/`
  Short summaries and scratch outputs that were worth preserving as context.

## Ground Rules

- Expect some scripts here to reference old paths or older repo layouts.
- Do not treat this directory as the official test suite.
- If a script becomes useful again, either modernize it in place and document it, or promote it into a maintained location elsewhere in the repo.
