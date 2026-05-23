# Asymmetric Development Map

This directory is the entry point for the tokamak and asymmetric-development work carried in this fork.

## Read This First

- `current-status.md`
  Concise summary of what changed relative to upstream and which files matter for active work.
- `archive/`
  Chronological investigation notes, comparison writeups, and intermediate plans kept for reference.

## Active Implementation Areas

These are the main source locations that carry the actual feature work:

- `src/vmecpp/__init__.py`
  Python interface adjustments used by the benchmark and debug workflows.
- `src/vmecpp/cpp/vmecpp/common/vmec_indata/`
  Input parsing and asymmetric array sizing.
- `src/vmecpp/cpp/vmecpp/vmec/boundaries/`
  Tokamak and asymmetric boundary handling, including axis-domain behavior.
- `src/vmecpp/cpp/vmecpp/vmec/fourier_asymmetric/`
  New asymmetric transform code plus a large set of supporting tests.
- `src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/`
  Integration of asymmetric geometry and force processing into the solver.

## What Was Moved Out Of The Root

- Historical markdown notes went to `archive/`.
- One-off scripts, scratch inputs, and run summaries went to `tools/investigation/`.
- Generated logs and comparison outputs are no longer part of the tracked repo surface.

## Current Intent

Preserve the investigation trail, but make the active development surface small enough that a human can see where to continue work without reading every debugging artifact from the original analysis phase.
