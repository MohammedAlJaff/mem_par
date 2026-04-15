# Canonical Python Style

This document is the source of truth for how canonical first-party Python is
written in this repo.

If this guide conflicts with older canonical code, this guide wins.

## Applies To

- Canonical first-party Python under `src/`
- New shared helpers that support canonical experiments
- First-party modules promoted from prototype status into canonical use

## Does Not Apply To

- Vendored code under `baselines/*/vendor`
- `third_party/`
- Archived material preserved in `archives/`
- Historical code that has not been touched yet

## Reference Exemplar

The cleaned GDFM prototype in `prototyping/gdfm_components.py` is the style and
notation exemplar for canonical Python writing.

Use it as the reference for:

- section-block layout
- math exposition
- naming and notation
- runtime validation tone
- compact helper structure

## Core Writing Principles

- Math-first, teaching-first
  Explain the object before the mechanism.
- Write for a future researcher
  The reader should be able to recover the scientific idea from the code.
- Prefer compact research prose
  Short declarative comments beat architecture-flavored explanation.
- Prefer local derivation over framework indirection
  Keep important algebra and conventions close to where they are used.
- Prefer clarity over abstraction cleverness
  Small direct helpers beat generic dispatch unless the abstraction is already
  stable and scientifically justified.

## Visual Structure

- Large section-divider comment blocks are allowed and preferred for major
  conceptual regions.
- Short local comment blocks before important code regions are preferred over
  dense inline comments.
- Docstrings should explain meaning, convention, and scientific role, not just
  restate the signature.
- Prefer direct statements such as `Compute ...`, `Convert ...`, or
  `Use ... so that ...`.
- Avoid software-architecture phrasing such as `this module owns`,
  `bridge between`, `canonical surface`, or `public API` unless the boundary
  itself is scientifically important.
- Important numerical decisions should be documented at the point of use.

## Naming And Notation

- Use `eps`, never `epsilon`.
- Use `z_t`, `v_t`, `a(t)`, `b(t)`, `a'(t)`, and `b'(t)` consistently in prose
  and code-adjacent explanation.
- Use `prediction_space` and `loss_space` in internal names and public
  interfaces when practical.
- Legacy names such as `pred_space` may remain at CLI or config boundaries when
  changing them would create unnecessary churn.
- Keep the GDFM time convention explicit everywhere it matters:
  `t = 0` means noise, `t = 1` means data.
- Prefer direct helper names such as `_bcast`, `_safe_div`,
  `z_t_interpolate`, `get_true_target`, and `convert`.

## Code Form

- No type hints in canonical first-party Python.
- No `from __future__ import annotations` in canonical first-party Python.
- Avoid `typing` imports in canonical code unless a third-party API makes one
  unavoidable.
- Prefer explicit runtime validation with readable `ValueError` messages over
  type-level signaling.
- Prefer direct `if/elif` logic and compact helper functions over generic
  registries or abstraction layers unless they are already stable.

## Numerics And Scientific Conventions

- Every clamp, denominator floor, endpoint convention, and conversion identity
  should be justified where it is implemented.
- Important research conventions must live in the code, not only in paper
  drafts, notebooks, or external notes.
- If a symbol is overloaded in the literature, the code should disambiguate it
  locally with comments or explicit naming.
- Do not explain workflow if the local mathematical or numerical reason is the
  real point. Explain the quantity, the convention, or the invariant first.

## Rollout Rule

- All new canonical Python follows this guide immediately.
- When touching an existing canonical file, normalize the touched region toward
  this guide:
  - remove type hints
  - align comments and docstrings with the GDFM explanatory style
  - align notation where the change is local and safe
- Do not do large unrelated rewrites just to chase style consistency.
