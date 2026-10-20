# ADR 0004: Package names differ slightly from the PRD tree

**Status:** accepted · **PRD:** 22

- `app/` is the importable package `gazeos/`, so `pip install` does not put a
  generic top-level `app` module on users' machines.
- `app/typing/` is `gazeos/text_entry/`, so it never shadows Python's standard
  `typing` module.
- GazeOS's own buttons (keyboard keys, action bar, menus) are registered as
  gaze targets and invoked directly instead of receiving injected OS clicks, so
  they can never steal focus from the app being typed into, on any platform.
