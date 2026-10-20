"""GazeOS entry point.

    gazeos                      run with the last profile (onboarding on first run)
    gazeos --profile Ramesh     run with a named profile
    gazeos --simulate           no camera: a synthetic face looks where the mouse is
    gazeos --dry-run            track and show the overlay, but inject no input
    gazeos --list-cameras       print the available camera indices
"""

from __future__ import annotations

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gazeos", description="Hands-free computer control with a webcam.")
    p.add_argument("--profile", help="profile name (default: last used)")
    p.add_argument("--home", help="data folder (default: per-user app data; or $GAZEOS_HOME)")
    p.add_argument("--camera", type=int, help="camera index (see --list-cameras)")
    p.add_argument("--simulate", action="store_true", help="drive the pipeline from the mouse, no camera")
    p.add_argument("--dry-run", action="store_true", help="never inject mouse or keyboard input")
    p.add_argument("--list-cameras", action="store_true", help="list cameras and exit")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.list_cameras:
        from gazeos.capture.camera import list_cameras

        cams = list_cameras()
        print("\n".join(f"camera {i}" for i in cams) if cams else "no cameras found")
        return 0
    from gazeos.control.mapping import make_dpi_aware

    make_dpi_aware()  # before any window or Qt object exists (PRD 12.5)
    from gazeos.ui.app import GazeOSApp

    return GazeOSApp(args).run()


if __name__ == "__main__":
    sys.exit(main())
