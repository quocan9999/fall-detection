"""Create a two-class pose dataset variant where Sleeping is treated as fall.

This entrypoint deliberately reuses the validated conversion pipeline in
convert_clean_pose_to_2class.py. Only the class mapping and publish locations
are changed, so the existing pose, split, dataleak, and integrity checks stay
identical to the baseline two-class conversion.
"""

from __future__ import annotations

from pathlib import Path

import convert_clean_pose_to_2class as converter


OUTPUT_NAME = "fall-detection-clean-fastdup-2class-sleeping-as-fall"
REPORT_NAME = "convert_2class_sleeping_as_fall"
SLEEPING_AS_FALL_REMAP = {
    0: 0,  # Sitting -> no_fall
    1: 1,  # Sleeping -> fall
    2: 0,  # Standing -> no_fall
    3: 0,  # Walking -> no_fall
    4: 1,  # falling -> fall
}


def configure_variant() -> None:
    converter.DEFAULT_OUTPUT_NAME = OUTPUT_NAME
    converter.SCRIPT_OUTPUT_DIR = converter.OUTPUT_DIR / Path(__file__).name
    converter.REPORTS_DIR = converter.SCRIPT_OUTPUT_DIR / "reports"
    converter.WORK_DIR = converter.SCRIPT_OUTPUT_DIR / "work"
    converter.CLASS_REMAP = SLEEPING_AS_FALL_REMAP


def main() -> None:
    configure_variant()
    converter.log("Mapping variant: Sitting/Standing/Walking -> no_fall; Sleeping/falling -> fall.")
    converter.main()


if __name__ == "__main__":
    main()
