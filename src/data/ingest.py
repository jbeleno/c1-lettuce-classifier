"""Bring the Roboflow download under ``data/raw`` via symlink. Non-destructive
— the original folder downloaded by the user is left untouched."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from src.config import RAW_DATASET_PATH, RAW_DIR
from src.utils import configure_logging

log = logging.getLogger(__name__)
LINK_NAME = "lettuce_pallets"


def main() -> Path:
    configure_logging()
    link = RAW_DIR / LINK_NAME
    target = RAW_DATASET_PATH

    if not target.exists():
        raise FileNotFoundError(f"Raw dataset not found at {target}")

    if link.is_symlink() or link.exists():
        if link.is_symlink() and Path(os.readlink(link)) == target:
            log.info("symlink already points to %s", target)
            return link
        link.unlink() if link.is_symlink() else None

    link.symlink_to(target)
    log.info("created symlink: %s -> %s", link, target)

    for split in ("train", "valid", "test"):
        sub = link / split
        if not sub.exists():
            raise FileNotFoundError(f"Expected split folder missing: {sub}")
        csv = sub / "_classes.csv"
        if not csv.exists():
            raise FileNotFoundError(f"Missing _classes.csv in {sub}")
    log.info("dataset structure validated (train/valid/test + _classes.csv each)")
    return link


if __name__ == "__main__":
    main()
