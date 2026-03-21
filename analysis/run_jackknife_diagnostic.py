"""
Offline Jackknife/LOAO diagnostic driver.

This script runs the jackknife fits (leave-one-animal-out) for selected
compound–isomer pairs and relies on `optimization.fit.run_jackknife_for_pair`
to save results under:

    results/optimization/jackknife/jackknife_<COMPOUND>_<ISOMER>_LOAO.csv

It is *not* used by the main optimization + Monte Carlo pipeline and can be
invoked separately when you want to inspect LOAO robustness.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimization.config import setup_context
from optimization.io import get_project_root
from optimization.fit import run_jackknife_for_pair

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline Jackknife/LOAO fits for diagnostic purposes."
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        metavar=("COMPOUND", "ISOMER"),
        help='Run only a single compound–isomer pair (e.g. "PFOS Linear").',
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    args = parse_args(argv)

    project_root = get_project_root()
    context = setup_context(project_root=project_root)

    all_pairs: List[Tuple[str, str]] = context.data_cache.get_all_pairs()

    if args.pair:
        selected_pair = tuple(args.pair)
        if selected_pair not in all_pairs:
            logger.error(
                "Requested pair %s %s not found in available pairs: %s",
                args.pair[0],
                args.pair[1],
                sorted(all_pairs),
            )
            return 1
        pairs: List[Tuple[str, str]] = [selected_pair]  # type: ignore[assignment]
    else:
        pairs = all_pairs

    logger.info(
        "[JACKKNIFE DIAGNOSTIC] Running LOAO fits for %d compound–isomer pairs",
        len(pairs),
    )

    n_success = 0
    for compound, isomer in pairs:
        logger.info("[JACKKNIFE DIAGNOSTIC] Starting %s %s", compound, isomer)
        res = run_jackknife_for_pair((compound, isomer), context=context)
        if res is not None:
            n_success += 1

    logger.info(
        "[JACKKNIFE DIAGNOSTIC] Completed. Successful jackknife fits: %d/%d",
        n_success,
        len(pairs),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

