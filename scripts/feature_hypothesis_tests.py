"""Statistical hypothesis testing of the position model's engineered features.

Runs the leakage-aware battery in ``pipeline/analyze.py`` — univariate
association tests with within-Race permutation p and BH-FDR, cluster-robust /
mixed-effects / GEE multivariable inference, GroupKFold(by Race) model-based
importance, and leakage quantification — over a *raw* (leakage-flagged) and a
*clean* (leakage-removed) feature frame, and writes:

    outputs/feature_hypothesis_tests.md
    outputs/feature_hypothesis_tests.csv                 (univariate)
    outputs/feature_hypothesis_tests_multivariable.csv
    outputs/feature_hypothesis_tests_model.csv
    outputs/feature_hypothesis_tests_leakage.json

Examples
--------
    venv/bin/python scripts/feature_hypothesis_tests.py --no-model --frames both
    venv/bin/python scripts/feature_hypothesis_tests.py --target Position --seed 42
    venv/bin/python scripts/feature_hypothesis_tests.py --frames both --plots
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.analyze import ALL_TARGETS, run_feature_analysis, write_outputs  # noqa: E402
from pipeline.config_loader import get_config  # noqa: E402

OUT_DIR = ROOT / "outputs"


def _parse_targets(raw: str) -> tuple[str, ...]:
    wanted = [t.strip() for t in raw.split(",") if t.strip()]
    bad = [t for t in wanted if t not in ALL_TARGETS]
    if bad:
        raise SystemExit(f"unknown target(s) {bad}; choose from {list(ALL_TARGETS)}")
    return tuple(wanted)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--target", default=",".join(ALL_TARGETS),
                    help="comma list; default all four (%(default)s)")
    ap.add_argument("--frames", choices=["raw", "clean", "both"], default="both")
    ap.add_argument("--no-model", action="store_true", help="skip Phase 4 (model-based importance)")
    ap.add_argument("--permutations", type=int, default=2000, help="within-group permutation B")
    ap.add_argument("--perm-importance-repeats", type=int, default=50)
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--drop-rates", action="store_true",
                    help="clean frame drops driver_win_rate / team_reliability entirely")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = get_config()
    targets = _parse_targets(args.target)
    frames = ("raw", "clean") if args.frames == "both" else (args.frames,)
    out_dir = Path(args.out_dir)

    analysis = run_feature_analysis(
        cfg,
        targets=targets,
        frames=frames,
        run_model=not args.no_model,
        n_perm=args.permutations,
        perm_importance_repeats=args.perm_importance_repeats,
        cv_splits=args.cv_splits,
        seed=args.seed,
        make_plots=args.plots,
        drop_rates=args.drop_rates,
        out_dir=(out_dir / "plots" / "feature_stats") if args.plots else None,
    )

    paths = write_outputs(analysis, out_dir)
    print(analysis["markdown"])
    print("\nwrote:")
    for name, path in paths.items():
        print(f"  {name:14} {path}")
    _sanity_checks(analysis)


def _sanity_checks(analysis: dict) -> None:
    """Print the Phase-9 sanity checks so a run is self-verifying."""
    uni = analysis.get("univariate")
    model = analysis.get("model")
    leak = analysis.get("leakage", {})
    lines = ["", "sanity checks:"]

    if uni is not None and not uni.empty:
        raw_pos = uni[(uni["frame"] == "raw") & (uni["target"] == "Position")
                      & (uni["test"] == "spearman")]
        if not raw_pos.empty:
            ranked = raw_pos.reindex(raw_pos["statistic"].abs().sort_values(ascending=False).index)
            ranked = ranked.reset_index(drop=True)
            lines.append(f"  raw/Position top-3 by |Spearman|: "
                         + ", ".join(f"{r['feature']}({r['statistic']:+.2f})"
                                     for _, r in ranked.head(3).iterrows()))
            pc = ranked[ranked["feature"] == "PositionChange"]
            if not pc.empty:
                rank = int(pc.index[0]) + 1
                lines.append(f"  PositionChange raw: |Spearman| rank {rank}/{len(ranked)}, "
                             f"rho={pc.iloc[0]['statistic']:+.3f}, p_perm={pc.iloc[0]['p_perm']:.4g}, "
                             f"leaky={bool(pc.iloc[0]['leaky'])} "
                             "(leakage surfaces in the multivariable degenerate fit and "
                             "drop-one CV delta, not univariate Spearman)")
        clean_feats = set(uni[uni["frame"] == "clean"]["feature"])
        lines.append(f"  clean frame contains PositionChange: {'PositionChange' in clean_feats}")
        for f in ("driver_win_rate", "team_reliability"):
            row = uni[(uni["frame"] == "raw") & (uni["feature"] == f)]
            if not row.empty:
                lines.append(f"  {f} raw perm_scheme: {row.iloc[0]['perm_scheme']} "
                             f"(p_perm={row.iloc[0]['p_perm']:.3g})")

    lt = leak.get("leaky_trio_cv_mae", {})
    if lt:
        lines.append(f"  position-model CV MAE with leaky trio={lt['cv_mae_with_trio']:.3f} "
                     f"vs without={lt['cv_mae_without_trio']:.3f}")
    idy = leak.get("identity", {})
    if idy:
        lines.append(f"  identity R^2={idy['r2']:.6f} (max abs residual {idy['max_abs_residual']:.2g})")
    if model is not None and not model.empty:
        assert (model["p_perm_importance"].dropna().between(0, 1)).all(), "p_perm_importance out of [0,1]"
        lines.append("  model p_perm_importance all in [0, 1]: True")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
