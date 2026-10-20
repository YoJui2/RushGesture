"""Validacion cruzada agrupada por participante (Leave-One-Participant-Out).

En cada fold un participante completo queda como test, otro se reserva para
validacion y el resto se usa para entrenar. Asi cada persona es evaluada una
vez y el resultado agregado (media +/- desviacion) es mas estable que un unico
split cuando hay pocos participantes.

Uso:
    python scripts/run_cv.py
    python scripts/run_cv.py --dataset data/landmarks.csv
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_rush import config  # noqa: E402
from gesture_rush.dataset import load_dataset, validate_dataset  # noqa: E402
from gesture_rush.train import cross_validate_by_participant  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validacion cruzada agrupada por participante"
    )
    parser.add_argument("--dataset", default=str(config.DATASET_CSV))
    parser.add_argument("--validation-fraction", type=float, default=0.25,
                        help="Fraccion de participantes restantes para validacion")
    parser.add_argument(
        "--report", default=str(config.REPORTS_DIR / "cv_report.md")
    )
    parser.add_argument(
        "--summary", default=str(config.REPORTS_DIR / "cv_summary.json")
    )
    parser.add_argument(
        "--confusion-plot",
        default=str(config.REPORTS_DIR / "cv_confusion_matrix.png"),
    )
    args = parser.parse_args()

    df = load_dataset(args.dataset)
    report = validate_dataset(df)
    if report.get("sources", {}).get("synthetic"):
        print("[aviso] El dataset contiene datos sinteticos: los resultados no "
              "demuestran generalizacion a personas reales.")

    result = cross_validate_by_participant(
        df,
        validation_fraction=args.validation_fraction,
        report_path=args.report,
        summary_path=args.summary,
        confusion_plot_path=args.confusion_plot,
    )

    print("\n== Resultados por fold (test) ==")
    for fold in result["folds"]:
        print(
            f"  fold {fold['fold']}: test={','.join(fold['test_groups'])} "
            f"train={','.join(fold['train_groups'])} "
            f"val={','.join(fold['val_groups'])} "
            f"modelo={fold['best_model']} "
            f"acc={fold['test']['accuracy']:.3f} "
            f"f1={fold['test']['f1_macro']:.3f}"
        )

    agg = result["aggregate"]
    print("\n== Agregado en test ==")
    print(f"  accuracy : {agg['accuracy_mean']:.4f} +/- {agg['accuracy_std']:.4f}")
    print(f"  f1_macro : {agg['f1_macro_mean']:.4f} +/- {agg['f1_macro_std']:.4f}")
    print(f"  f1_weighted: {agg['f1_weighted_mean']:.4f} +/- "
          f"{agg['f1_weighted_std']:.4f}")
    print("\n  F1 por clase:")
    for label, info in agg["per_class"].items():
        print(f"    {label:15s} {info['f1_mean']:.3f} +/- {info['f1_std']:.3f}")
    if agg.get("top_confusions"):
        print("\n  confusiones frecuentes:")
        for c in agg["top_confusions"]:
            print(f"    {c['true']} -> {c['pred']} ({c['count']})")
    print(f"\n  informe : {result['report_path']}")
    print(f"  resumen : {result['summary_path']}")
    print(f"  matriz  : {result['confusion_plot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
