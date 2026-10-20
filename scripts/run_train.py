"""Entrena y compara Random Forest, SVM-RBF y MLP.

Uso:
    python scripts/run_train.py
    python scripts/run_train.py --dataset data/landmarks.csv
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_rush import config  # noqa: E402
from gesture_rush.dataset import load_dataset, validate_dataset  # noqa: E402
from gesture_rush.train import train_and_select  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrenamiento y comparacion de modelos")
    parser.add_argument("--dataset", default=str(config.DATASET_CSV))
    args = parser.parse_args()

    df = load_dataset(args.dataset)
    report = validate_dataset(df)
    sources = report.get("sources", {})
    if sources.get("synthetic"):
        print("[aviso] El dataset contiene datos sinteticos: los resultados no "
              "demuestran generalizacion a personas reales.")

    result = train_and_select(df)
    comp = result["comparison"]
    print("\n== Comparacion en validacion (macro-F1) ==")
    for name, info in comp.items():
        grouped = "" if info.get("cv_grouped", True) else " (intra-participante)"
        print(f"  {name:14s} cv={info['cv_score']:.3f}{grouped}  "
              f"val_f1={info['val_f1_macro']:.3f}  "
              f"params={info['best_params']}")

    tm = result["test_metrics"]
    print(f"\n== Mejor modelo: {result['best_model']} ==")
    print(f"  test accuracy : {tm['accuracy']:.3f}")
    print(f"  test f1_macro : {tm['f1_macro']:.3f}")
    print(f"  inferencia    : {tm['inference_ms_per_sample']:.3f} ms/muestra")
    print(f"  modelo        : {result['model_path']}")
    print(f"  informe       : {result['evaluation_path']}")
    print(f"  matriz        : {result['confusion_plot']}")
    if tm.get("top_confusions"):
        print("  confusiones frecuentes:")
        for c in tm["top_confusions"]:
            print(f"    {c['true']} -> {c['pred']} ({c['count']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
