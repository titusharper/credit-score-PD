from datetime import datetime

def build_report_markdown(model_name: str, metrics: dict, ks: float, gini: float, artifacts: list[str]) -> str:
    lines = []
    lines.append(f"# Credit Risk PD Model Report ({model_name.upper()})")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## Performance (OOF)")
    lines.append(f"- AUC: {metrics['oof_auc']:.6f}")
    lines.append(f"- PR-AUC: {metrics['oof_pr_auc']:.6f}")
    lines.append(f"- LogLoss: {metrics['oof_logloss']:.6f}")
    lines.append(f"- Brier: {metrics['oof_brier']:.6f}")
    lines.append(f"- ECE (15 bins): {metrics['oof_ece_15bins']:.6f}")
    lines.append(f"- KS: {ks:.6f}")
    lines.append(f"- Gini: {gini:.6f}")
    lines.append("")
    lines.append("## Saved artifacts (outputs/)")
    for a in artifacts:
        lines.append(f"- {a}")
    lines.append("")
    return "\n".join(lines)
