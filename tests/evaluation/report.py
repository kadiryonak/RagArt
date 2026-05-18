"""Markdown ve JSON rapor üretimi."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from tests.evaluation.runner import RunReport


def render_markdown(report: RunReport, title: Optional[str] = None) -> str:
    """RunReport → human-readable Markdown."""
    title = title or "RAG Evaluation Report"
    lines = [
        f"# {title}",
        "",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Özet",
        "",
        f"- **Toplam örnek:** {len(report.items)}",
        f"- **Toplam süre:** {report.total_duration_s:.2f} sn",
        f"- **Çalıştırılan katmanlar:** {', '.join(report.layers_used)}",
        f"- **Ortalama skor:** {report.to_dict()['overall_avg_score']:.4f}",
        "",
        "## Katman bazında özet",
        "",
        "| Katman | N | Ort. skor | Pass oranı | Ort. latency (ms) |",
        "|---|---|---|---|---|",
    ]
    for name, stats in report.aggregate_by_layer().items():
        lines.append(
            f"| {name} | {stats['n']} | {stats['avg_score']:.4f} | "
            f"{stats['pass_rate']:.2%} | {stats['avg_latency_ms']:.1f} |"
        )

    lines += ["", "## Zorluk bazında özet", "", "| Zorluk | N | Ort. skor |", "|---|---|---|"]
    for diff, stats in report.aggregate_by_difficulty().items():
        lines.append(f"| {diff} | {stats['n']} | {stats['avg_score']:.4f} |")

    lines += ["", "## Örnek detayları", ""]
    for item in report.items:
        flag = "✓" if item.all_passed else "✗"
        lines.append(
            f"### {flag} `{item.item.id}` (skor: {item.overall_score:.3f}, "
            f"zorluk: {item.item.difficulty}, kategori: {item.item.category}"
            + (", **CRITICAL**" if item.item.critical else "")
            + ")"
        )
        lines.append("")
        lines.append(f"**Soru:** {item.item.question}")
        lines.append("")
        lines.append(f"**Cevap:** {item.output.answer[:300]}"
                     + ("…" if len(item.output.answer) > 300 else ""))
        lines.append("")
        lines.append("| Katman | Skor | Geçti | Latency |")
        lines.append("|---|---|---|---|")
        for layer in item.layers:
            note = ""
            if layer.error:
                note = f" — error: {layer.error}"
            elif layer.details.get("skipped"):
                note = " — skipped"
            lines.append(
                f"| {layer.layer} | {layer.score:.3f} | "
                f"{'✓' if layer.passed else '✗'} | {layer.latency_ms:.1f}ms{note} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_reports(
    report: RunReport,
    out_dir: str = "tests/evaluation/reports",
    name_prefix: Optional[str] = None,
) -> dict:
    """Hem markdown hem JSON yaz. Dönüş: {'md': path, 'json': path}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{name_prefix}_{ts}" if name_prefix else f"eval_{ts}"

    md_path = out / f"{prefix}.md"
    json_path = out / f"{prefix}.json"

    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"md": str(md_path), "json": str(json_path)}
