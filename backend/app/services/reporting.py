"""
PDF report generation for forensic analysis.
"""

import io
from datetime import UTC, datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app import __version__
from app.logging_config import get_logger

logger = get_logger(__name__)


def generate_pdf_report(
    job_id: str,
    verdict: str,
    confidence: float,
    sha256: str,
    model_version: str,
    runtime_ms: int,
    device: str,
    created_at: datetime,
    total_frames: int | None = None,
    analyzed_frames: int | None = None,
    timeline_chart: bytes | None = None,
    distribution_chart: bytes | None = None,
    suspicious_frames_montage: bytes | None = None,
) -> bytes:
    """
    Generate PDF forensic report.

    Args:
        job_id: Analysis job ID
        verdict: Detection verdict
        confidence: Confidence score
        sha256: File hash
        model_version: Model version used
        runtime_ms: Processing time
        device: Device used
        created_at: Analysis timestamp
        total_frames: Total video frames
        analyzed_frames: Number of analyzed frames
        timeline_chart: Timeline chart image bytes
        distribution_chart: Distribution chart image bytes
        suspicious_frames_montage: Suspicious frames montage bytes

    Returns:
        PDF document as bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor("#1a1a2e"),
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor("#16213e"),
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=6,
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.gray,
        spaceBefore=20,
        spaceAfter=10,
    )

    elements = []

    # Title
    elements.append(Paragraph("Deepfake Detection Analysis Report", title_style))

    # Disclaimer banner
    disclaimer_text = (
        "⚠️ <b>IMPORTANT DISCLAIMER:</b> This is a forensic estimate, not certainty. "
        "Automated detection systems can produce false positives and false negatives. "
        "Results should be verified by qualified human experts before making any decisions."
    )
    elements.append(Paragraph(disclaimer_text, disclaimer_style))
    elements.append(Spacer(1, 12))

    # Summary section
    elements.append(Paragraph("Analysis Summary", heading_style))

    summary_data = [
        ["Verdict", verdict],
        ["Confidence", f"{confidence:.1%}"],
        ["Job ID", job_id],
        ["Analysis Date", created_at.strftime("%Y-%m-%d %H:%M:%S UTC")],
    ]

    summary_table = Table(summary_data, colWidths=[2 * inch, 4 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#333333")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.gray),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # Technical details
    elements.append(Paragraph("Technical Details", heading_style))

    tech_data = [
        ["File Hash (SHA256)", sha256[:32] + "..."],
        ["Model Version", model_version],
        ["Processing Device", device],
        ["Processing Time", f"{runtime_ms} ms"],
        ["API Version", __version__],
    ]

    if total_frames is not None:
        tech_data.append(["Total Frames", str(total_frames)])
    if analyzed_frames is not None:
        tech_data.append(["Analyzed Frames", str(analyzed_frames)])

    tech_table = Table(tech_data, colWidths=[2 * inch, 4 * inch])
    tech_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.gray),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(tech_table)

    # Charts section
    if timeline_chart or distribution_chart:
        elements.append(PageBreak())
        elements.append(Paragraph("Analysis Visualizations", heading_style))

        if timeline_chart:
            elements.append(Paragraph("Score Timeline", body_style))
            img = Image(io.BytesIO(timeline_chart), width=6.5 * inch, height=2 * inch)
            elements.append(img)
            elements.append(Spacer(1, 20))

        if distribution_chart:
            elements.append(Paragraph("Score Distribution", body_style))
            img = Image(io.BytesIO(distribution_chart), width=5 * inch, height=2.5 * inch)
            elements.append(img)

    # Suspicious frames
    if suspicious_frames_montage:
        elements.append(PageBreak())
        elements.append(Paragraph("Suspicious Frames", heading_style))
        elements.append(
            Paragraph(
                "The following frames showed the highest fake probability scores:", body_style
            )
        )
        elements.append(Spacer(1, 10))
        img = Image(io.BytesIO(suspicious_frames_montage), width=6.5 * inch, height=4 * inch)
        elements.append(img)

    # Limitations section
    elements.append(PageBreak())
    elements.append(Paragraph("Limitations & Methodology", heading_style))

    limitations_text = """
    <b>Detection Methodology:</b><br/>
    This analysis uses deep learning models trained on known deepfake datasets. The model
    examines visual artifacts, facial inconsistencies, and temporal patterns that may
    indicate manipulation.<br/><br/>

    <b>Known Limitations:</b><br/>
    • The model may not detect novel deepfake techniques not present in training data<br/>
    • Heavy compression or low resolution can reduce detection accuracy<br/>
    • Legitimate video effects may trigger false positives<br/>
    • Results are probabilistic estimates, not definitive conclusions<br/><br/>

    <b>Recommended Use:</b><br/>
    This tool is intended for preliminary screening only. Any significant findings should be
    verified by qualified digital forensics experts using multiple independent methods before
    drawing conclusions or taking action.
    """
    elements.append(Paragraph(limitations_text, body_style))

    # Footer disclaimer
    elements.append(Spacer(1, 30))
    footer_text = (
        "This report was generated automatically by the Deepfake Detection Platform. "
        f"Report generated on {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}."
    )
    elements.append(Paragraph(footer_text, disclaimer_style))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)

    return buffer.getvalue()
