
import os
import re
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


GRADE_LABELS = {
    0: "No DR",
    1: "Mild Non-Proliferative DR",
    2: "Moderate Non-Proliferative DR",
    3: "Severe Non-Proliferative DR",
    4: "Proliferative DR"
}


def create_report(grade, probabilities, image_name="Retinal Image"):

    # Validate prediction
    grade = int(grade)
    probabilities = [float(x) for x in probabilities]

    if grade not in GRADE_LABELS:
        raise ValueError("Grade must be between 0 and 4.")

    if len(probabilities) != 5:
        raise ValueError("Expected exactly 5 probabilities.")

    if any(not (0 <= p <= 1) for p in probabilities):
        raise ValueError("Probabilities must be between 0 and 1.")

    if not all(map(lambda x: x == x and abs(x) != float("inf"), probabilities)):
        raise ValueError("Invalid probability values.")

    total = sum(probabilities)

    if total <= 0:
        raise ValueError("Probability sum must be greater than zero.")

    # Normalize probabilities if necessary
    probabilities = [p / total for p in probabilities]

    # Use model's actual predicted grade
    predicted_grade = grade
    predicted_label = GRADE_LABELS[grade]

    confidence = probabilities[grade] * 100
    screening_result = (
        "Referable DR" if grade >= 2 else "Non-Referable DR"
    )

    if grade >= 2:
        recommendation = (
            "The model prediction falls in the referable screening category. "
            "Further evaluation by a qualified ophthalmologist is recommended."
        )
    else:
        recommendation = (
            "The model prediction falls in the non-referable screening category. "
            "Continue routine eye screening as advised by an eye-care professional."
        )

    if confidence >= 80:
        confidence_level = "High model probability"
    elif confidence >= 60:
        confidence_level = "Moderate model probability"
    else:
        confidence_level = "Low model probability"

    current_time = datetime.now().strftime("%d-%m-%Y %H:%M")

    # Natural Language Generation
    report_text = f"""
DIABETIC RETINOPATHY SCREENING REPORT

Image: {image_name}
Date: {current_time}

PREDICTION RESULTS

Predicted Grade: Grade {predicted_grade}
Classification: {predicted_label}
Screening Category: {screening_result}

MODEL OUTPUT

Highest Class Probability: {confidence:.2f}%
Probability Interpretation: {confidence_level}

NATURAL LANGUAGE SUMMARY

The uploaded retinal image has been analyzed by the
Deep Learning model. The model classified the image as
Grade {predicted_grade}, corresponding to {predicted_label}.

The predicted class probability is {confidence:.2f}%.
The screening category generated from this prediction
is {screening_result}.

RECOMMENDATION

{recommendation}

CLASS-WISE PROBABILITIES

{chr(10).join(
    f"Grade {i} - {GRADE_LABELS[i]}: {probabilities[i] * 100:.2f}%"
    for i in range(5)
)}

DISCLAIMER

This is an AI-generated research screening report.
It is not a medical diagnosis. Model predictions may be incorrect.
Consult a qualified ophthalmologist for clinical evaluation.
"""

    # Create reports directory
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Safe filename
    safe_name = re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        os.path.splitext(os.path.basename(str(image_name)))[0]
    )[:40]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    pdf_path = os.path.join(
        reports_dir,
        f"DR_Report_{safe_name}_{timestamp}.pdf"
    )

    # PDF document
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#173B65"),
        spaceAfter=18
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#173B65"),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        leading=15,
        spaceAfter=7
    )

    story = []

    story.append(
        Paragraph("Diabetic Retinopathy Screening Report", title_style)
    )

    story.append(
        Paragraph(
            f"<b>Image:</b> {escape(os.path.basename(str(image_name)))}<br/>"
            f"<b>Date:</b> {current_time}",
            body_style
        )
    )

    story.append(
        Paragraph("Prediction Results", heading_style)
    )

    result_data = [
        ["Parameter", "Result"],
        ["Predicted Grade", f"Grade {predicted_grade}"],
        ["Classification", predicted_label],
        ["Screening Category", screening_result],
        ["Class Probability", f"{confidence:.2f}%"],
        ["Probability Interpretation", confidence_level]
    ]

    result_table = Table(
        result_data,
        colWidths=[2.5 * inch, 3.8 * inch]
    )

    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173B65")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    story.append(result_table)

    story.append(
        Paragraph("Natural Language Summary", heading_style)
    )

    story.append(
        Paragraph(
            escape(
                f"The uploaded retinal image was classified as Grade "
                f"{predicted_grade} ({predicted_label}). "
                f"The model assigned {confidence:.2f}% probability "
                f"to this predicted class."
            ),
            body_style
        )
    )

    story.append(
        Paragraph("Recommendation", heading_style)
    )

    story.append(
        Paragraph(escape(recommendation), body_style)
    )

    story.append(
        Paragraph("Class-wise Probabilities", heading_style)
    )

    probability_data = [
        ["Grade", "Classification", "Probability"]
    ]

    for i in range(5):
        probability_data.append([
            f"Grade {i}",
            GRADE_LABELS[i],
            f"{probabilities[i] * 100:.2f}%"
        ])

    probability_table = Table(
        probability_data,
        colWidths=[1.1 * inch, 3.7 * inch, 1.5 * inch]
    )

    probability_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173B65")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7)
    ]))

    story.append(probability_table)

    story.append(Spacer(1, 18))

    story.append(
        Paragraph("Important Disclaimer", heading_style)
    )

    story.append(
        Paragraph(
            "This is an AI-generated research screening report, "
            "not a medical diagnosis. Model predictions may be incorrect. "
            "Consult a qualified ophthalmologist for clinical evaluation.",
            body_style
        )
    )

    doc.build(story)

    return report_text.strip(), pdf_path
