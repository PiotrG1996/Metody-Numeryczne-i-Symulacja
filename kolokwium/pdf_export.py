from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
from fpdf import FPDF

EXAM_TITLE = "Termin 1 - Laboratorium MNiS 21.06.2026"
FONT_DIR = Path(__file__).parent / "fonts"


def _resolve_fonts() -> tuple[str, str]:
    for regular, bold in (
        (FONT_DIR / "DejaVuSans.ttf", FONT_DIR / "DejaVuSans-Bold.ttf"),
        (FONT_DIR / "ArialUnicode.ttf", FONT_DIR / "ArialBold.ttf"),
    ):
        if regular.is_file() and bold.is_file():
            return str(regular), str(bold)
    raise FileNotFoundError(
        "Brak czcionek PDF w kolokwium/fonts/ (DejaVuSans lub ArialUnicode)."
    )


class ResultsPDF(FPDF):
    def __init__(self):
        super().__init__()
        regular, bold = _resolve_fonts()
        self.add_font("ExamFont", "", regular)
        self.add_font("ExamFont", "B", bold)

    def header_block(self, student: str, score_text: str, created: str) -> None:
        self.set_font("ExamFont", "B", 14)
        self.multi_cell(0, 8, EXAM_TITLE, align="C")
        self.ln(4)
        self.set_font("ExamFont", "B", 12)
        self.cell(0, 8, "Pobieranie wyniku", ln=True, align="C")
        self.ln(6)
        self.set_font("ExamFont", "", 11)
        self.cell(0, 7, f"Student: {student}", ln=True)
        self.cell(0, 7, f"Wynik: {score_text}", ln=True)
        self.cell(0, 7, f"Data: {created}", ln=True)
        self.ln(6)


def _score_from_df(df: pd.DataFrame) -> str:
    if df.empty or "Wynik" not in df.columns:
        return "—"
    correct = (df["Wynik"] == "Poprawnie").sum()
    return f"{correct}/{len(df)}"


def build_results_pdf_bytes(csv_path: Path, student: str, created: str | None = None) -> bytes:
    df = pd.read_csv(csv_path, encoding="utf-8")
    return _build_results_pdf_from_df(df, student, created)


def build_results_pdf_bytes_from_csv(
    csv_content: str, student: str, created: str | None = None
) -> bytes:
    from io import StringIO

    df = pd.read_csv(StringIO(csv_content), encoding="utf-8")
    return _build_results_pdf_from_df(df, student, created)


def _build_results_pdf_from_df(
    df: pd.DataFrame, student: str, created: str | None = None
) -> bytes:
    created = created or datetime.now().strftime("%Y-%m-%d %H:%M")
    score_text = _score_from_df(df)

    pdf = ResultsPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.header_block(student, score_text, created)

    col_widths = (10, 72, 18, 18, 22)
    headers = ("#", "Pytanie", "Twoja", "Poprawna", "Wynik")

    pdf.set_font("ExamFont", "B", 9)
    pdf.set_fill_color(230, 230, 230)
    for header, width in zip(headers, col_widths):
        pdf.cell(width, 7, header, border=1, fill=True)
    pdf.ln()

    pdf.set_font("ExamFont", "", 8)
    for idx, row in df.iterrows():
        if pdf.get_y() > 270:
            pdf.add_page()
            pdf.set_font("ExamFont", "B", 9)
            for header, width in zip(headers, col_widths):
                pdf.cell(width, 7, header, border=1, fill=True)
            pdf.ln()
            pdf.set_font("ExamFont", "", 8)

        question = str(row.get("Pytanie", ""))[:120]
        cells = (
            str(idx + 1),
            question,
            str(row.get("Twoja odpowiedź", "")),
            str(row.get("Poprawna odpowiedź", "")),
            str(row.get("Wynik", "")),
        )
        x0, y0 = pdf.get_x(), pdf.get_y()
        line_heights = []
        for text, width in zip(cells, col_widths):
            pdf.set_xy(x0, y0)
            pdf.multi_cell(width, 5, text, border=0)
            line_heights.append(pdf.get_y() - y0)
            x0 += width
        row_h = max(line_heights + [5])
        x0, y0 = pdf.get_x(), pdf.get_y()
        pdf.set_xy(pdf.l_margin, y0 - row_h)
        for width in col_widths:
            pdf.cell(width, row_h, "", border=1)
            pdf.set_x(pdf.get_x())
        pdf.set_xy(pdf.l_margin, y0)

    buffer = BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()


def save_results_pdf(csv_path: Path, student: str, created: str | None = None) -> Path:
    pdf_path = csv_path.with_suffix(".pdf")
    pdf_path.write_bytes(build_results_pdf_bytes(csv_path, student, created))
    return pdf_path
