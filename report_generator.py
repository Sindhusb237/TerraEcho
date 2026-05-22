import os

try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
except ImportError as e:
    raise ImportError(
        "report_generator requires the reportlab package. Install dependencies with 'pip install -r requirements.txt' or 'pip install reportlab'."
    ) from e


_TITLE_STYLE = ParagraphStyle(
    name='Title',
    fontSize=18,
    leading=22,
    spaceAfter=12
)

def generate_report(result, filename, waveform_path=None, metadata=None):
    """Generate a PDF report including key metrics and optional waveform image.

    result: dict of analysis results
    filename: output PDF path
    waveform_path: optional path to PNG waveform to embed
    metadata: optional dict with sample metadata
    """
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()

    content = []

    title = Paragraph("TerraEcho Soil Health Report", _TITLE_STYLE)
    # try to include a colored header bar with optional logo
    logo_path = os.path.join('assets', 'logo.png')
    header_cells = []
    if os.path.exists(logo_path):
        try:
            logo = Image(logo_path, width=0.8 * inch, height=0.8 * inch)
            header_cells.append(logo)
        except Exception:
            header_cells.append(Paragraph(''))
    else:
        header_cells.append(Paragraph(''))

    header_cells.append(Paragraph("TerraEcho Soil Health Report", _TITLE_STYLE))
    header_table = Table([header_cells], colWidths=[0.9 * inch, 6.1 * inch])
    header_table.setStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#6b8e23')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.whitesmoke),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ])
    content.append(header_table)
    content.append(Spacer(1, 6))

    # metadata
    if metadata:
        meta_items = [[Paragraph(f"<b>{k}</b>", styles['BodyText']), Paragraph(str(v), styles['BodyText'])] for k, v in metadata.items() if v]
        if meta_items:
            tbl = Table(meta_items, hAlign='LEFT', colWidths=[1.5 * inch, 4.5 * inch])
            content.append(tbl)
            content.append(Spacer(1, 10))

    # Key metrics table
    metrics = [
        ['Soil Type', result.get('soil_type', '')],
        ['Moisture', result.get('moisture', '')],
        ['Compaction', result.get('compaction', '')],
        ['Dryness', result.get('dryness', '')],
        ['Health Score', f"{result.get('health_score', '')}%"],
        ['Recommendation', result.get('recommendation', '')]
    ]
    tbl = Table(metrics, hAlign='LEFT', colWidths=[2 * inch, 4 * inch])
    tbl.setStyle([('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke)])
    content.append(tbl)
    content.append(Spacer(1, 12))

    # waveform image
    if waveform_path and os.path.exists(waveform_path):
        try:
            img = Image(waveform_path)
            img._restrictSize(6.5 * inch, 3 * inch)
            content.append(Paragraph("Waveform", styles['Heading3']))
            content.append(Spacer(1, 6))
            content.append(img)
            content.append(Spacer(1, 12))
        except Exception:
            content.append(Paragraph("[Waveform image unavailable]", styles['BodyText']))

    # feature plot (bar chart)
    feature_path = metadata.get('feature_plot') if metadata else None
    if feature_path and os.path.exists(feature_path):
        try:
            fimg = Image(feature_path)
            fimg._restrictSize(6.5 * inch, 2.5 * inch)
            content.append(Paragraph("Feature Summary", styles['Heading3']))
            content.append(Spacer(1, 6))
            content.append(fimg)
        except Exception:
            content.append(Paragraph("[Feature plot unavailable]", styles['BodyText']))

    doc.build(content)


if __name__ == '__main__':
    sample_result = {
        'soil_type': 'Loam',
        'moisture': 'Moderate',
        'compaction': 'Low',
        'dryness': 'Optimal',
        'health_score': 92,
        'recommendation': 'Maintain current irrigation and add organic mulch.'
    }
    sample_metadata = {
        'Sample ID': 'TEST-001',
        'Location': 'Demo Field',
        'feature_plot': None
    }
    output_path = os.path.join('reports', 'test_report.pdf')
    generate_report(sample_result, output_path, waveform_path=None, metadata=sample_metadata)
    print(f'Generated {output_path}')
