import tempfile
import os
from report_generator import generate_report
from reportlab.lib.pagesizes import letter

def make_dummy_png(path):
    # create a tiny PNG using matplotlib
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0,1,2], [0,1,0])
    fig.savefig(path)
    plt.close(fig)


def test_generate_report_with_images():
    tmpdir = tempfile.mkdtemp()
    try:
        report_path = os.path.join(tmpdir, 'r.pdf')
        wf = os.path.join(tmpdir, 'wf.png')
        fp = os.path.join(tmpdir, 'fp.png')
        make_dummy_png(wf)
        make_dummy_png(fp)
        result = {
            'soil_type': 'UnitSoil',
            'moisture': 'Low',
            'compaction': 'Low',
            'dryness': 'Low',
            'health_score': 80,
            'recommendation': 'OK'
        }
        metadata = {'sample_id':'t1', 'location':'lab', 'uploaded_at':'now', 'filename':'r.pdf', 'feature_plot': fp}
        generate_report(result, report_path, waveform_path=wf, metadata=metadata)
        assert os.path.exists(report_path)
        assert os.path.getsize(report_path) > 0
    finally:
        try:
            os.remove(report_path)
            os.remove(wf)
            os.remove(fp)
            os.rmdir(tmpdir)
        except Exception:
            pass
