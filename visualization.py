import matplotlib.pyplot as plt
import librosa.display

def plot_waveform(y, sr):

    fig, ax = plt.subplots(figsize=(10, 3), facecolor='none')

    librosa.display.waveshow(
        y,
        sr=sr,
        ax=ax,
        color='#2e7d32'
    )

    ax.set_title("Soil Sound Waveform", color='#111827', fontweight='bold')
    ax.set_facecolor('none')
    ax.grid(alpha=0.15, color='#2e7d32')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors='#4b5563')

    return fig


def save_waveform(y, sr, out_path, dpi=150):
    """Create and save a waveform PNG to out_path and return the path."""
    fig = plot_waveform(y, sr)
    fig.savefig(out_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    return out_path


def save_feature_plot(features: dict, out_path: str, dpi=150):
    """Save a small bar chart of extracted audio features to out_path."""
    # select a few numeric features
    keys = ["frequency", "energy", "zero_crossing", "duration"]
    values = [features.get(k, 0) for k in keys]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(keys, values, color=['#6b8e23', '#8b5a2b', '#9fbf9f', '#c9b089'])
    ax.set_title("Extracted Audio Features")
    ax.set_ylabel("Value")
    ax.grid(axis='y', alpha=0.3)

    # label bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    return out_path