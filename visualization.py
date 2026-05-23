import matplotlib.pyplot as plt
import librosa.display

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3), facecolor='none')

    librosa.display.waveshow(
        y,
        sr=sr,
        ax=ax,
        color='#00e676'
    )

    ax.set_title("Soil Sound Waveform", color='#ffffff', fontweight='bold', fontsize=12)
    ax.set_facecolor('none')
    ax.grid(alpha=0.18, color='#10b981')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors='#a7f3d0')

    return fig


def save_waveform(y, sr, out_path, dpi=150):
    """Create and save a waveform PNG to out_path and return the path."""
    fig = plot_waveform(y, sr)
    fig.savefig(out_path, bbox_inches='tight', dpi=dpi, facecolor='none')
    plt.close(fig)
    return out_path


def save_feature_plot(features: dict, out_path: str, dpi=150):
    """Save a small bar chart of extracted audio features to out_path."""
    keys = ["frequency", "energy", "zero_crossing", "duration"]
    values = [features.get(k, 0) for k in keys]

    # Create figure with dark facecolor for PDF context or clear layout
    fig, ax = plt.subplots(figsize=(6, 3), facecolor='#060b07')
    ax.set_facecolor('none')
    
    bars = ax.bar(keys, values, color=['#10b981', '#06b6d4', '#00e676', '#34d399'])
    ax.set_title("Extracted Audio Features", color='#ffffff', fontweight='bold')
    ax.set_ylabel("Value", color='#a7f3d0')
    ax.grid(axis='y', alpha=0.18, color='#10b981')
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    ax.tick_params(colors='#a7f3d0')

    # label bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8, color='#ffffff')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=dpi, facecolor='#060b07')
    plt.close(fig)
    return out_path