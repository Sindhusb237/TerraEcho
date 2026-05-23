import streamlit as st
import tempfile
import os
import uuid
import io
import base64
import csv
import zipfile
import subprocess
import sys
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from translate import t
from audio_processing import process_audio
from prediction_engine import predict_soil
from visualization import plot_waveform, save_feature_plot
from report_generator import generate_report
from database import init_db, save_result, fetch_all

# ─── Page config MUST be the first Streamlit call ──────────────────────────
st.set_page_config(
    page_title="TerraEcho | AgriTech IQ",
    page_icon="🌿",
    layout="wide",
)

# ─── Database init ──────────────────────────────────────────────────────────
init_db()

# ─── CSS ────────────────────────────────────────────────────────────────────
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.markdown(
        """
        <style>
        .stApp { font-family: 'Inter', sans-serif; background: #f8faf8; color: #0b170b; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ─── Sidebar ─────────────────────────────────────────────────────────────────
NAV_OPTIONS = [
    "🏠 Dashboard",
    "🌱 Soil Analysis",
    "📄 Reports",
    "🤖 AI Insights",
    "🕒 History",
    "ℹ About",
]

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px;padding:16px 0 8px 0;">
            <div style="font-size:2rem;">🌿</div>
            <div>
                <div style="font-size:1.1rem;font-weight:800;color:#0f3d1e;">TerraEcho</div>
                <div style="font-size:0.7rem;color:#6b7280;margin-top:2px;">Gen5 Soil Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    languages = ["English", "Kannada", "Hindi"]
    lang = st.selectbox("🌐 Language", languages, key="global_lang")

    mode = st.radio("Navigation", NAV_OPTIONS, index=1, key="nav_mode")

    st.markdown("<hr style='border-color:#d1fae5;margin:12px 0;'>", unsafe_allow_html=True)

    with st.form(key="meta_form"):
        st.markdown(
            "<div style='font-size:0.7rem;font-weight:700;color:#059669;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>Sample Metadata</div>",
            unsafe_allow_html=True,
        )
        sample_id = st.text_input("Sample ID", value=str(uuid.uuid4())[:8])
        location = st.text_input("Location")
        notes = st.text_area("Notes", height=80)
        submitted = st.form_submit_button("💾 Save Metadata")

    st.markdown("<hr style='border-color:#d1fae5;margin:12px 0;'>", unsafe_allow_html=True)

    if st.button("🔧 Run Diagnostics"):
        with st.expander("Test Output", expanded=True):
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                st.code(proc.stdout or "No output")
                if proc.stderr:
                    st.code(proc.stderr)
                st.success("Diagnostics complete")
            except subprocess.TimeoutExpired:
                st.error("Diagnostics timed out")
            except Exception as exc:
                st.error(f"Diagnostics error: {exc}")

    st.markdown(
        "<div style='font-size:0.65rem;color:#9ca3af;text-align:center;margin-top:16px;'>TerraEcho • Sustainable agriculture for farms, researchers, and agri-tech innovators.</div>",
        unsafe_allow_html=True,
    )


# ─── Helper functions ─────────────────────────────────────────────────────────

def build_soil_profile(result):
    """Build an extended soil profile dict from prediction result."""
    confidence = min(100, max(65, int(result["health_score"] * 0.38 + 25)))
    crop_map = {
        "Dry Soil": "Millet, Sorghum, Chickpea",
        "Healthy Soil": "Wheat, Maize, Soybeans",
        "Compact Soil": "Barley, Beans, Radish",
    }
    water_map = {
        "Dry Soil": "High",
        "Healthy Soil": "Moderate",
        "Compact Soil": "Low",
    }
    treatment_map = {
        "Dry Soil": "Irrigate & mulch to lock moisture.",
        "Healthy Soil": "Maintain organic cover and routine sampling.",
        "Compact Soil": "Aerate and add compost to improve texture.",
    }
    sustainability = min(100, result["health_score"] + 10)
    return {
        "confidence": confidence,
        "suggested_crops": crop_map.get(result["soil_type"], "Adaptive grasses"),
        "water": water_map.get(result["soil_type"], "Moderate"),
        "treatment": treatment_map.get(result["soil_type"], "Optimize organic matter and aeration."),
        "sustainability": sustainability,
    }


def get_ai_confidence(result, features):
    """Estimate AI confidence from soil health and acoustic features."""
    base_score = result.get("health_score", 50)
    energy = min(1.0, features.get("energy", 0) * 7)
    freq_ratio = min(1.0, features.get("frequency", 0) / 5000)
    confidence = int(min(100, max(65, base_score * 0.65 + energy * 18 + freq_ratio * 17)))
    return confidence


def get_comparison_dataframe(result):
    return pd.DataFrame(
        {
            "Parameter": ["Moisture", "Compaction", "Health Score"],
            "Current": [
                result.get("moisture", "-"),
                result.get("compaction", "-"),
                f"{result.get('health_score', 0)}%",
            ],
            "Ideal": ["Moderate", "Low", ">85%"],
        }
    )


def get_ai_assistant_response(question):
    prompt = question.lower()
    if "compact" in prompt or "aerate" in prompt or "drainage" in prompt:
        return "Loosen the soil and add compost. Reduce heavy equipment and improve drainage for better root health."
    if "dry" in prompt or "irrigation" in prompt or "moisture" in prompt:
        return "Increase irrigation, add mulch, and retain moisture with cover crops or organic mulch."
    if "healthy" in prompt or "good" in prompt or "maintain" in prompt:
        return "Soil condition is good. Keep monitoring and maintain your current sustainable practices."
    return "TerraEcho suggests field sampling and routine soil monitoring. Ask about compact, dry, or healthy soil conditions."


# ─── Page renderers ───────────────────────────────────────────────────────────

def render_hero():
    st.markdown(
        """
        <section style="background:linear-gradient(135deg,#ecfdf5 0%,#f0fdf4 60%,#dcfce7 100%);border:1px solid #bbf7d0;border-radius:2rem;padding:2.5rem;margin:1.5rem 0;position:relative;overflow:hidden;">
            <div style="position:absolute;top:-40px;left:-40px;width:180px;height:180px;border-radius:50%;background:rgba(16,185,129,0.08);filter:blur(40px);pointer-events:none;"></div>
            <div style="position:absolute;bottom:-40px;right:-40px;width:220px;height:220px;border-radius:50%;background:rgba(5,150,105,0.08);filter:blur(40px);pointer-events:none;"></div>
            <div style="position:relative;z-index:1;">
                <span style="font-size:0.65rem;font-weight:700;color:#059669;text-transform:uppercase;letter-spacing:0.15em;">AI-Based Soil Sound Intelligence System</span>
                <h1 style="font-size:2.8rem;font-weight:900;color:#0f3d1e;margin:0.5rem 0 1rem 0;line-height:1.1;">🌱 TerraEcho</h1>
                <p style="color:#374151;font-size:0.95rem;max-width:560px;line-height:1.7;margin-bottom:1.5rem;">
                    A professional agriculture dashboard for soil sound analytics, AI-driven health scores, and smart farming guidance.
                </p>
                <div style="display:flex;gap:12px;flex-wrap:wrap;">
                    <span style="display:inline-block;padding:10px 24px;border-radius:9999px;background:linear-gradient(135deg,#059669,#10b981);color:#fff;font-weight:700;font-size:0.85rem;">Analyze Soil</span>
                    <span style="display:inline-block;padding:10px 24px;border-radius:9999px;border:1px solid #6ee7b7;color:#059669;font-weight:600;font-size:0.85rem;">Download Report</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard():
    rows = fetch_all()
    total_samples = len(rows)
    avg_health = int(sum([r.get("health_score", 0) for r in rows]) / total_samples) if total_samples else 0
    recent_soil = rows[0]["soil_type"] if rows else "N/A"
    health_bucket = "Excellent" if avg_health >= 80 else "Stable" if avg_health >= 60 else "Attention"

    render_hero()

    st.markdown("<h2 style='color:#0f3d1e;'>🏠 Enterprise Soil Overview</h2>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, total_samples, "Processed Samples"),
        (c2, f"{avg_health}%", "Average Health Score"),
        (c3, recent_soil, "Recent Soil Type"),
        (c4, health_bucket, "Farming Readiness"),
    ]:
        col.markdown(
            f"""
            <div style="padding:1.5rem;background:#fff;border:1px solid #d1fae5;border-radius:1.25rem;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="font-size:1.8rem;font-weight:900;color:#059669;">{val}</div>
                <div style="font-size:0.65rem;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:0.1em;margin-top:4px;">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<h3 style='color:#0f3d1e;margin-top:2rem;'>📈 Trend Monitoring</h3>", unsafe_allow_html=True)
    if rows:
        health_series = [r.get("health_score", 0) for r in rows[:8]][::-1]
        dates = [r.get("uploaded_at", "")[-6:] for r in rows[:8]][::-1]
        fig, ax = plt.subplots(figsize=(10, 3), facecolor="none")
        ax.plot(dates, health_series, color="#059669", marker="o", linewidth=2)
        ax.fill_between(dates, health_series, color="#059669", alpha=0.15)
        ax.set_title("Soil Health Trend", color="#062c11", fontsize=11, fontweight="bold")
        ax.set_facecolor("none")
        ax.grid(alpha=0.15, color="#059669")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors="#4a6b52")
        st.pyplot(fig)
    else:
        st.info("Analyze your first sample to populate enterprise dashboards.")


def render_analysis():
    st.markdown("<h2 style='color:#0f3d1e;'>🌱 Soil Analysis</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#374151;font-size:0.9rem;margin-bottom:1.5rem;'>Upload soil tapping audio to analyze soil health with pro-grade AI intelligence.</p>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([2.8, 1.2])
    with left:
        st.markdown(
            """
            <div style="background:#f0fdf4;border:2px dashed #6ee7b7;border-radius:1.5rem;padding:2rem;text-align:center;margin-bottom:1rem;">
                <div style="font-size:2.5rem;margin-bottom:0.5rem;">🎧</div>
                <div style="font-weight:700;color:#0f3d1e;font-size:1rem;margin-bottom:0.5rem;">Upload soil tapping audio</div>
                <div style="font-size:0.8rem;color:#6b7280;">Drag & drop your .wav or .mp3 file · Max 20 MB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3"], label_visibility="collapsed")
        if audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format=audio_file.type)
        else:
            audio_bytes = None

    with right:
        st.markdown(
            """
            <div style="background:#fff;border:1px solid #d1fae5;border-radius:1.25rem;padding:1.5rem;">
                <div style="font-weight:700;color:#0f3d1e;margin-bottom:0.75rem;">Premium Analysis Kit</div>
                <ul style="font-size:0.82rem;color:#374151;line-height:2;padding-left:1.2rem;">
                    <li>Modern soil dashboard</li>
                    <li>AI-driven recommendations</li>
                    <li>Rich visual storytelling</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not (audio_file and audio_bytes is not None):
        return

    with st.spinner("🔍 TerraEcho AI analyzing soil..."):
        progress = st.progress(0)
        progress.progress(20)
        suffix = os.path.splitext(audio_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name
        progress.progress(45)
        features = process_audio(temp_path)
        result = predict_soil(features)
        profile = build_soil_profile(result)
        confidence = get_ai_confidence(result, features)
        duration = round(features.get("duration", 0), 1)
        progress.progress(100)

    # Audio details
    st.markdown(
        f"""
        <div style="background:#fff;border:1px solid #d1fae5;border-radius:1.5rem;padding:1.5rem;margin:1.5rem 0;">
            <div style="font-weight:700;color:#059669;margin-bottom:0.75rem;">🎵 Uploaded Audio Details</div>
            <div style="font-size:0.85rem;font-weight:600;color:#0f3d1e;margin-bottom:1rem;">{audio_file.name}</div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem;">
                <div style="background:#f0fdf4;padding:0.75rem;border-radius:0.75rem;font-size:0.75rem;">
                    <div style="color:#6b7280;margin-bottom:4px;">Format</div>
                    <div style="font-weight:700;color:#0f3d1e;">{audio_file.type or 'Audio'}</div>
                </div>
                <div style="background:#f0fdf4;padding:0.75rem;border-radius:0.75rem;font-size:0.75rem;">
                    <div style="color:#6b7280;margin-bottom:4px;">Size</div>
                    <div style="font-weight:700;color:#0f3d1e;">{round(audio_file.size / 1024, 1)} KB</div>
                </div>
                <div style="background:#f0fdf4;padding:0.75rem;border-radius:0.75rem;font-size:0.75rem;">
                    <div style="color:#6b7280;margin-bottom:4px;">Duration</div>
                    <div style="font-weight:700;color:#0f3d1e;">{duration} sec</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    soil_type = result["soil_type"]
    if soil_type == "Healthy Soil":
        badge_bg, badge_text = "#ecfdf5", "#065f46"
    elif soil_type == "Dry Soil":
        badge_bg, badge_text = "#fffbeb", "#92400e"
    else:
        badge_bg, badge_text = "#fff1f2", "#9f1239"

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, label, value in [
        (c1, "🟫", "Soil Type", result["soil_type"]),
        (c2, "💧", "Moisture", result["moisture"]),
        (c3, "🌱", "Health Score", f"{result['health_score']}%"),
        (c4, "🤖", "AI Confidence", f"{confidence}%"),
    ]:
        col.markdown(
            f"""
            <div style="background:{badge_bg};border:1px solid #d1fae5;border-radius:1.25rem;padding:1.25rem;min-height:130px;">
                <div style="font-size:0.65rem;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">{icon} {label}</div>
                <div style="font-size:1.6rem;font-weight:900;color:{badge_text};">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Health meter
    meter_status = "Healthy" if result["health_score"] >= 70 else "Moderate" if result["health_score"] >= 40 else "Critical"
    st.markdown("<h4 style='color:#0f3d1e;margin:1.5rem 0 0.5rem;'>📊 Soil Health Meter</h4>", unsafe_allow_html=True)
    st.progress(result["health_score"] / 100)
    st.caption(f"{result['health_score']}% — {meter_status}")

    # Technical parameters
    st.markdown(
        f"""
        <div style="background:#fff;border:1px solid #d1fae5;border-radius:1.5rem;padding:1.5rem;margin:1.5rem 0;">
            <div style="font-weight:700;color:#059669;margin-bottom:1rem;">📈 Soil Technical Parameters</div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem;font-size:0.8rem;">
                <div style="background:#f0fdf4;padding:0.75rem;border-radius:0.75rem;">
                    <div style="color:#6b7280;margin-bottom:4px;">Moisture Level</div>
                    <div style="font-weight:700;color:#0f3d1e;">{result['moisture']}</div>
                </div>
                <div style="background:#f0fdf4;padding:0.75rem;border-radius:0.75rem;">
                    <div style="color:#6b7280;margin-bottom:4px;">Compaction</div>
                    <div style="font-weight:700;color:#0f3d1e;">{result['compaction']}</div>
                </div>
                <div style="background:#f0fdf4;padding:0.75rem;border-radius:0.75rem;">
                    <div style="color:#6b7280;margin-bottom:4px;">Dryness</div>
                    <div style="font-weight:700;color:#0f3d1e;">{result['dryness']}</div>
                </div>
                <div style="background:#f0fdf4;padding:0.75rem;border-radius:0.75rem;">
                    <div style="color:#6b7280;margin-bottom:4px;">Acoustic Frequency</div>
                    <div style="font-weight:700;color:#0f3d1e;">{int(features['frequency'])} Hz</div>
                </div>
                <div style="background:#f0fdf4;padding:0.75rem;border-radius:0.75rem;">
                    <div style="color:#6b7280;margin-bottom:4px;">Signal Energy</div>
                    <div style="font-weight:700;color:#0f3d1e;">{features['energy']:.3f}</div>
                </div>
                <div style="background:#f0fdf4;padding:0.75rem;border-radius:0.75rem;">
                    <div style="color:#6b7280;margin-bottom:4px;">Scan Duration</div>
                    <div style="font-weight:700;color:#0f3d1e;">{duration} sec</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Waveform + feature graph
    st.markdown("<h4 style='color:#0f3d1e;margin:1.5rem 0 0.5rem;'>📊 Feature Graphs</h4>", unsafe_allow_html=True)
    w1, w2 = st.columns([1.4, 1])
    with w1:
        st.markdown("<div style='font-weight:600;color:#059669;margin-bottom:0.5rem;'>🌊 Soil Sound Waveform</div>", unsafe_allow_html=True)
        waveform_fig = plot_waveform(features["y"], features["sr"])
        waveform_fig.axes[0].set_xlabel("Time (s)", color="#4a6b52")
        waveform_fig.axes[0].set_ylabel("Amplitude", color="#4a6b52")
        waveform_fig.axes[0].grid(alpha=0.15, color="#059669")
        st.pyplot(waveform_fig)
    with w2:
        st.markdown("<div style='font-weight:600;color:#059669;margin-bottom:0.5rem;'>📊 Extracted Audio Features</div>", unsafe_allow_html=True)
        graph_type = st.radio("View as", ["Bar chart", "Line chart", "Radar chart"], horizontal=True, key="feature_graph_type")
        keys = ["frequency", "energy", "duration"]
        values = [features.get(k, 0) for k in keys]
        if graph_type == "Radar chart":
            feature_fig = plt.figure(figsize=(5.5, 3), facecolor="none")
            ax = feature_fig.add_subplot(111, polar=True)
            angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
            values_cyclic = values + values[:1]
            angles += angles[:1]
            ax.plot(angles, values_cyclic, color="#059669", linewidth=2)
            ax.fill(angles, values_cyclic, color="#10b981", alpha=0.25)
            ax.set_thetagrids(np.degrees(angles[:-1]), [k.capitalize() for k in keys])
            ax.set_title("Feature Radar Summary", color="#062c11", y=1.08, fontweight="bold")
            ax.grid(alpha=0.18, color="#059669")
            ax.set_facecolor("none")
            ax.tick_params(colors="#4a6b52")
        else:
            feature_fig, ax = plt.subplots(figsize=(5.5, 3), facecolor="none")
            if graph_type == "Line chart":
                ax.plot(keys, values, marker="o", color="#059669", linewidth=2)
                ax.set_title("Feature Trend Summary", color="#062c11", fontweight="bold")
                ax.set_ylabel("Value", color="#4a6b52")
                ax.grid(alpha=0.15, color="#059669")
            else:
                bars = ax.bar(keys, values, color=["#059669", "#10b981", "#34d399"])
                ax.set_title("Feature Values", color="#062c11", fontweight="bold")
                ax.set_ylabel("Value", color="#4a6b52")
                ax.grid(axis="y", alpha=0.15, color="#059669")
                for bar in bars:
                    h = bar.get_height()
                    ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, color="#062c11")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(colors="#4a6b52")
        st.pyplot(feature_fig)

    # Recommendation
    if soil_type == "Compact Soil":
        rec_bg, rec_title = "#fff1f2", "⚠ Soil Aeration Recommended"
        rec_bullets = "<li>Add organic compost</li><li>Improve drainage channels</li><li>Avoid heavy machinery on wet days</li><li>Re-sample after 3 days</li>"
    elif soil_type == "Dry Soil":
        rec_bg, rec_title = "#fffbeb", "💧 Soil Irrigation Needed"
        rec_bullets = "<li>Increase watering with drip irrigation</li><li>Add organic mulch to lock moisture</li><li>Use shade nets to reduce evaporation</li>"
    else:
        rec_bg, rec_title = "#ecfdf5", "🌱 Soil Condition Stable & Good"
        rec_bullets = "<li>Maintain current sustainable practices</li><li>Continue weekly routine sampling</li>"

    st.markdown(
        f"""
        <div style="background:{rec_bg};border:1px solid #d1fae5;border-radius:1.5rem;padding:1.5rem;margin:1.5rem 0;">
            <div style="font-weight:700;color:#0f3d1e;margin-bottom:0.75rem;">🧠 {rec_title}</div>
            <ul style="font-size:0.82rem;color:#374151;line-height:2;padding-left:1.2rem;">{rec_bullets}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Comparison table
    st.markdown("<h4 style='color:#0f3d1e;margin:1.5rem 0 0.5rem;'>📋 Current vs Ideal Parameters</h4>", unsafe_allow_html=True)
    comp_df = get_comparison_dataframe(result)
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # AI assistant
    st.markdown("<h4 style='color:#0f3d1e;margin:1.5rem 0 0.5rem;'>🤖 Ask TerraEcho AI</h4>", unsafe_allow_html=True)
    user_question = st.text_input("Ask TerraEcho AI", key="terraecho_ai", placeholder="My soil is compact. What should I do?", label_visibility="collapsed")
    if st.button("Ask TerraEcho", key="assistant_button"):
        answer = get_ai_assistant_response(user_question)
        st.success(f"**TerraEcho Agronomist:** {answer}")

    # Save & report
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    report_path = os.path.join("reports", f"soil_report_{sample_id}_{timestamp}.pdf")
    waveform_path = os.path.join("reports", f"wave_{sample_id}_{timestamp}.png")
    feature_path = os.path.join("reports", f"features_{sample_id}_{timestamp}.png")

    try:
        waveform_fig.savefig(waveform_path, bbox_inches="tight", dpi=150)
    except Exception:
        waveform_path = None
    try:
        save_feature_plot(features, feature_path)
    except Exception:
        feature_path = None

    metadata = {
        "sample_id": sample_id,
        "location": location,
        "notes": notes,
        "uploaded_at": timestamp,
        "filename": os.path.basename(report_path),
        "waveform": os.path.basename(waveform_path) if waveform_path and os.path.exists(waveform_path) else None,
        "feature_file": os.path.basename(feature_path) if feature_path and os.path.exists(feature_path) else None,
        "feature_plot": feature_path if feature_path and os.path.exists(feature_path) else None,
    }

    generate_report(result, report_path, waveform_path=waveform_path, metadata=metadata)
    save_result(result, metadata)

    with open(report_path, "rb") as file:
        st.download_button("📄 Download Soil Report PDF", file, file_name=os.path.basename(report_path))

    st.balloons()


def render_reports():
    st.markdown("<h2 style='color:#0f3d1e;'>📄 Reports & Exports</h2>", unsafe_allow_html=True)
    rows = fetch_all()
    if not rows:
        st.warning("No reports available yet. Run a soil analysis to generate your first report.")
        return

    c1, c2 = st.columns(2)
    c1.metric("Total Reports", len(rows))
    c2.metric("Most Recent Run", rows[0].get("uploaded_at", "—"))

    if st.button("📦 Prepare Report Bundle", key="prepare_zip"):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            for r in rows:
                rf = os.path.join("reports", r.get("filename", ""))
                if os.path.exists(rf):
                    zf.write(rf, arcname=os.path.basename(rf))
        buffer.seek(0)
        st.download_button(
            "📥 Download TerraEcho Report Bundle",
            data=buffer.getvalue(),
            file_name="terraecho_reports.zip",
            mime="application/zip",
        )

    st.markdown("<h3 style='color:#0f3d1e;margin-top:1.5rem;'>Available Soil Reports</h3>", unsafe_allow_html=True)
    for r in rows:
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #d1fae5;border-radius:1.25rem;padding:1.25rem;margin-bottom:1rem;">
                <div style="font-weight:700;color:#0f3d1e;">Sample #{r.get('sample_id')}</div>
                <div style="font-size:0.8rem;color:#6b7280;margin-top:4px;">📍 {r.get('location','Unknown')} &nbsp;·&nbsp; {r.get('soil_type','—')} &nbsp;·&nbsp; Health: {r.get('health_score','—')}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        rf = os.path.join("reports", r.get("filename", ""))
        if os.path.exists(rf):
            with open(rf, "rb") as fh:
                st.download_button(
                    label="📥 Download PDF",
                    data=fh,
                    file_name=r.get("filename"),
                    key=f"dl_{r.get('sample_id')}_{r.get('uploaded_at','')}",
                )


def render_ai_insights():
    st.markdown("<h2 style='color:#0f3d1e;'>🤖 AI Agronomy Insights</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#374151;font-size:0.9rem;'>Deep soil acoustic intelligence for smarter agronomy and ecological health monitoring.</p>", unsafe_allow_html=True)

    rows = fetch_all()
    if rows:
        recent = rows[:6][::-1]
        values = [r.get("health_score", 0) for r in recent]
        st.markdown("<h4 style='color:#0f3d1e;margin-top:1.5rem;'>Health Score Distribution</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 3), facecolor="none")
        colors = ["#059669" if v >= 70 else "#34d399" for v in values]
        ax.bar(range(len(values)), values, color=colors, width=0.5, edgecolor="#047857", linewidth=0.5)
        ax.set_title("Historical Analysis Health Scores", color="#062c11", fontweight="bold", fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([r.get("sample_id") for r in recent], color="#4a6b52", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#10b981")
        ax.spines["bottom"].set_color("#10b981")
        ax.grid(axis="y", alpha=0.15, color="#059669")
        ax.tick_params(colors="#4a6b52")
        ax.set_facecolor("none")
        for idx, val in enumerate(values):
            ax.annotate(f"{val}%", xy=(idx, val), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, color="#062c11", fontweight="bold")
        st.pyplot(fig)
    else:
        st.info("Collect soil samples to activate AI insight dashboards.", icon="🌿")


def render_history():
    st.markdown("<h2 style='color:#0f3d1e;'>🕒 Soil History</h2>", unsafe_allow_html=True)
    rows = fetch_all()
    if not rows:
        st.info("No soil history available yet. Complete an analysis to start tracking.", icon="🧭")
        return
    for r in rows:
        stype = r.get("soil_type", "")
        pill_color = "#ecfdf5" if stype == "Healthy Soil" else "#fffbeb" if stype == "Dry Soil" else "#fff1f2"
        pill_text = "#065f46" if stype == "Healthy Soil" else "#92400e" if stype == "Dry Soil" else "#9f1239"
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #d1fae5;border-radius:1.25rem;padding:1.25rem;margin-bottom:1rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
                    <div style="font-weight:700;color:#0f3d1e;">Sample #{r.get('sample_id')}</div>
                    <span style="background:{pill_color};color:{pill_text};font-size:0.72rem;font-weight:700;padding:2px 10px;border-radius:9999px;">{stype}</span>
                </div>
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.5rem;font-size:0.78rem;">
                    <div><span style="color:#6b7280;">Health</span><br><strong>{r.get('health_score')}%</strong></div>
                    <div><span style="color:#6b7280;">Moisture</span><br><strong>{r.get('moisture')}</strong></div>
                    <div><span style="color:#6b7280;">Compaction</span><br><strong>{r.get('compaction')}</strong></div>
                    <div><span style="color:#6b7280;">Location</span><br><strong>{r.get('location','-')}</strong></div>
                </div>
                <div style="font-size:0.72rem;color:#9ca3af;margin-top:0.5rem;">📅 {r.get('uploaded_at')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_about():
    st.markdown("<h2 style='color:#0f3d1e;'>ℹ About TerraEcho</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:1.5rem;">
            <div style="background:#fff;border:1px solid #d1fae5;border-radius:1.25rem;padding:1.5rem;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">🎯</div>
                <h3 style="color:#0f3d1e;font-size:1rem;margin-bottom:0.5rem;">Mission</h3>
                <p style="font-size:0.82rem;color:#374151;line-height:1.7;">Empower sustainable agriculture with accurate, easy-to-use soil health intelligence using acoustic resonance.</p>
            </div>
            <div style="background:#fff;border:1px solid #d1fae5;border-radius:1.25rem;padding:1.5rem;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">🔮</div>
                <h3 style="color:#0f3d1e;font-size:1rem;margin-bottom:0.5rem;">Vision</h3>
                <p style="font-size:0.82rem;color:#374151;line-height:1.7;">Transform soil scanning into fast, confident decisions for every field to enhance global crop security.</p>
            </div>
            <div style="background:#fff;border:1px solid #d1fae5;border-radius:1.25rem;padding:1.5rem;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">🌱</div>
                <h3 style="color:#0f3d1e;font-size:1rem;margin-bottom:0.5rem;">Values</h3>
                <p style="font-size:0.82rem;color:#374151;line-height:1.7;">Precision, environmental sustainability, technological accessibility, and measurable climate impact.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Router ───────────────────────────────────────────────────────────────────

if mode == "🏠 Dashboard":
    render_dashboard()
elif mode == "🌱 Soil Analysis":
    render_analysis()
elif mode == "📄 Reports":
    render_reports()
elif mode == "🤖 AI Insights":
    render_ai_insights()
elif mode == "🕒 History":
    render_history()
elif mode == "ℹ About":
    render_about()

st.markdown("<hr style='border-color:#d1fae5;margin-top:3rem;'>", unsafe_allow_html=True)
st.caption("TerraEcho | AI-Powered Soil Intelligence for Farmers 🌱")
