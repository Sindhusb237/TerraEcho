import streamlit as st
import tempfile
import os
import uuid
import io
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
st.set_page_config(page_title="TerraEcho | AgriTech IQ", page_icon="🌿", layout="wide")

# ─── Database init ──────────────────────────────────────────────────────────
init_db()

# ─── CSS ────────────────────────────────────────────────────────────────────
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
                <div style="font-size:1.1rem;font-weight:800;color:#34d399;">TerraEcho</div>
                <div style="font-size:0.7rem;color:#6b7280;margin-top:2px;">Gen5 Soil Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    languages = ["English", "Kannada", "Hindi"]
    lang = st.selectbox("🌐 Language", languages, key="global_lang")

    mode = st.radio("Navigation", NAV_OPTIONS, index=1, key="nav_mode")

    st.markdown("<hr style='border-color:rgba(52,211,153,0.15);margin:12px 0;'>", unsafe_allow_html=True)

    with st.form(key="meta_form"):
        st.markdown(
            "<div style='font-size:0.7rem;font-weight:700;color:#34d399;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>Sample Metadata</div>",
            unsafe_allow_html=True,
        )
        sample_id = st.text_input("Sample ID", value=str(uuid.uuid4())[:8])
        location = st.text_input("Location")
        notes = st.text_area("Notes", height=80)
        submitted = st.form_submit_button("💾 Save Metadata")

    st.markdown("<hr style='border-color:rgba(52,211,153,0.15);margin:12px 0;'>", unsafe_allow_html=True)

    if st.button("🔧 Run Diagnostics"):
        with st.expander("Test Output", expanded=True):
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q"],
                    capture_output=True, text=True, timeout=120,
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
        "<div style='font-size:0.65rem;color:#6b7280;text-align:center;margin-top:16px;'>TerraEcho • Sustainable agriculture</div>",
        unsafe_allow_html=True,
    )


# ─── Helper functions ─────────────────────────────────────────────────────────

def build_soil_profile(result):
    confidence = min(100, max(65, int(result["health_score"] * 0.38 + 25)))
    crop_map = {"Dry Soil": "Millet, Sorghum, Chickpea", "Healthy Soil": "Wheat, Maize, Soybeans", "Compact Soil": "Barley, Beans, Radish"}
    water_map = {"Dry Soil": "High", "Healthy Soil": "Moderate", "Compact Soil": "Low"}
    treatment_map = {"Dry Soil": "Irrigate & mulch.", "Healthy Soil": "Maintain organic cover.", "Compact Soil": "Aerate and add compost."}
    return {
        "confidence": confidence,
        "suggested_crops": crop_map.get(result["soil_type"], "Adaptive grasses"),
        "water": water_map.get(result["soil_type"], "Moderate"),
        "treatment": treatment_map.get(result["soil_type"], "Optimize organic matter."),
        "sustainability": min(100, result["health_score"] + 10),
    }


def get_ai_confidence(result, features):
    base_score = result.get("health_score", 50)
    energy = min(1.0, features.get("energy", 0) * 7)
    freq_ratio = min(1.0, features.get("frequency", 0) / 5000)
    return int(min(100, max(65, base_score * 0.65 + energy * 18 + freq_ratio * 17)))


def get_comparison_dataframe(result):
    return pd.DataFrame({
        "Parameter": ["Moisture", "Compaction", "Health Score"],
        "Current": [result.get("moisture", "-"), result.get("compaction", "-"), f"{result.get('health_score', 0)}%"],
        "Ideal": ["Moderate", "Low", ">85%"],
    })


def get_ai_assistant_response(question):
    p = question.lower()
    if "compact" in p or "aerate" in p or "drainage" in p:
        return "Loosen the soil and add compost. Reduce heavy equipment and improve drainage."
    if "dry" in p or "irrigation" in p or "moisture" in p:
        return "Increase irrigation, add mulch, and retain moisture with cover crops."
    if "healthy" in p or "good" in p or "maintain" in p:
        return "Soil is good. Keep monitoring and maintain current sustainable practices."
    return "TerraEcho suggests field sampling and routine soil monitoring."


# ─── Reusable card builder (dark theme) ───────────────────────────────────────

def glass_card(content, extra_style=""):
    """Render HTML inside a dark glassmorphic card."""
    st.markdown(
        f'<div class="glass-panel" style="{extra_style}">{content}</div>',
        unsafe_allow_html=True,
    )


def stat_card(value, label):
    """Metric card for dashboard."""
    st.markdown(
        f"""
        <div class="glass-panel" style="text-align:center;min-height:120px;">
            <div style="font-size:2rem;font-weight:900;color:#34d399;">{value}</div>
            <div style="font-size:0.7rem;font-weight:600;color:#9ca3af;text-transform:uppercase;letter-spacing:0.1em;margin-top:6px;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Page renderers ───────────────────────────────────────────────────────────

def render_hero():
    st.markdown(
        f"""
        <div class="hero-panel" style="display:block;">
            <span style="font-size:0.65rem;font-weight:700;color:#34d399;text-transform:uppercase;letter-spacing:0.15em;">{t("AI-Based Soil Sound Intelligence System", lang)}</span>
            <h1 style="font-size:2.8rem;font-weight:900;color:#ffffff;margin:0.5rem 0 1rem 0;line-height:1.1;">🌱 TerraEcho</h1>
            <p style="color:#d1d5db;font-size:0.95rem;max-width:600px;line-height:1.7;margin-bottom:1.5rem;">
                {t("A professional agriculture dashboard for soil sound analytics, AI-driven health scores, and smart farming guidance.", lang)}
            </p>
            <div style="display:flex;gap:12px;flex-wrap:wrap;">
                <span class="hero-button">{t("Analyze Soil", lang)}</span>
                <span class="hero-button ghost">{t("Reports", lang)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard():
    rows = fetch_all()
    total_samples = len(rows)
    avg_health = int(sum(r.get("health_score", 0) for r in rows) / total_samples) if total_samples else 0
    recent_soil = rows[0]["soil_type"] if rows else "N/A"
    health_bucket = "Excellent" if avg_health >= 80 else "Stable" if avg_health >= 60 else "Attention"

    render_hero()

    st.markdown(f"<h2>🏠 {t('Enterprise Soil Overview', lang)}</h2>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card(total_samples, t("Processed Samples", lang))
    with c2:
        stat_card(f"{avg_health}%", t("Average Health Score", lang))
    with c3:
        stat_card(recent_soil, t("Recent Soil Type", lang))
    with c4:
        stat_card(health_bucket, t("Farming Readiness", lang))

    st.markdown(f"<h3>📈 {t('Trend Monitoring', lang)}</h3>", unsafe_allow_html=True)
    if rows:
        health_series = [r.get("health_score", 0) for r in rows[:8]][::-1]
        dates = [r.get("uploaded_at", "")[-6:] for r in rows[:8]][::-1]
        fig, ax = plt.subplots(figsize=(10, 3), facecolor="none")
        ax.plot(dates, health_series, color="#34d399", marker="o", linewidth=2)
        ax.fill_between(dates, health_series, color="#10b981", alpha=0.15)
        ax.set_title(t("Soil Health Trend", lang), color="#e5e7eb", fontsize=11, fontweight="bold")
        ax.set_facecolor("none")
        ax.grid(alpha=0.15, color="#10b981")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors="#9ca3af")
        st.pyplot(fig)
    else:
        st.info(t("Analyze your first sample to populate enterprise dashboards.", lang))


def render_analysis():
    st.markdown(f"<h2>🌱 {t('Soil Analysis', lang)}</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:#d1d5db;font-size:0.9rem;margin-bottom:1.5rem;'>{t('Upload soil tapping audio to analyze soil health with pro-grade AI intelligence.', lang)}</p>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([2.8, 1.2])
    with left:
        st.markdown(
            f"""
            <div class="upload-card">
                <div style="font-size:2.5rem;margin-bottom:0.5rem;text-align:center;">🎧</div>
                <div class="upload-title" style="text-align:center;">{t("Upload soil tapping audio to analyze soil health", lang)}</div>
                <div class="upload-caption" style="text-align:center;margin-top:0.5rem;">{t("Drop your .wav or .mp3 sample here for instant soil type classification, moisture analysis, and AI recommendation.", lang)}</div>
                <div class="upload-drop" style="text-align:center;margin-top:1rem;">
                    <div style="font-size:1.1rem;font-weight:600;color:#e5e7eb;">{t("Drag & drop your audio file", lang)}</div>
                    <div style="font-size:0.82rem;color:#9ca3af;margin-top:4px;">{t("Accepted formats: WAV, MP3 • Less than 20MB", lang)}</div>
                </div>
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
        glass_card(f"""
            <div style="font-weight:700;color:#e5e7eb;margin-bottom:0.75rem;">{t("Premium Analysis Kit", lang)}</div>
            <div style="font-size:0.82rem;color:#9ca3af;margin-bottom:1rem;">{t("Designed for farmers, research teams, and agri-tech judges.", lang)}</div>
            <ul style="font-size:0.82rem;color:#d1d5db;line-height:2.2;padding-left:1.2rem;">
                <li>{t("Modern soil dashboard", lang)}</li>
                <li>{t("AI-driven recommendations", lang)}</li>
                <li>{t("Rich visual storytelling", lang)}</li>
            </ul>
        """)

    if not (audio_file and audio_bytes is not None):
        return

    with st.spinner(f"🔍 TerraEcho AI {t('Soil Analysis', lang)}..."):
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

    # Audio details card
    glass_card(f"""
        <div style="font-weight:700;color:#34d399;margin-bottom:0.75rem;">🎵 {t("Uploaded Audio Details", lang)}</div>
        <div style="font-size:0.85rem;font-weight:600;color:#e5e7eb;margin-bottom:1rem;">{audio_file.name}</div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem;">
            <div style="background:rgba(16,185,129,0.08);padding:0.75rem;border-radius:0.75rem;border:1px solid rgba(52,211,153,0.15);">
                <div style="color:#9ca3af;font-size:0.72rem;margin-bottom:4px;">{t("Format", lang)}</div>
                <div style="font-weight:700;color:#e5e7eb;font-size:0.85rem;">{audio_file.type or 'Audio'}</div>
            </div>
            <div style="background:rgba(16,185,129,0.08);padding:0.75rem;border-radius:0.75rem;border:1px solid rgba(52,211,153,0.15);">
                <div style="color:#9ca3af;font-size:0.72rem;margin-bottom:4px;">{t("Size", lang)}</div>
                <div style="font-weight:700;color:#e5e7eb;font-size:0.85rem;">{round(audio_file.size / 1024, 1)} KB</div>
            </div>
            <div style="background:rgba(16,185,129,0.08);padding:0.75rem;border-radius:0.75rem;border:1px solid rgba(52,211,153,0.15);">
                <div style="color:#9ca3af;font-size:0.72rem;margin-bottom:4px;">{t("Duration", lang)}</div>
                <div style="font-weight:700;color:#e5e7eb;font-size:0.85rem;">{duration} {t("sec", lang)}</div>
            </div>
        </div>
    """)

    # Soil type badge colors for dark theme
    soil_type = result["soil_type"]
    if soil_type == "Healthy Soil":
        badge_bg, badge_border = "rgba(16,185,129,0.12)", "rgba(52,211,153,0.3)"
        badge_text = "#34d399"
    elif soil_type == "Dry Soil":
        badge_bg, badge_border = "rgba(245,158,11,0.12)", "rgba(251,191,36,0.3)"
        badge_text = "#fbbf24"
    else:
        badge_bg, badge_border = "rgba(239,68,68,0.12)", "rgba(248,113,113,0.3)"
        badge_text = "#f87171"

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("🟫", t("Soil Type", lang), t(result["soil_type"], lang)),
        ("💧", t("Moisture", lang), t(result["moisture"], lang)),
        ("🌱", t("Health Score", lang), f"{result['health_score']}%"),
        ("🤖", t("AI Confidence", lang), f"{confidence}%"),
    ]
    for col, (icon, label, value) in zip([c1, c2, c3, c4], metrics):
        col.markdown(
            f"""
            <div style="background:{badge_bg};border:1px solid {badge_border};border-radius:1.25rem;padding:1.25rem;min-height:130px;">
                <div style="font-size:0.65rem;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">{icon} {label}</div>
                <div style="font-size:1.5rem;font-weight:900;color:{badge_text};">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Health meter
    meter_status = "Healthy" if result["health_score"] >= 70 else "Moderate" if result["health_score"] >= 40 else "Critical"
    st.markdown(f"<h4>📊 {t('Soil Health Meter', lang)}</h4>", unsafe_allow_html=True)
    st.progress(result["health_score"] / 100)
    st.caption(f"{result['health_score']}% — {meter_status}")

    # Technical parameters
    glass_card(f"""
        <div style="font-weight:700;color:#34d399;margin-bottom:1rem;">📈 {t("Soil Technical Parameters", lang)}</div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem;font-size:0.8rem;">
            <div style="background:rgba(16,185,129,0.06);padding:0.75rem;border-radius:0.75rem;border:1px solid rgba(52,211,153,0.1);">
                <div style="color:#9ca3af;margin-bottom:4px;">{t("Moisture Level", lang)}</div>
                <div style="font-weight:700;color:#e5e7eb;">{t(result['moisture'], lang)}</div>
            </div>
            <div style="background:rgba(16,185,129,0.06);padding:0.75rem;border-radius:0.75rem;border:1px solid rgba(52,211,153,0.1);">
                <div style="color:#9ca3af;margin-bottom:4px;">{t("Compaction", lang)}</div>
                <div style="font-weight:700;color:#e5e7eb;">{result['compaction']}</div>
            </div>
            <div style="background:rgba(16,185,129,0.06);padding:0.75rem;border-radius:0.75rem;border:1px solid rgba(52,211,153,0.1);">
                <div style="color:#9ca3af;margin-bottom:4px;">{t("Dryness", lang)}</div>
                <div style="font-weight:700;color:#e5e7eb;">{result['dryness']}</div>
            </div>
            <div style="background:rgba(16,185,129,0.06);padding:0.75rem;border-radius:0.75rem;border:1px solid rgba(52,211,153,0.1);">
                <div style="color:#9ca3af;margin-bottom:4px;">{t("Acoustic Frequency", lang)}</div>
                <div style="font-weight:700;color:#e5e7eb;">{int(features['frequency'])} Hz</div>
            </div>
            <div style="background:rgba(16,185,129,0.06);padding:0.75rem;border-radius:0.75rem;border:1px solid rgba(52,211,153,0.1);">
                <div style="color:#9ca3af;margin-bottom:4px;">{t("Signal Energy", lang)}</div>
                <div style="font-weight:700;color:#e5e7eb;">{features['energy']:.3f}</div>
            </div>
            <div style="background:rgba(16,185,129,0.06);padding:0.75rem;border-radius:0.75rem;border:1px solid rgba(52,211,153,0.1);">
                <div style="color:#9ca3af;margin-bottom:4px;">{t("Scan Duration", lang)}</div>
                <div style="font-weight:700;color:#e5e7eb;">{duration} {t("sec", lang)}</div>
            </div>
        </div>
    """)

    # Waveform + feature graph
    st.markdown(f"<h4>📊 {t('Feature Graphs', lang)}</h4>", unsafe_allow_html=True)
    w1, w2 = st.columns([1.4, 1])
    with w1:
        st.markdown(f"<div style='font-weight:600;color:#34d399;margin-bottom:0.5rem;'>🌊 {t('Soil Sound Waveform', lang)}</div>", unsafe_allow_html=True)
        waveform_fig = plot_waveform(features["y"], features["sr"])
        waveform_fig.axes[0].set_xlabel("Time (s)", color="#9ca3af")
        waveform_fig.axes[0].set_ylabel("Amplitude", color="#9ca3af")
        waveform_fig.axes[0].grid(alpha=0.15, color="#10b981")
        st.pyplot(waveform_fig)
    with w2:
        st.markdown(f"<div style='font-weight:600;color:#34d399;margin-bottom:0.5rem;'>📊 {t('Extracted Audio Features', lang)}</div>", unsafe_allow_html=True)
        graph_type = st.radio(t("View as", lang), [t("Bar chart", lang), t("Line chart", lang), t("Radar chart", lang)], horizontal=True, key="feature_graph_type")
        keys = ["frequency", "energy", "duration"]
        values = [features.get(k, 0) for k in keys]
        if graph_type == t("Radar chart", lang):
            feature_fig = plt.figure(figsize=(5.5, 3), facecolor="none")
            ax = feature_fig.add_subplot(111, polar=True)
            angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
            values_cyclic = values + values[:1]
            angles += angles[:1]
            ax.plot(angles, values_cyclic, color="#34d399", linewidth=2)
            ax.fill(angles, values_cyclic, color="#10b981", alpha=0.25)
            ax.set_thetagrids(np.degrees(angles[:-1]), [k.capitalize() for k in keys])
            ax.set_title(t("Feature Radar Summary", lang), color="#e5e7eb", y=1.08, fontweight="bold")
            ax.grid(alpha=0.18, color="#10b981")
            ax.set_facecolor("none")
            ax.tick_params(colors="#9ca3af")
        else:
            feature_fig, ax = plt.subplots(figsize=(5.5, 3), facecolor="none")
            if graph_type == t("Line chart", lang):
                ax.plot(keys, values, marker="o", color="#34d399", linewidth=2)
                ax.set_title(t("Feature Trend Summary", lang), color="#e5e7eb", fontweight="bold")
                ax.set_ylabel(t("Value", lang), color="#9ca3af")
                ax.grid(alpha=0.15, color="#10b981")
            else:
                bars = ax.bar(keys, values, color=["#059669", "#10b981", "#34d399"])
                ax.set_title(t("Feature Values", lang), color="#e5e7eb", fontweight="bold")
                ax.set_ylabel(t("Value", lang), color="#9ca3af")
                ax.grid(axis="y", alpha=0.15, color="#10b981")
                for bar in bars:
                    h = bar.get_height()
                    ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, color="#e5e7eb")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(colors="#9ca3af")
        st.pyplot(feature_fig)

    # Recommendation
    if soil_type == "Compact Soil":
        rec_title = t("⚠ Soil Aeration Recommended", lang)
        rec_text = t("recs_compact", lang)
        rec_color = "#f87171"
    elif soil_type == "Dry Soil":
        rec_title = t("💧 Soil Irrigation Needed", lang)
        rec_text = t("recs_dry", lang)
        rec_color = "#fbbf24"
    else:
        rec_title = t("✅ Soil Condition Stable & Good", lang)
        rec_text = t("recs_healthy", lang)
        rec_color = "#34d399"

    rec_bullets = "".join(f"<li style='margin-bottom:4px;'>{line.lstrip('- ')}</li>" for line in rec_text.split("\n") if line.strip())
    glass_card(f"""
        <div style="font-weight:700;color:{rec_color};margin-bottom:0.75rem;">🧠 {rec_title}</div>
        <ul style="font-size:0.82rem;color:#d1d5db;line-height:1.8;padding-left:1.2rem;">{rec_bullets}</ul>
    """)

    # Comparison table
    st.markdown(f"<h4>📋 {t('Current vs Ideal Parameters', lang)}</h4>", unsafe_allow_html=True)
    comp_df = get_comparison_dataframe(result)
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # AI assistant
    st.markdown(f"<h4>🤖 {t('Ask TerraEcho AI', lang)}</h4>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#9ca3af;font-size:0.82rem;'>{t('Chat with our AI agronomist assistant about soil aeration, crop cycles, or irrigation.', lang)}</p>", unsafe_allow_html=True)
    user_question = st.text_input(t("Ask TerraEcho AI", lang), key="terraecho_ai", placeholder=t("My soil is compact. What should I do?", lang), label_visibility="collapsed")
    if st.button(t("Ask TerraEcho", lang), key="assistant_button"):
        answer = get_ai_assistant_response(user_question)
        st.success(f"**{t('TerraEcho Agronomist', lang)}:** {answer}")

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
        st.download_button(f"📄 {t('Download Soil Report PDF', lang)}", file, file_name=os.path.basename(report_path))
    st.balloons()


def render_reports():
    st.markdown(f"<h2>📄 {t('Reports & Exports', lang)}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#d1d5db;font-size:0.9rem;'>{t('Download investor-grade soil health reports and export your portfolio.', lang)}</p>", unsafe_allow_html=True)
    rows = fetch_all()
    if not rows:
        st.warning(t("Analyze your first sample to populate enterprise dashboards.", lang))
        return

    c1, c2 = st.columns(2)
    with c1:
        stat_card(len(rows), t("Total Reports", lang))
    with c2:
        stat_card(rows[0].get("uploaded_at", "—"), t("Recent Run", lang))

    if st.button(f"📦 {t('Prepare Report Bundle', lang)}", key="prepare_zip"):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            for r in rows:
                rf = os.path.join("reports", r.get("filename", ""))
                if os.path.exists(rf):
                    zf.write(rf, arcname=os.path.basename(rf))
        buffer.seek(0)
        st.download_button(f"📥 {t('Download TerraEcho Report Bundle', lang)}", data=buffer.getvalue(), file_name="terraecho_reports.zip", mime="application/zip")

    st.markdown(f"<h3>{t('Available Soil Reports', lang)}</h3>", unsafe_allow_html=True)
    for r in rows:
        stype = r.get("soil_type", "")
        glass_card(f"""
            <div style="font-weight:700;color:#e5e7eb;">{t("Sample", lang)} #{r.get('sample_id')}</div>
            <div style="font-size:0.8rem;color:#9ca3af;margin-top:4px;">📍 {r.get('location', t('Unknown Location', lang))} · {t(stype, lang)} · {t('Health Score', lang)}: {r.get('health_score','—')}%</div>
        """, extra_style="margin-bottom:12px;")
        rf = os.path.join("reports", r.get("filename", ""))
        if os.path.exists(rf):
            with open(rf, "rb") as fh:
                st.download_button(
                    label=f"📥 {t('Download PDF', lang)}",
                    data=fh,
                    file_name=r.get("filename"),
                    key=f"dl_{r.get('sample_id')}_{r.get('uploaded_at','')}",
                )


def render_ai_insights():
    st.markdown(f"<h2>🤖 {t('AI Agronomy Insights', lang)}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#d1d5db;font-size:0.9rem;'>{t('Deep soil acoustic intelligence for smarter agronomy and ecological health monitoring.', lang)}</p>", unsafe_allow_html=True)

    cols = st.columns(3)
    cards_data = [
        ("🔊", t("Frequency Intelligence", lang), t("Understand how sound spectral patterns map to soil moisture and compaction level variations.", lang)),
        ("🌾", t("Crop Reflection", lang), t("Match agro-recommendations to the detected soil health profile for maximum crop yield potential.", lang)),
        ("🍃", t("Sustainability Index", lang), t("Track regenerative farming signals through acoustic soil analysis and organic matter trends.", lang)),
    ]
    for col, (emoji, title, desc) in zip(cols, cards_data):
        with col:
            glass_card(f"""
                <div style="font-size:2rem;margin-bottom:0.5rem;">{emoji}</div>
                <strong style="color:#34d399;font-weight:700;">{title}</strong>
                <p style="font-size:0.78rem;color:#d1d5db;margin-top:0.5rem;line-height:1.6;">{desc}</p>
            """)

    rows = fetch_all()
    if rows:
        recent = rows[:6][::-1]
        values = [r.get("health_score", 0) for r in recent]
        st.markdown(f"<h4>{t('Health Score Distribution', lang)}</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 3), facecolor="none")
        colors = ["#10b981" if v >= 70 else "#34d399" for v in values]
        ax.bar(range(len(values)), values, color=colors, width=0.5, edgecolor="#047857", linewidth=0.5)
        ax.set_title(t("Historical Analysis Health Scores", lang), color="#e5e7eb", fontweight="bold", fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([r.get("sample_id") for r in recent], color="#9ca3af", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#10b981")
        ax.spines["bottom"].set_color("#10b981")
        ax.grid(axis="y", alpha=0.15, color="#10b981")
        ax.tick_params(colors="#9ca3af")
        ax.set_facecolor("none")
        for idx, val in enumerate(values):
            ax.annotate(f"{val}%", xy=(idx, val), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, color="#e5e7eb", fontweight="bold")
        st.pyplot(fig)
    else:
        st.info(t("Collect soil samples to activate AI insight dashboards.", lang), icon="🌿")


def render_history():
    st.markdown(f"<h2>🕒 {t('Soil History', lang)}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#d1d5db;font-size:0.9rem;'>{t('Review and compare past soil analyses performed in your workspace.', lang)}</p>", unsafe_allow_html=True)
    rows = fetch_all()
    if not rows:
        st.info(t("Analyze your first sample to populate enterprise dashboards.", lang), icon="🧭")
        return
    for r in rows:
        stype = r.get("soil_type", "")
        if stype == "Healthy Soil":
            pill_bg, pill_color = "rgba(16,185,129,0.15)", "#34d399"
        elif stype == "Dry Soil":
            pill_bg, pill_color = "rgba(245,158,11,0.15)", "#fbbf24"
        else:
            pill_bg, pill_color = "rgba(239,68,68,0.15)", "#f87171"

        glass_card(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
                <div style="font-weight:700;color:#e5e7eb;">{t("Sample", lang)} #{r.get('sample_id')}</div>
                <span style="background:{pill_bg};color:{pill_color};font-size:0.72rem;font-weight:700;padding:4px 12px;border-radius:9999px;">{t(stype, lang)}</span>
            </div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.5rem;font-size:0.78rem;">
                <div><span style="color:#9ca3af;">{t("Health Score", lang)}</span><br><strong style="color:#e5e7eb;">{r.get('health_score')}%</strong></div>
                <div><span style="color:#9ca3af;">{t("Moisture", lang)}</span><br><strong style="color:#e5e7eb;">{t(str(r.get('moisture','')), lang)}</strong></div>
                <div><span style="color:#9ca3af;">{t("Compaction", lang)}</span><br><strong style="color:#e5e7eb;">{r.get('compaction')}</strong></div>
                <div><span style="color:#9ca3af;">📍</span><br><strong style="color:#e5e7eb;">{r.get('location','-')}</strong></div>
            </div>
            <div style="font-size:0.72rem;color:#6b7280;margin-top:0.5rem;">📅 {r.get('uploaded_at')}</div>
        """, extra_style="margin-bottom:12px;")


def render_about():
    st.markdown(f"<h2>ℹ {t('About TerraEcho', lang)}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#d1d5db;font-size:0.9rem;'>{t('Built for farmers, researchers, NGOs, and agri-tech innovators focused on resilient soil systems.', lang)}</p>", unsafe_allow_html=True)

    cols = st.columns(3)
    about_data = [
        ("🎯", t("Mission", lang), t("Empower sustainable agriculture with accurate, easy-to-use soil health intelligence using acoustic resonance.", lang)),
        ("🔮", t("Vision", lang), t("Transform soil scanning into fast, confident decisions for every field to enhance global crop security.", lang)),
        ("🌱", t("Values", lang), t("Precision, environmental sustainability, technological accessibility, and measurable climate impact.", lang)),
    ]
    for col, (emoji, title, desc) in zip(cols, about_data):
        with col:
            glass_card(f"""
                <div style="font-size:2rem;margin-bottom:0.5rem;">{emoji}</div>
                <h3 style="color:#34d399;font-size:1rem;margin-bottom:0.5rem;">{title}</h3>
                <p style="font-size:0.82rem;color:#d1d5db;line-height:1.7;">{desc}</p>
            """)

    cols2 = st.columns(3)
    contact_data = [
        (t("Get in Touch", lang), t("Contact", lang), "hello@terraecho.ai"),
        (t("Open Source", lang), t("GitHub Repository", lang), "github.com/Sindhusb237/TerraEcho"),
        (t("Eco Impact", lang), t("Sustainability", lang), t("Supporting soil resilience, carbon sequestration, and climate-smart food networks.", lang)),
    ]
    for col, (cat, title, desc) in zip(cols2, contact_data):
        with col:
            glass_card(f"""
                <div style="font-size:0.65rem;color:#9ca3af;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">{cat}</div>
                <strong style="color:#e5e7eb;font-size:0.9rem;">{title}</strong>
                <p style="font-size:0.78rem;color:#d1d5db;margin-top:0.3rem;">{desc}</p>
            """)


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

st.markdown("<hr style='border-color:rgba(52,211,153,0.1);margin-top:3rem;'>", unsafe_allow_html=True)
st.caption("TerraEcho | AI-Powered Soil Intelligence for Farmers 🌱")
