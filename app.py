import streamlit as st
import tempfile
import os
import uuid
from datetime import datetime
import subprocess
import sys
import io
import base64
import csv
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from audio_processing import process_audio
from prediction_engine import predict_soil
from visualization import plot_waveform, save_feature_plot
from report_generator import generate_report
from database import init_db, save_result, fetch_all

st.set_page_config(page_title="TerraEcho", page_icon="🌿", layout="wide")

# Initialize local database
init_db()

# Inject custom dashboard styling
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.markdown(
        """
        <style>
        .stApp { font-family: 'Inter', sans-serif; background: #0b170b; color: #eef4e8; }
        </style>
        """,
        unsafe_allow_html=True,
    )

NAV_OPTIONS = [
    "🏠 Dashboard",
    "🌱 Soil Analysis",
    "📄 Reports",
    "🤖 AI Insights",
    "🕒 History",
    "ℹ️ About",
]

mode = st.sidebar.radio("", NAV_OPTIONS, index=1)

with st.sidebar.container():
    st.markdown(
        """
        <div class="flex items-center gap-3 p-4 mb-4 rounded-2xl bg-white/95 border border-emerald-500/10 shadow-lg text-emerald-950">
            <div class="w-12 h-12 rounded-xl flex items-center justify-center text-2xl bg-gradient-to-br from-emerald-600 to-teal-600 text-white font-bold">🌿</div>
            <div>
                <div class="font-bold text-lg text-emerald-900 leading-none">TerraEcho</div>
                <div class="text-xs text-emerald-600/70 mt-1">Smart Soil Intelligence</div>
            </div>
        </div>
        <div class="mt-4 p-4 rounded-2xl bg-emerald-950/40 border border-emerald-500/10 text-emerald-100">
            <div class="text-xs font-bold text-emerald-300 mb-1 uppercase tracking-wide">Modern Agri Navigation</div>
            <div class="text-xs text-emerald-200/80 leading-relaxed">Fast access to soil analysis, reports, AI insights, and historic samples.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar.expander("Upload & Data", expanded=True):
    st.write("Drop soil tapping audio or record raw samples.")
    st.write("Accepted: .wav, .mp3")

st.sidebar.markdown("<div class='my-4 border-t border-emerald-500/10'></div>", unsafe_allow_html=True)

if st.sidebar.button("Run Diagnostics"):
    with st.sidebar.expander("Test Output"):
        try:
            proc = subprocess.run([sys.executable, '-m', 'pytest', '-q'], capture_output=True, text=True, timeout=300)
            st.code(proc.stdout or "No output")
            if proc.stderr:
                st.code(proc.stderr)
            st.success("Diagnostics complete")
        except subprocess.TimeoutExpired:
            st.error("Diagnostics timed out")
        except Exception as exc:
            st.error(f"Diagnostics error: {exc}")

st.sidebar.markdown("<div class='mt-6 mx-4 text-xs leading-relaxed text-emerald-300/40 text-center border-t border-emerald-500/10 pt-4'>TerraEcho &bull; Sustainable agriculture for farms, researchers, and agri-tech innovators.</div>", unsafe_allow_html=True)

with st.sidebar.form(key="meta_form"):
    st.markdown("<div class='text-xs font-bold mb-2 text-emerald-300 uppercase tracking-wider'>Sample Metadata</div>", unsafe_allow_html=True)
    sample_id = st.text_input("Sample ID", value=str(uuid.uuid4())[:8])
    location = st.text_input("Location")
    notes = st.text_area("Notes")
    submitted = st.form_submit_button("Save Metadata")


def render_hero():
    st.markdown(
        """
        <section class="grid grid-cols-1 lg:grid-cols-2 gap-8 p-8 md:p-10 my-6 mx-6 rounded-3xl bg-white text-slate-950 shadow-2xl border border-slate-200 relative overflow-hidden">
            <div class="absolute -top-16 -left-16 w-48 h-48 rounded-full bg-emerald-500/10 blur-3xl pointer-events-none"></div>
            <div class="absolute -bottom-16 -right-16 w-48 h-48 rounded-full bg-teal-500/10 blur-3xl pointer-events-none"></div>
            
            <div class="flex flex-col justify-center">
                <div class="text-xs uppercase tracking-widest text-emerald-600 font-bold mb-3">AI-Based Soil Sound Intelligence System</div>
                <h1 class="text-4xl md:text-5xl font-extrabold leading-tight tracking-tight mb-4 text-slate-950">
                    🌱 TerraEcho
                </h1>
                <p class="text-slate-600 text-sm md:text-base leading-relaxed mb-8 max-w-xl">
                    A professional agriculture dashboard for soil sound analytics, AI-driven health scores, and smart farming guidance.
                </p>
                <div class="flex flex-wrap gap-4">
                    <a href="#soil-analysis" class="inline-flex items-center justify-center px-6 py-3 rounded-full bg-gradient-to-r from-emerald-600 to-emerald-500 hover:from-emerald-500 hover:to-emerald-400 text-white font-semibold shadow-lg hover:shadow-emerald-500/20 transform hover:-translate-y-0.5 transition-all duration-300">
                        Analyze Soil
                    </a>
                    <a href="#reports" class="inline-flex items-center justify-center px-6 py-3 rounded-full bg-emerald-800/40 hover:bg-emerald-800/60 border border-emerald-500/30 text-emerald-200 hover:text-white font-semibold transition-all duration-300">
                        Download Report
                    </a>
                </div>
            </div>
            
            <div class="grid gap-6">
                <div class="p-6 rounded-2xl bg-emerald-900/40 border border-emerald-500/10 backdrop-blur-md">
                    <div class="text-xs font-bold text-emerald-400 uppercase tracking-wide mb-3">Soil Waveform Preview</div>
                    <div class="flex items-end justify-between h-28 gap-2 px-4 py-2">
                        <span class="block w-full bg-gradient-to-t from-emerald-600 to-teal-400 rounded-full animate-pulse" style="height: 48%; animation-delay: 0.1s;"></span>
                        <span class="block w-full bg-gradient-to-t from-emerald-600 to-teal-400 rounded-full animate-pulse" style="height: 78%; animation-delay: 0.2s;"></span>
                        <span class="block w-full bg-gradient-to-t from-emerald-600 to-teal-400 rounded-full animate-pulse" style="height: 92%; animation-delay: 0.3s;"></span>
                        <span class="block w-full bg-gradient-to-t from-emerald-600 to-teal-400 rounded-full animate-pulse" style="height: 68%; animation-delay: 0.4s;"></span>
                        <span class="block w-full bg-gradient-to-t from-emerald-600 to-teal-400 rounded-full animate-pulse" style="height: 100%; animation-delay: 0.5s;"></span>
                        <span class="block w-full bg-gradient-to-t from-emerald-600 to-teal-400 rounded-full animate-pulse" style="height: 84%; animation-delay: 0.6s;"></span>
                        <span class="block w-full bg-gradient-to-t from-emerald-600 to-teal-400 rounded-full animate-pulse" style="height: 72%; animation-delay: 0.7s;"></span>
                    </div>
                </div>
                
                <div class="grid grid-cols-3 gap-4">
                    <div class="p-4 rounded-xl bg-emerald-900/20 border border-emerald-500/5 text-center">
                        <div class="text-sm md:text-base font-bold text-emerald-300">Smart Soil</div>
                        <div class="text-[10px] text-emerald-400/60 uppercase tracking-wider mt-1">Acoustic Intel</div>
                    </div>
                    <div class="p-4 rounded-xl bg-emerald-900/20 border border-emerald-500/5 text-center">
                        <div class="text-sm md:text-base font-bold text-emerald-300">Agri-Tech</div>
                        <div class="text-[10px] text-emerald-400/60 uppercase tracking-wider mt-1">Premium Platform</div>
                    </div>
                    <div class="p-4 rounded-xl bg-emerald-900/20 border border-emerald-500/5 text-center">
                        <div class="text-sm md:text-base font-bold text-emerald-300">Sustainable</div>
                        <div class="text-[10px] text-emerald-400/60 uppercase tracking-wider mt-1">Field Ready</div>
                    </div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_app_header():
    st.markdown(
        """
        <div class="app-header mx-6 mt-6 mb-4 p-6">
            <div class="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
                <div class="flex items-center gap-4">
                    <div class="logo-badge">🌱</div>
                    <div>
                        <div class="text-2xl md:text-3xl font-extrabold text-slate-950">TerraEcho Soil Analysis Studio</div>
                        <div class="text-sm text-slate-500 mt-1">AI Powered Soil Intelligence</div>
                    </div>
                </div>
                <div class="pulse-badge">AI Powered</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard():
    rows = fetch_all()
    total_samples = len(rows)
    avg_health = int(sum([r.get('health_score', 0) for r in rows]) / total_samples) if total_samples else 0
    recent_soil = rows[0]['soil_type'] if rows else 'N/A'
    health_bucket = 'Excellent' if avg_health >= 80 else 'Stable' if avg_health >= 60 else 'Attention'

    render_hero()

    st.markdown("<h2 class='text-2xl font-extrabold text-emerald-900 tracking-tight mx-6 mt-8 mb-4'>Enterprise Soil Overview</h2>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f"""
        <div class="p-6 rounded-2xl bg-white border border-emerald-500/10 shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300 min-h-[120px]">
            <div class="text-3xl font-extrabold text-emerald-700 mb-1">{total_samples}</div>
            <div class="text-xs font-semibold text-emerald-600/70 uppercase tracking-wider">Processed Samples</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    c2.markdown(
        f"""
        <div class="p-6 rounded-2xl bg-white border border-emerald-500/10 shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300 min-h-[120px]">
            <div class="text-3xl font-extrabold text-emerald-700 mb-1">{avg_health}%</div>
            <div class="text-xs font-semibold text-emerald-600/70 uppercase tracking-wider">Average Health Score</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    c3.markdown(
        f"""
        <div class="p-6 rounded-2xl bg-white border border-emerald-500/10 shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300 min-h-[120px]">
            <div class="text-2xl font-extrabold text-emerald-700 mb-2 truncate" title="{recent_soil}">{recent_soil}</div>
            <div class="text-xs font-semibold text-emerald-600/70 uppercase tracking-wider">Recent Soil Type</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    c4.markdown(
        f"""
        <div class="p-6 rounded-2xl bg-white border border-emerald-500/10 shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300 min-h-[120px]">
            <div class="text-2xl font-extrabold text-emerald-700 mb-2">{health_bucket}</div>
            <div class="text-xs font-semibold text-emerald-600/70 uppercase tracking-wider">Farming Readiness</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h3 class='text-xl font-bold text-emerald-900 tracking-tight mx-6 mt-8 mb-4'>Smart Insights</h3>", unsafe_allow_html=True)
    cols = st.columns(3)
    cols[0].markdown(
        """
        <div class="p-6 rounded-2xl bg-white border border-emerald-500/10 shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300 min-h-[160px] flex flex-col justify-between">
            <h4 class="text-md font-bold text-emerald-800">Soil Acoustics Engine</h4>
            <p class="text-xs text-emerald-700/80 leading-relaxed mt-2">Convert acoustic patterns into actionable soil health intelligence.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    cols[1].markdown(
        """
        <div class="p-6 rounded-2xl bg-white border border-emerald-500/10 shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300 min-h-[160px] flex flex-col justify-between">
            <h4 class="text-md font-bold text-emerald-800">Water Efficiency</h4>
            <p class="text-xs text-emerald-700/80 leading-relaxed mt-2">Predict water demand and avoid waste across the farm lifecycle.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    cols[2].markdown(
        """
        <div class="p-6 rounded-2xl bg-white border border-emerald-500/10 shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300 min-h-[160px] flex flex-col justify-between">
            <h4 class="text-md font-bold text-emerald-800">Climate-Smart Actions</h4>
            <p class="text-xs text-emerald-700/80 leading-relaxed mt-2">Deliver recommendations that balance yield, sustainability, and soil longevity.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h3 class='text-xl font-bold text-emerald-900 tracking-tight mx-6 mt-8 mb-4'>Trend Monitoring</h3>", unsafe_allow_html=True)
    if rows:
        health_series = [r.get('health_score', 0) for r in rows[:8]][::-1]
        dates = [r.get('uploaded_at')[-6:] for r in rows[:8]][::-1]
        fig, ax = plt.subplots(figsize=(10, 3), facecolor='none')
        ax.plot(dates, health_series, color='#059669', marker='o', linewidth=2)
        ax.fill_between(dates, health_series, color='#059669', alpha=0.15)
        ax.set_title('Soil Health Trend', color='#062c11', fontsize=11, fontweight='bold')
        ax.set_facecolor('none')
        ax.grid(alpha=0.15, color='#059669')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors='#4a6b52')
        st.pyplot(fig)
    else:
        st.info('Analyze your first sample to populate enterprise dashboards.')

    st.markdown("<h3 class='text-xl font-bold text-emerald-900 tracking-tight mx-6 mt-8 mb-4'>Platform Advantages</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mx-6 mb-8">
            <div class="p-6 rounded-2xl bg-white border border-emerald-500/10 shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300">
                <strong class="block text-emerald-800 font-bold mb-2">Precision Crop Plans</strong>
                <p class="text-xs text-emerald-700/80 leading-relaxed">Match crop choices to soil acoustic signatures.</p>
            </div>
            <div class="p-6 rounded-2xl bg-white border border-emerald-500/10 shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300">
                <strong class="block text-emerald-800 font-bold mb-2">Nutrient Forecast</strong>
                <p class="text-xs text-emerald-700/80 leading-relaxed">Pinpoint soil fertility risks before planting.</p>
            </div>
            <div class="p-6 rounded-2xl bg-white border border-emerald-500/10 shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300">
                <strong class="block text-emerald-800 font-bold mb-2">Community Impact</strong>
                <p class="text-xs text-emerald-700/80 leading-relaxed">Support climate-positive agriculture with clarity.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_soil_profile(result):
    confidence = min(100, max(65, int(result['health_score'] * 0.38 + 25)))
    crop_map = {
        'Dry Soil': 'Millet, Sorghum, Chickpea',
        'Healthy Soil': 'Wheat, Maize, Soybeans',
        'Compact Soil': 'Barley, Beans, Radish',
    }
    water_map = {
        'Dry Soil': 'High',
        'Healthy Soil': 'Moderate',
        'Compact Soil': 'Low',
    }
    treatment_map = {
        'Dry Soil': 'Irrigate & mulch to lock moisture.',
        'Healthy Soil': 'Maintain organic cover and routine sampling.',
        'Compact Soil': 'Aerate and add compost to improve texture.',
    }
    sustainability = min(100, result['health_score'] + 10)
    return {
        'confidence': confidence,
        'suggested_crops': crop_map.get(result['soil_type'], 'Adaptive grasses'),
        'water': water_map.get(result['soil_type'], 'Moderate'),
        'treatment': treatment_map.get(result['soil_type'], 'Optimize organic matter and aeration.'),
        'sustainability': sustainability,
    }


def get_ai_confidence(result, features):
    """Estimate AI confidence from soil health and acoustic features."""
    base_score = result.get('health_score', 50)
    energy = min(1.0, features.get('energy', 0) * 7)
    freq_ratio = min(1.0, features.get('frequency', 0) / 5000)
    confidence = int(min(100, max(65, base_score * 0.65 + energy * 18 + freq_ratio * 17)))
    return confidence


def get_recommendation_panel(soil_type):
    if soil_type == 'Compact Soil':
        st.warning(
            "**Compact Soil Recommendations**\n"
            "• Aerate soil\n"
            "• Add organic compost\n"
            "• Improve drainage\n"
            "• Reduce heavy machinery use"
        )
    elif soil_type == 'Dry Soil':
        st.warning(
            "**Dry Soil Recommendations**\n"
            "• Increase irrigation\n"
            "• Add mulch\n"
            "• Retain moisture"
        )
    else:
        st.success(
            "**Healthy Soil Recommendations**\n"
            "• Maintain current practices\n"
            "• Continue monitoring"
        )


def get_comparison_dataframe(result):
    return pd.DataFrame(
        {
            'Parameter': ['Moisture', 'Compaction', 'Health Score'],
            'Current': [result.get('moisture', '-'), result.get('compaction', '-'), f"{result.get('health_score', 0)}%"],
            'Ideal': ['Moderate', 'Low', '>85%'],
        }
    )


def get_ai_assistant_response(question):
    prompt = question.lower()
    if 'compact' in prompt or 'aerate' in prompt or 'drainage' in prompt:
        return 'Loosen the soil and add compost. Reduce heavy equipment and improve drainage for better root health.'
    if 'dry' in prompt or 'irrigation' in prompt or 'moisture' in prompt:
        return 'Increase irrigation, add mulch, and retain moisture with cover crops or organic mulch.'
    if 'healthy' in prompt or 'good' in prompt or 'maintain' in prompt:
        return 'Soil condition is good. Keep monitoring and maintain your current sustainable practices.'
    return 'TerraEcho suggests field sampling and routine soil monitoring. Ask about compact, dry, or healthy soil conditions.'


def render_analysis():
    st.markdown("<h2 id='soil-analysis' class='text-2xl font-extrabold text-slate-950 tracking-tight mx-6 mt-8 mb-2'>TerraEcho Soil Analysis Studio</h2>", unsafe_allow_html=True)
    st.markdown("<p class='text-sm text-slate-600 mx-6 mb-6'>Upload soil tapping audio to analyze soil health with pro-grade AI intelligence.</p>", unsafe_allow_html=True)

    with st.container():
        left, right = st.columns([2.8, 1.2])
        with left:
            st.markdown(
                """
                <div class="upload-card mx-6">
                    <div class="text-4xl mb-4">🎧</div>
                    <div class="upload-title mb-2">Upload soil tapping audio to analyze soil health</div>
                    <div class="upload-caption mb-6">Drop your .wav or .mp3 sample here for instant soil type classification, moisture analysis, and AI recommendation.</div>
                    <div class="upload-drop mb-6">
                        <div class="text-xl font-semibold text-slate-900 mb-2">Drag & drop your audio file</div>
                        <div class="text-sm upload-instructions">Accepted formats: WAV, MP3 • Less than 20MB</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            audio_file = st.file_uploader('', type=['wav', 'mp3'], label_visibility='collapsed')
            if audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format=audio_file.type)
            else:
                audio_bytes = None
            st.markdown("<div class='text-xs text-slate-500 mt-3 ml-1'>Accepted: .wav, .mp3 • Upload a sample to begin scoring.</div>", unsafe_allow_html=True)
        with right:
            st.markdown(
                """
                <div class="status-card">
                    <div class="text-md font-bold text-slate-900 mb-3">Premium Analysis Kit</div>
                    <div class="text-sm text-slate-600 leading-relaxed mb-4">Designed for farmers, research teams, and agri-tech judges.</div>
                    <ul class="text-sm text-slate-600 space-y-3 list-disc list-inside">
                        <li>Modern soil dashboard</li>
                        <li>AI-driven recommendations</li>
                        <li>Rich visual storytelling</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if audio_file and audio_bytes is not None:
        with st.spinner('Analyzing soil acoustic signal...'):
            progress = st.progress(0)
            progress.progress(20)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name
            progress.progress(45)
            features = process_audio(temp_path)
            result = predict_soil(features)
            profile = build_soil_profile(result)
            confidence = get_ai_confidence(result, features)
            duration = round(features.get('duration', 0), 1)
            progress.progress(100)

        st.markdown(
            f"""
            <div class="p-6 rounded-3xl bg-white border border-emerald-500/10 shadow-sm my-6 mx-6">
                <div class="text-md font-bold text-emerald-800 flex items-center gap-2 mb-3">
                    <span>🎵</span> Uploaded Audio Details
                </div>
                <div class="text-sm font-semibold text-emerald-900 truncate mb-4">{audio_file.name}</div>
                <ul class="grid grid-cols-3 gap-4 text-xs text-emerald-700/80">
                    <li class="bg-emerald-50/50 p-3 rounded-xl border border-emerald-500/5">
                        <strong class="block text-emerald-600/70 mb-1">Format</strong>
                        <span class="font-bold">{audio_file.type or 'Audio'}</span>
                    </li>
                    <li class="bg-emerald-50/50 p-3 rounded-xl border border-emerald-500/5">
                        <strong class="block text-emerald-600/70 mb-1">Size</strong>
                        <span class="font-bold">{round(audio_file.size / 1024, 1)} KB</span>
                    </li>
                    <li class="bg-emerald-50/50 p-3 rounded-xl border border-emerald-500/5">
                        <strong class="block text-emerald-600/70 mb-1">Duration</strong>
                        <span class="font-bold">{duration} sec</span>
                    </li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        soil_type = result['soil_type']
        if soil_type == 'Healthy Soil':
            soil_badge = "bg-emerald-50 border-emerald-500/20 text-emerald-900"
            soil_accent = "text-emerald-700"
        elif soil_type == 'Dry Soil':
            soil_badge = "bg-amber-50 border-amber-500/20 text-amber-900"
            soil_accent = "text-amber-700"
        else: # Compact Soil
            soil_badge = "bg-rose-50 border-rose-500/20 text-rose-900"
            soil_accent = "text-rose-700"

        c1, c2, c3, c4 = st.columns(4, gap='large')
        c1.markdown(
            f"""
            <div class="p-6 rounded-2xl border shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300 min-h-[150px] {soil_badge}">
                <div class="text-xs uppercase tracking-wide text-emerald-600/70 font-semibold mb-2">🟫 Soil Type</div>
                <div class="text-2xl font-black mb-1 {soil_accent}">{result['soil_type']}</div>
                <div class="text-[11px] text-emerald-700/60 leading-tight">Acoustic classification</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c2.markdown(
            f"""
            <div class="p-6 rounded-2xl border shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300 min-h-[150px] {soil_badge}">
                <div class="text-xs uppercase tracking-wide text-emerald-600/70 font-semibold mb-2">💧 Moisture</div>
                <div class="text-2xl font-black mb-1 {soil_accent}">{result['moisture']}</div>
                <div class="text-[11px] text-emerald-700/60 leading-tight">Soil moisture trend</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c3.markdown(
            f"""
            <div class="p-6 rounded-2xl border shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300 min-h-[150px] bg-white border border-emerald-500/10 text-emerald-950">
                <div class="text-xs uppercase tracking-wide text-emerald-600/70 font-semibold mb-2">🌱 Health Score</div>
                <div class="text-3xl font-black text-emerald-700 mb-1">{result['health_score']}%</div>
                <div class="text-[11px] text-emerald-700/60 leading-tight">AI soil health rating</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c4.markdown(
            f"""
            <div class="p-6 rounded-2xl border shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300 min-h-[150px] bg-white border border-emerald-500/10 text-emerald-950">
                <div class="text-xs uppercase tracking-wide text-emerald-600/70 font-semibold mb-2">🤖 AI Confidence</div>
                <div class="text-3xl font-black text-emerald-700 mb-1">{confidence}%</div>
                <div class="text-[11px] text-emerald-700/60 leading-tight">Model certainty</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        meter_status = 'Healthy' if result['health_score'] >= 70 else 'Moderate' if result['health_score'] >= 40 else 'Critical'
        meter_color_class = 'text-emerald-700 bg-emerald-50 border-emerald-500/20' if result['health_score'] >= 70 else 'text-amber-700 bg-amber-50 border-amber-500/20' if result['health_score'] >= 40 else 'text-rose-700 bg-rose-50 border-rose-500/20'
        
        st.markdown(
            f"""
            <div class="p-6 rounded-3xl bg-white border border-emerald-500/10 shadow-sm my-6 mx-6">
                <div class="text-md font-bold text-emerald-800 mb-3 flex items-center gap-2">
                    <span>📊</span> Soil Health Meter
                </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(result['health_score'] / 100)
        st.markdown(
            f"""
                <div class="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold border mt-3 {meter_color_class}">
                    {result['health_score']}% &mdash; {meter_status}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="p-6 rounded-3xl bg-white border border-emerald-500/10 shadow-sm my-6 mx-6">
                <div class="text-md font-bold text-emerald-800 mb-4 flex items-center gap-2">
                    <span>📈</span> Soil Technical Parameters
                </div>
                <ul class="grid grid-cols-2 md:grid-cols-3 gap-4 text-xs text-emerald-700/85">
                    <li class="bg-emerald-50/30 p-3 rounded-xl border border-emerald-500/5">
                        <strong class="text-emerald-600/70 block mb-1">Moisture Level</strong>
                        <span class="font-bold text-emerald-950 text-sm">{result['moisture']}</span>
                    </li>
                    <li class="bg-emerald-50/30 p-3 rounded-xl border border-emerald-500/5">
                        <strong class="text-emerald-600/70 block mb-1">Compaction</strong>
                        <span class="font-bold text-emerald-950 text-sm">{result['compaction']}</span>
                    </li>
                    <li class="bg-emerald-50/30 p-3 rounded-xl border border-emerald-500/5">
                        <strong class="text-emerald-600/70 block mb-1">Dryness</strong>
                        <span class="font-bold text-emerald-950 text-sm">{result['dryness']}</span>
                    </li>
                    <li class="bg-emerald-50/30 p-3 rounded-xl border border-emerald-500/5">
                        <strong class="text-emerald-600/70 block mb-1">Acoustic Frequency</strong>
                        <span class="font-bold text-emerald-950 text-sm">{int(features['frequency'])} Hz</span>
                    </li>
                    <li class="bg-emerald-50/30 p-3 rounded-xl border border-emerald-500/5">
                        <strong class="text-emerald-600/70 block mb-1">Signal Energy</strong>
                        <span class="font-bold text-emerald-950 text-sm">{features['energy']:.3f}</span>
                    </li>
                    <li class="bg-emerald-50/30 p-3 rounded-xl border border-emerald-500/5">
                        <strong class="text-emerald-600/70 block mb-1">Scan Duration</strong>
                        <span class="font-bold text-emerald-950 text-sm">{duration} sec</span>
                    </li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='my-6 mx-6'>", unsafe_allow_html=True)
        w1, w2 = st.columns([1.4, 1])
        with w1:
            st.markdown(
                """
                <div class="p-6 rounded-3xl bg-white border border-emerald-500/10 shadow-sm h-full">
                    <div class="text-md font-bold text-emerald-800 mb-4 flex items-center gap-2">
                        <span>🌊</span> Soil Sound Waveform
                    </div>
                """,
                unsafe_allow_html=True,
            )
            waveform_fig = plot_waveform(features['y'], features['sr'])
            waveform_fig.axes[0].set_xlabel('Time (s)', color='#4a6b52')
            waveform_fig.axes[0].set_ylabel('Amplitude', color='#4a6b52')
            waveform_fig.axes[0].grid(alpha=0.15, color='#059669')
            st.pyplot(waveform_fig)
            st.markdown("</div>", unsafe_allow_html=True)
        with w2:
            st.markdown(
                """
                <div class="p-6 rounded-3xl bg-white border border-emerald-500/10 shadow-sm h-full">
                    <div class="text-md font-bold text-emerald-800 mb-4 flex items-center gap-2">
                        <span>📊</span> Extracted Audio Features
                    </div>
                """,
                unsafe_allow_html=True,
            )
            graph_type = st.radio('View as', ['Bar chart', 'Line chart', 'Radar chart'], horizontal=True, key='feature_graph_type')
            keys = ['frequency', 'energy', 'duration']
            values = [features.get(k, 0) for k in keys]

            if graph_type == 'Radar chart':
                feature_fig = plt.figure(figsize=(5.5, 3), facecolor='none')
                ax = feature_fig.add_subplot(111, polar=True)
                angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
                values_cyclic = values + values[:1]
                angles += angles[:1]
                ax.plot(angles, values_cyclic, color='#059669', linewidth=2)
                ax.fill(angles, values_cyclic, color='#10b981', alpha=0.25)
                ax.set_thetagrids(np.degrees(angles[:-1]), [key.capitalize() for key in keys])
                ax.set_title('Feature Radar Summary', color='#062c11', y=1.08, fontweight='bold')
                ax.grid(alpha=0.18, color='#059669')
                ax.set_facecolor('none')
                ax.tick_params(colors='#4a6b52')
                ax.spines['polar'].set_color('#10b981')
            else:
                feature_fig, ax = plt.subplots(figsize=(5.5, 3), facecolor='none')
                if graph_type == 'Line chart':
                    ax.plot(keys, values, marker='o', color='#059669', linewidth=2)
                    ax.set_title('Feature Trend Summary', color='#062c11', fontweight='bold')
                    ax.set_ylabel('Value', color='#4a6b52')
                    ax.grid(alpha=0.15, color='#059669')
                else:
                    bars = ax.bar(keys, values, color=['#059669', '#10b981', '#34d399'])
                    ax.set_title('Feature Values', color='#062c11', fontweight='bold')
                    ax.set_ylabel('Value', color='#4a6b52')
                    ax.grid(axis='y', alpha=0.15, color='#059669')
                    for bar in bars:
                        height = bar.get_height()
                        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=9, color='#062c11')
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(colors='#4a6b52')

            st.pyplot(feature_fig)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if result['soil_type'] == 'Compact Soil':
            rec_badge = "bg-rose-50 border border-rose-500/20 text-rose-900"
            rec_title = "⚠ Soil Aeration Recommended"
            rec_bullets = """
                <li>Add organic compost to loosen texture and introduce bio-matter</li>
                <li>Improve field drainage channels to prevent waterlogging</li>
                <li>Avoid heavy machinery usage during high moisture days</li>
                <li>Re-sample and monitor soil moisture after 3 days</li>
            """
        elif result['soil_type'] == 'Dry Soil':
            rec_badge = "bg-amber-50 border border-amber-500/20 text-amber-900"
            rec_title = "💧 Soil Irrigation Needed"
            rec_bullets = """
                <li>Increase watering frequency with drip irrigation if possible</li>
                <li>Add organic mulch layers to restrict soil moisture loss</li>
                <li>Maintain low crop cover or shade nets to avoid excessive evaporation</li>
            """
        else:
            rec_badge = "bg-emerald-50 border border-emerald-500/20 text-emerald-900"
            rec_title = "🌱 Soil Condition Stable & Good"
            rec_bullets = """
                <li>Maintain current sustainable farming practices</li>
                <li>Continue routine sampling weekly to build soil acoustic trends</li>
            """

        st.markdown(
            f"""
            <div class="p-6 rounded-3xl border shadow-sm my-6 mx-6 {rec_badge}">
                <div class="text-md font-bold flex items-center gap-2 mb-3">
                    <span>🧠</span> {rec_title}
                </div>
                <ul class="text-xs space-y-2 list-disc list-inside opacity-90 leading-relaxed">
                    {rec_bullets}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        comp_df = get_comparison_dataframe(result)
        st.markdown(
            """
            <div class="p-6 rounded-3xl bg-white border border-emerald-500/10 shadow-sm my-6 mx-6">
                <div class="text-md font-bold text-emerald-800 mb-3 flex items-center gap-2">
                    <span>📋</span> Current vs Ideal Parameters
                </div>
            """,
            unsafe_allow_html=True
        )
        st.dataframe(comp_df, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="p-6 rounded-3xl bg-white border border-emerald-500/10 shadow-sm my-6 mx-6 relative">
                <div class="absolute top-6 right-6 flex items-center gap-2 bg-emerald-50 px-2.5 py-1 rounded-full border border-emerald-500/20">
                    <span class="relative flex h-2 w-2">
                        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                        <span class="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                    </span>
                    <span class="text-[10px] text-emerald-700 font-bold uppercase tracking-wider">Online</span>
                </div>
                <h3 class="text-md font-bold text-emerald-800 mb-1">🤖 Ask TerraEcho AI</h3>
                <p class="text-xs text-emerald-700/60 mb-4">Chat with our AI agronomist assistant about soil aeration, crop cycles, or irrigation.</p>
            """,
            unsafe_allow_html=True
        )
        user_question = st.text_input('Ask TerraEcho AI', key='terraecho_ai', placeholder='My soil is compact. What should I do?')
        if st.button('Ask TerraEcho', key='assistant_button'):
            answer = get_ai_assistant_response(user_question)
            st.markdown(
                f"""
                <div class="mt-4 p-4 rounded-2xl bg-emerald-50/70 border border-emerald-500/15 text-xs text-emerald-950 leading-relaxed shadow-sm">
                    <strong class="text-emerald-800 block mb-1">TerraEcho Agronomist:</strong>
                    {answer}
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        os.makedirs('reports', exist_ok=True)
        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        report_path = os.path.join('reports', f'soil_report_{sample_id}_{timestamp}.pdf')
        waveform_path = os.path.join('reports', f'wave_{sample_id}_{timestamp}.png')
        feature_path = os.path.join('reports', f'features_{sample_id}_{timestamp}.png')

        try:
            waveform_fig.savefig(waveform_path, bbox_inches='tight', dpi=150)
        except Exception:
            waveform_path = None

        try:
            save_feature_plot(features, feature_path)
        except Exception:
            feature_path = None

        metadata = {
            'sample_id': sample_id,
            'location': location,
            'notes': notes,
            'uploaded_at': timestamp,
            'filename': os.path.basename(report_path),
            'waveform': os.path.basename(waveform_path) if waveform_path and os.path.exists(waveform_path) else None,
            'feature_file': os.path.basename(feature_path) if feature_path and os.path.exists(feature_path) else None,
            'feature_plot': feature_path if feature_path and os.path.exists(feature_path) else None,
        }

        generate_report(result, report_path, waveform_path=waveform_path, metadata=metadata)
        save_result(result, metadata)

        st.markdown(
            """
            <div class="p-6 rounded-3xl bg-white border border-emerald-500/10 shadow-sm my-6 mx-6 text-center">
            """,
            unsafe_allow_html=True
        )
        with open(report_path, 'rb') as file:
            st.download_button('📄 Download Soil Report PDF', file, file_name=os.path.basename(report_path))
        st.markdown("</div>", unsafe_allow_html=True)

        st.balloons()


def render_reports():
    st.markdown("<h2 id='reports' class='text-2xl font-extrabold text-emerald-900 tracking-tight mx-6 mt-8 mb-2'>Reports & Exports</h2>", unsafe_allow_html=True)
    st.markdown("<p class='text-sm text-emerald-700/80 mx-6 mb-6'>Download investor-grade soil health reports and export your portfolio.</p>", unsafe_allow_html=True)

    rows = fetch_all()
    if not rows:
        st.warning('No reports available yet. Run a soil analysis to generate your first report.')
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="p-6 rounded-2xl bg-white border border-emerald-500/10 shadow-sm min-h-[100px]">
                <div class="text-xs font-semibold text-emerald-600/70 uppercase tracking-wider mb-2">Total Reports</div>
                <div class="text-3xl font-black text-emerald-700">{len(rows)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="p-6 rounded-2xl bg-white border border-emerald-500/10 shadow-sm min-h-[100px]">
                <div class="text-xs font-semibold text-emerald-600/70 uppercase tracking-wider mb-2">Recent Run</div>
                <div class="text-lg font-bold text-emerald-800 truncate" title="{rows[0].get('uploaded_at')}">{rows[0].get('uploaded_at')}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="p-6 rounded-2xl bg-emerald-50/50 border border-emerald-500/10 shadow-sm mx-6 my-6">
            <div class="flex flex-col md:flex-row items-center justify-between gap-4">
                <div>
                    <div class="font-bold text-emerald-950 text-sm">Export Entire Portfolio</div>
                    <div class="text-xs text-emerald-700/70">Download all generated soil reports in a single zip archive.</div>
                </div>
        """,
        unsafe_allow_html=True
    )
    if st.button('Prepare Report Bundle', key='prepare_zip'):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zf:
            for r in rows:
                report_file = os.path.join('reports', r.get('filename', ''))
                if report_file and os.path.exists(report_file):
                    zf.write(report_file, arcname=os.path.basename(report_file))
        buffer.seek(0)
        st.download_button('📥 Download TerraEcho Report Bundle', data=buffer.getvalue(), file_name='terraecho_reports.zip', mime='application/zip')
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("<h3 class='text-lg font-bold text-emerald-900 tracking-tight mx-6 mt-8 mb-4'>Available Soil Reports</h3>", unsafe_allow_html=True)
    for r in rows:
        st.markdown(
            f"""
            <div class="p-6 rounded-2xl bg-white border border-emerald-500/10 shadow-sm hover:shadow-md transition-all duration-300 mx-6 mb-4">
                <div class="grid grid-cols-1 md:grid-cols-3 items-center gap-4">
                    <div>
                        <div class="text-xs uppercase tracking-wider text-emerald-600/70 font-semibold mb-1">Sample Metadata</div>
                        <div class="font-bold text-emerald-950 text-base">Sample #{r.get('sample_id')}</div>
                        <div class="text-xs text-emerald-700/80 mt-1">📍 {r.get('location','Unknown Location')}</div>
                    </div>
                    <div>
                        <div class="text-xs uppercase tracking-wider text-emerald-600/70 font-semibold mb-1">Analysis Result</div>
                        <div class="font-bold text-emerald-950 text-base">{r.get('soil_type','Unknown Soil')}</div>
                        <div class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-bold bg-emerald-50 border border-emerald-500/10 text-emerald-700 mt-1">
                            Health Score: {r.get('health_score','-')}%
                        </div>
                    </div>
                    <div class="flex md:justify-end">
            """,
            unsafe_allow_html=True
        )
        report_file = os.path.join('reports', r.get('filename', ''))
        if os.path.exists(report_file):
            with open(report_file, 'rb') as fh:
                st.download_button(
                    label=f"📥 Download PDF",
                    data=fh,
                    file_name=r.get('filename'),
                    key=f"dl_{r.get('sample_id')}",
                    use_container_width=True
                )
        st.markdown(
            """
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_ai_insights():
    st.markdown("<h2 class='text-2xl font-extrabold text-emerald-900 tracking-tight mx-6 mt-8 mb-2'>AI Agronomy Insights</h2>", unsafe_allow_html=True)
    st.markdown("<p class='text-sm text-emerald-700/80 mx-6 mb-6'>Deep soil acoustic intelligence for smarter agronomy and ecological health monitoring.</p>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mx-6 mb-8">
            <div class="p-6 rounded-2xl bg-white border border-emerald-500/10 shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300">
                <div class="text-3xl mb-3">🔊</div>
                <strong class="block text-emerald-800 font-bold mb-2">Frequency Intelligence</strong>
                <p class="text-xs text-emerald-700/80 leading-relaxed">Understand how sound spectral patterns map to soil moisture and compaction level variations.</p>
            </div>
            <div class="p-6 rounded-2xl bg-white border border-emerald-500/10 shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300">
                <div class="text-3xl mb-3">🌾</div>
                <strong class="block text-emerald-800 font-bold mb-2">Crop Reflection</strong>
                <p class="text-xs text-emerald-700/80 leading-relaxed">Match agro-recommendations to the detected soil health profile for maximum crop yield potential.</p>
            </div>
            <div class="p-6 rounded-2xl bg-white border border-emerald-500/10 shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300">
                <div class="text-3xl mb-3">🍃</div>
                <strong class="block text-emerald-800 font-bold mb-2">Sustainability Index</strong>
                <p class="text-xs text-emerald-700/80 leading-relaxed">Track regenerative farming signals through acoustic soil analysis and organic matter trends.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows = fetch_all()
    if rows:
        recent = rows[:6][::-1]
        values = [r.get('health_score', 0) for r in recent]
        
        st.markdown("<h3 class='text-xl font-bold text-emerald-900 tracking-tight mx-6 mt-8 mb-4'>Health Score Distribution</h3>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(9, 3), facecolor='none')
        colors = ['#059669' if v >= 70 else '#34d399' for v in values]
        ax.bar(range(len(values)), values, color=colors, width=0.5, edgecolor='#047857', linewidth=0.5)
        ax.set_title('Historical Analysis Health Scores', color='#062c11', fontweight='bold', fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([r.get('sample_id') for r in recent], color='#4a6b52', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#10b981')
        ax.spines['bottom'].set_color('#10b981')
        ax.grid(axis='y', alpha=0.15, color='#059669')
        ax.tick_params(colors='#4a6b52')
        ax.set_facecolor('none')
        
        for idx, val in enumerate(values):
            ax.annotate(f'{val}%', xy=(idx, val), xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=9, color='#062c11', fontweight='bold')
            
        st.pyplot(fig)
    else:
        st.info('Collect soil samples to activate AI insight dashboards.', icon='🌿')


def render_history():
    st.markdown("<h2 class='text-2xl font-extrabold text-emerald-900 tracking-tight mx-6 mt-8 mb-2'>Soil History</h2>", unsafe_allow_html=True)
    st.markdown("<p class='text-sm text-emerald-700/80 mx-6 mb-6'>Review and compare past soil analyses performed in your workspace.</p>", unsafe_allow_html=True)
    
    rows = fetch_all()
    if not rows:
        st.info('No soil history available yet. Complete an analysis to start tracking.', icon='🧭')
        return

    for r in rows:
        soil_type = r.get('soil_type')
        if soil_type == 'Healthy Soil':
            status_pill = "bg-emerald-50 text-emerald-700 border-emerald-500/20"
        elif soil_type == 'Dry Soil':
            status_pill = "bg-amber-50 text-amber-700 border-amber-500/20"
        else:
            status_pill = "bg-rose-50 text-rose-700 border-rose-500/20"
            
        st.markdown(
            f"""
            <div class="p-6 rounded-2xl bg-white border border-emerald-500/10 shadow-sm hover:shadow-md transition-all duration-300 mx-6 mb-4">
                <div class="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-emerald-500/5 pb-4 mb-4 gap-2">
                    <div>
                        <span class="text-xs text-emerald-600/70 font-semibold uppercase tracking-wider">Analysis Log</span>
                        <div class="text-lg font-bold text-emerald-950 mt-0.5">Sample #{r.get('sample_id')}</div>
                    </div>
                    <div class="flex items-center gap-3">
                        <span class="text-xs text-emerald-600/60">📅 {r.get('uploaded_at')}</span>
                        <span class="px-3 py-1 rounded-full text-xs font-bold border {status_pill}">{soil_type}</span>
                    </div>
                </div>
                
                <div class="grid grid-cols-2 md:grid-cols-5 gap-4 text-xs">
                    <div class="bg-emerald-50/20 p-3 rounded-xl border border-emerald-500/5">
                        <span class="text-emerald-600/70 block mb-1">Health Score</span>
                        <span class="font-bold text-emerald-950 text-sm">{r.get('health_score')}%</span>
                    </div>
                    <div class="bg-emerald-50/20 p-3 rounded-xl border border-emerald-500/5">
                        <span class="text-emerald-600/70 block mb-1">Moisture</span>
                        <span class="font-bold text-emerald-950 text-sm">{r.get('moisture')}</span>
                    </div>
                    <div class="bg-emerald-50/20 p-3 rounded-xl border border-emerald-500/5">
                        <span class="text-emerald-600/70 block mb-1">Compaction</span>
                        <span class="font-bold text-emerald-950 text-sm">{r.get('compaction')}</span>
                    </div>
                    <div class="bg-emerald-50/20 p-3 rounded-xl border border-emerald-500/5">
                        <span class="text-emerald-600/70 block mb-1">Dryness</span>
                        <span class="font-bold text-emerald-950 text-sm">{r.get('dryness')}</span>
                    </div>
                    <div class="bg-emerald-50/20 p-3 rounded-xl border border-emerald-500/5 col-span-2 md:col-span-1">
                        <span class="text-emerald-600/70 block mb-1">Location</span>
                        <span class="font-bold text-emerald-950 text-sm truncate block" title="{r.get('location','-')}">{r.get('location','-')}</span>
                    </div>
                </div>
                
                {f'<div class="mt-4 p-3 rounded-xl bg-emerald-50/40 border border-emerald-500/5 text-xs text-emerald-800 leading-relaxed"><strong>Notes:</strong> {r.get("notes")}</div>' if r.get("notes") else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_about():
    st.markdown("<h2 class='text-2xl font-extrabold text-emerald-900 tracking-tight mx-6 mt-8 mb-2'>About TerraEcho</h2>", unsafe_allow_html=True)
    st.markdown("<p class='text-sm text-emerald-700/80 mx-6 mb-6'>Built for farmers, researchers, NGOs, and agri-tech innovators focused on resilient soil systems.</p>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mx-6 mb-8">
            <div class="p-6 rounded-2xl bg-white border border-emerald-500/10 shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300">
                <div class="text-3xl mb-3">🎯</div>
                <h3 class="text-lg font-bold text-emerald-800 mb-2">Mission</h3>
                <p class="text-xs text-emerald-700/80 leading-relaxed">Empower sustainable agriculture with accurate, easy-to-use soil health intelligence using acoustic resonance.</p>
            </div>
            <div class="p-6 rounded-2xl bg-white border border-emerald-500/10 shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300">
                <div class="text-3xl mb-3">🔮</div>
                <h3 class="text-lg font-bold text-emerald-800 mb-2">Vision</h3>
                <p class="text-xs text-emerald-700/80 leading-relaxed">Transform soil scanning into fast, confident decisions for every field to enhance global crop security.</p>
            </div>
            <div class="p-6 rounded-2xl bg-white border border-emerald-500/10 shadow-sm hover:shadow-md hover:scale-[1.02] transition-all duration-300">
                <div class="text-3xl mb-3">🌱</div>
                <h3 class="text-lg font-bold text-emerald-800 mb-2">Values</h3>
                <p class="text-xs text-emerald-700/80 leading-relaxed">Precision, environmental sustainability, technological accessibility, and measurable climate impact.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mx-6 mb-8">
            <div class="p-6 rounded-2xl bg-emerald-50/50 border border-emerald-500/10 shadow-sm">
                <div class="text-xs text-emerald-600/70 font-semibold uppercase tracking-wider mb-2">Get in Touch</div>
                <strong class="block text-emerald-955 text-sm mb-1">Contact</strong>
                <p class="text-xs text-emerald-700/80">hello@terraecho.ai</p>
            </div>
            <div class="p-6 rounded-2xl bg-emerald-50/50 border border-emerald-500/10 shadow-sm">
                <div class="text-xs text-emerald-600/70 font-semibold uppercase tracking-wider mb-2">Open Source</div>
                <strong class="block text-emerald-955 text-sm mb-1">GitHub Repository</strong>
                <p class="text-xs text-emerald-700/80">github.com/terraecho</p>
            </div>
            <div class="p-6 rounded-2xl bg-emerald-50/50 border border-emerald-500/10 shadow-sm">
                <div class="text-xs text-emerald-600/70 font-semibold uppercase tracking-wider mb-2">Eco Impact</div>
                <strong class="block text-emerald-955 text-sm mb-1">Sustainability</strong>
                <p class="text-xs text-emerald-700/80 leading-relaxed">Supporting soil resilience, carbon sequestration, and climate-smart food networks.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_app_header()
if mode == '🏠 Dashboard':
    render_dashboard()
elif mode == '🌱 Soil Analysis':
    render_analysis()
elif mode == '📄 Reports':
    render_reports()
elif mode == '🤖 AI Insights':
    render_ai_insights()
elif mode == '🕒 History':
    render_history()
elif mode == 'ℹ️ About':
    render_about()
