from pathlib import Path
import streamlit as st
import pandas as pd
from utils.icons import get_icon

def load_theme(css_path: str = "styles/theme.css"):
    path = Path(css_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_ops_header(title: str, subtitle: str, badge_text: str):
    icon_svg = get_icon("layers")
    html = f"""
    <div class="ops-header">
        <div class="ops-title-group">
            <div class="ops-icon-emblem">{icon_svg}</div>
            <div>
                <h1 class="ops-title">{title}</h1>
                <span class="ops-subtitle">{subtitle}</span>
            </div>
        </div>
        <div class="ops-status-pill">
            <span class="pulse-dot"></span>
            {badge_text}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_bento_telemetry(total_dotacion: int, asignados: int, ambiguos: int, siglas_unk: int):
    pct = round((asignados / total_dotacion * 100), 1) if total_dotacion > 0 else 0.0
    sin_asignar = max(total_dotacion - asignados, 0)
    
    color_asignados = "#57F2CC" if asignados > 0 else "#F3F5F8"
    color_ambiguos = "#FF5A5F" if ambiguos > 0 else "#57F2CC"
    color_siglas = "#E0A96D" if siglas_unk > 0 else "#57F2CC"

    html = f"""
    <div class="bento-grid">
        <div class="bento-card">
            <div class="bento-card-top">
                <span class="bento-label">Dotacion Registrada</span>
                <span class="bento-icon">{get_icon("users")}</span>
            </div>
            <div class="bento-value">{total_dotacion}</div>
            <div class="bento-meta">Colaboradores en plantilla BUK</div>
        </div>
        <div class="bento-card">
            <div class="bento-card-top">
                <span class="bento-label">Con Turnos Asignados</span>
                <span class="bento-icon" style="color: {color_asignados};">{get_icon("check_circle")}</span>
            </div>
            <div class="bento-value" style="color: {color_asignados};">{asignados} <span style="font-size: 1rem; opacity: 0.75;">({pct}%)</span></div>
            <div class="bento-meta">Turno Base / Sin Asignar: {sin_asignar}</div>
        </div>
        <div class="bento-card">
            <div class="bento-card-top">
                <span class="bento-label">Nombres Ambiguos</span>
                <span class="bento-icon" style="color: {color_ambiguos};">{get_icon("alert_triangle")}</span>
            </div>
            <div class="bento-value" style="color: {color_ambiguos};">{ambiguos}</div>
            <div class="bento-meta">Pendientes de desempate por area</div>
        </div>
        <div class="bento-card">
            <div class="bento-card-top">
                <span class="bento-label">Siglas No Registradas</span>
                <span class="bento-icon" style="color: {color_siglas};">{get_icon("clock")}</span>
            </div>
            <div class="bento-value" style="color: {color_siglas};">{siglas_unk}</div>
            <div class="bento-meta">Requieren homologacion manual</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_area_banner(title: str, description: str, code_tag: str):
    html = f"""
    <div class="spec-banner">
        <div>
            <div class="spec-title">{title}</div>
            <div class="spec-desc">{description}</div>
        </div>
        <span class="spec-code">{code_tag}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_shift_audit_table(df: pd.DataFrame, date_cols: list[str]):
    """Renderiza la matriz de turnos en vidrio oscuro con badges semánticos por sigla."""
    cols_to_show = ['Nombre del Colaborador', 'Área'] + date_cols
    sub_df = df[cols_to_show].copy()

    headers_html = "".join([f"<th>{col}</th>" for col in cols_to_show])
    rows_html = []

    for _, row in sub_df.iterrows():
        cells = [
            f"<td style='font-weight: 600;'>{row['Nombre del Colaborador']}</td>",
            f"<td><span class='tag-area'>{row['Área']}</span></td>"
        ]
        for d_col in date_cols:
            val = str(row[d_col]).strip()
            if val in ['BASE', 'nan', '']:
                cells.append(f"<td><span class='shift-base'>BASE</span></td>")
            elif val in ['L', 'C']:
                cells.append(f"<td><span class='shift-libre'>{val}</span></td>")
            else:
                cells.append(f"<td><span class='shift-active'>{val}</span></td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    table_html = f"""
    <div class="audit-table-wrapper">
        <table class="audit-table">
            <thead>
                <tr>{headers_html}</tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)