import os
import streamlit as st

def get_svg_icon(name: str) -> str:
    """Catalogo centralizado de iconos vectoriales limpios."""
    icons = {
        "alert": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        "check": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
        "shield": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
        "filter": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>'
    }
    return icons.get(name, "")

def load_custom_css(file_path: str = "styles/theme.css"):
    """Carga la hoja CSS desacoplada."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"Hoja de estilos no encontrada en: {file_path}")

def render_header(title: str, subtitle: str, tag: str):
    """Renderiza la cabecera principal."""
    html = f"""
    <div class="header-container">
        <div>
            <span class="badge-mono badge-accent">{tag}</span>
            <h1 class="header-title">{title}</h1>
            <p class="header-subtitle">{subtitle}</p>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_telemetry(total_buk: int, extraidos: int, ambiguos: int, desconocidos: int):
    """Panel Bento de conteo inicial."""
    html = f"""
    <div class="telemetry-grid">
        <div class="telemetry-card">
            <div class="telemetry-label">Base Oficial BUK</div>
            <div class="telemetry-value">{total_buk}</div>
        </div>
        <div class="telemetry-card">
            <div class="telemetry-label">Turnos Detectados</div>
            <div class="telemetry-value">{extraidos}</div>
        </div>
        <div class="telemetry-card">
            <div class="telemetry-label">Identidades Ambiguas</div>
            <div class="telemetry-value" style="color: {'var(--status-warn-text)' if ambiguos > 0 else 'var(--text-primary)'};">{ambiguos}</div>
        </div>
        <div class="telemetry-card">
            <div class="telemetry-label">Siglas no Registradas</div>
            <div class="telemetry-value" style="color: {'var(--status-error-text)' if desconocidos > 0 else 'var(--text-primary)'};">{desconocidos}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_area_spec(title: str, description: str, tag: str):
    """Ficha informativa por area de carga."""
    html = f"""
    <div class="area-spec-box">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span class="area-spec-title">{title}</span>
            <span class="badge-mono">{tag}</span>
        </div>
        <div class="area-spec-desc">{description}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_queue_status(conteos_dict: dict):
    """Balance de documentos en cola."""
    items_html = "".join([
        f"<span class='badge-mono' style='margin-right: 0.5rem;'>{area}: {count}</span>"
        for area, count in conteos_dict.items() if count > 0
    ])
    html = f"""
    <div style="margin: 0.85rem 0; display: flex; align-items: center; gap: 0.6rem;">
        <span class="badge-mono badge-accent">COLA DE PROCESAMIENTO</span>
        {items_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_preview_metrics(area_label: str, total_colabs: int, con_turno: int, turnos_base: int):
    """Metricas operativas de la vista previa filtrada por area."""
    pct = round((con_turno / total_colabs * 100), 1) if total_colabs > 0 else 0
    html = f"""
    <div class="preview-summary-bar">
        <div class="preview-stat">
            <span class="preview-stat-label">Area Auditada</span>
            <span class="preview-stat-val" style="color: var(--accent-amber);">{area_label}</span>
        </div>
        <div class="preview-stat" style="border-left: 1px solid var(--border-medium); padding-left: 1rem;">
            <span class="preview-stat-label">Dotacion Registrada</span>
            <span class="preview-stat-val">{total_colabs}</span>
        </div>
        <div class="preview-stat" style="border-left: 1px solid var(--border-medium); padding-left: 1rem;">
            <span class="preview-stat-label">Con Turnos Asignados</span>
            <span class="preview-stat-val" style="color: var(--status-ok-text);">{con_turno} ({pct}%)</span>
        </div>
        <div class="preview-stat" style="border-left: 1px solid var(--border-medium); padding-left: 1rem;">
            <span class="preview-stat-label">Turno Base / Sin Asignar</span>
            <span class="preview-stat-val" style="color: var(--text-muted);">{turnos_base}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_notice(level: str, title: str, message: str):
    """Alertas operativas con contraste accesible."""
    icon = get_svg_icon("alert" if level in ["warning", "error"] else "check")
    html = f"""
    <div class="notice-box {level}">
        <span class="notice-icon">{icon}</span>
        <div><strong>{title}:</strong> {message}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)