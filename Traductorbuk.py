import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import json
import difflib

# Modulos de diseno desacoplados
from utils.ui import (
    load_custom_css,
    render_header,
    render_telemetry,
    render_area_spec,
    render_queue_status,
    render_preview_metrics,
    render_notice,
    get_svg_icon
)

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# -----------------------------------------------------------------------------
# CONFIGURACION DE PAGINA Y ESTILOS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Traductor de Cuadrantes · Bar El Bajo",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_custom_css("styles/theme.css")

render_header(
    title="Traductor de Turnos BUK",
    subtitle="Procesamiento automatizado de cuadrantes semanales mediante Vision Artificial",
    tag="Bar El Bajo · Ops"
)

# -----------------------------------------------------------------------------
# DICCIONARIO DE ALIAS OPERATIVOS
# -----------------------------------------------------------------------------
ALIAS_PREDEFINIDOS = {
    "FRAN(CISCO)": "FRANCISCO IGNACIO GUERRA GALLARDO",
    "REEMPLAZO(RICARDO)": "RICARDO FELIPE CARCAMO ARAVENA",
    "AMELIA": "ELIZABETH AMELIA ESMERALDA MADARIAGA OCARES",
    "JAZMIN": "JAZMÍN ALEJANDRA PALMA SOTO",
    "SULAN": "SULAN NADINE VILCHES RUÍZ",
    "DAYANA": "DAYANA MICHELLE CABRERA ALVARADO",
    "JOSMAR": "JOSMARYOHANA CAROLINA GONZALEZ ZAMBRANO",
    "BENJAMIN": "BENJAMIN JESÚS CARES DÍAZ",
    "MIGUEL": "MIGUEL ANGEL BANDA MUÑOZ",
    "ANDREINA": "RAULYMAR ANDREINA RODRIGUEZ RUIZ",
    "GUILLERMO": "GUILLERMO IGNACIO GONZÁLEZ SALAS",
    "JEFFERSON": "JEFFERSON JOSUE RINCON CHACON",
    "BETTY": "BETTY LUISANA COLOMBO CHIRINOS",
    "SEBASTIAN": "SEBASTIÁN ALONSO CARTES LARA",
    "IGNACIO": "IGNACIO MOYA RUFFO",
    "SOFIA": "SOFÍA VALENTINA DANTY MOREIRA",
    "MARTIN": "MARTÍN ANTONIO RIQUELME FUENZALIDA",
    "JOCELYN": "JOCELYN NICOLE RECABAL MARTÍNEZ",
    "JAIDER": "JAIDER ALEGRIA MINOTA",
    "EVELYN": "EVELYN GEORGINA FRANCHESCA TRAMOLAO LAGOS",
    "JULIA": "JULIA FRANCISCA TUESCA VIZCAINO",
    "KEYLA": "KEYLA EDEN SOTO YEPEZ",
    "KEN": "KEN JOSE LUGO AÑEZ",
    "KAHIL": "KAHIL MAHESH BLANCO SOTO",
    "JENNYFER": "JENNIFER LOPEZ PESINA",
    "MARIA": "MARIA ANDREA IZQUIERDO VILLA",
    "LUISA": "LUISA FERNANDA CAICEDO",
    "RENATO": "RENATO MIGUEL PIMINCHUMO RODRIGUEZ",
    "MICHELL": "DENNIS MICHAEL LABAN AREVALO",
    "ROXANA": "ROXANA XIOMARA VILLACRESES ALBAN",
    "NICOLAS": "NICOLAS ALEJANDRO TORO TOBAR",
    "JORGE": "JORGE ANDRÉS DELGADO SEGUEL",
    "DYLAN": "DYLAN MÉNDEZ DINAMARCA",
    "GASPAR": "GASPAR SEBASTIÁN IGNACIO SILVA MARTÍNEZ"
}

# -----------------------------------------------------------------------------
# FUNCIONES AUXILIARES DE PROCESAMIENTO
# -----------------------------------------------------------------------------
def normalize_area(section_str):
    """Mapea el area detectada a la nomenclatura oficial de BUK."""
    if not section_str:
        return None
    sec = str(section_str).upper().strip()
    if any(k in sec for k in ['BARTENDER', 'BARRA', 'BAR']):
        return 'BARRA'
    if any(k in sec for k in ['GARZON', 'GARZONES', 'RUNNER', 'SALON', 'SALÓN']):
        return 'SALON'
    if any(k in sec for k in ['NIKKEI', 'FRIO', 'FRÍO', 'PIZZA', 'PIZZAS', 'CALIENTE', 'PRODUCCION', 'PRODUCCIÓN', 'COOPERIA', 'COCINA', 'JEFATURA']):
        return 'COCINA'
    return None

def build_shift_dictionary(xls_file):
    """Construye el mapa de Entrada-Salida a Siglas oficiales desde turnosSemanales."""
    try:
        ts_df = pd.read_excel(xls_file, sheet_name='turnosSemanales')
        shift_map = {}
        all_siglas = set()
        
        for _, row in ts_df.iterrows():
            sigla = str(row['Siglas']).strip()
            ent = str(row['Entrada del turno']).strip()
            sal = str(row['Salida del turno']).strip()
            if sigla and sigla != 'nan':
                all_siglas.add(sigla)
            if ent != '-' and sal != '-':
                key = f"{ent}-{sal}"
                shift_map[key] = sigla
        
        return shift_map, sorted(list(all_siglas))
    except Exception as e:
        render_notice("error", "Error de lectura", f"No se pudo procesar la hoja turnosSemanales ({e})")
        return {}, []

def convert_time_to_buk_sigla(val, shift_map):
    """Convierte un rango horario o texto a sigla BUK normalizada."""
    if pd.isna(val):
        return 'BASE', False
    
    clean_val = str(val).strip().upper()
    
    if clean_val in ['LIBRE', 'L', 'DM', 'SG', 'SIN GOCE']:
        return 'L', False
    if clean_val in ['COMPENSADO', 'C']:
        return 'C', False
    if clean_val in ['LICENCIA', 'LIC']:
        return 'BASE', False
    if clean_val in ['VACACION', 'VACACIONES', 'V']:
        return 'BASE', False
    if clean_val in ['BASE', 'TURNO BASE']:
        return 'BASE', False

    clean_val = clean_val.replace(" ", "")
    
    if '-' in clean_val:
        parts = clean_val.split('-')
        if len(parts) == 2:
            def norm_hhmm(t):
                tp = t.split(':')
                if len(tp) == 2:
                    return f"{int(tp[0]):02d}:{int(tp[1]):02d}"
                return t
            p1, p2 = norm_hhmm(parts[0]), norm_hhmm(parts[1])
            key = f"{p1}-{p2}"
            if key in shift_map:
                return shift_map[key], False
            else:
                return clean_val, True

    return clean_val, True

def process_image_with_gemini(image_bytes, api_key, area_context=None):
    """Inferencia espacial con Gemini Vision Flash."""
    if not HAS_GEMINI:
        render_notice("error", "Dependencia faltante", "Libreria google-generativeai no instalada")
        return None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.5-flash')
    
    prompt = f"""
    Analiza esta imagen que contiene un cuadrante de horarios de personal.
    Contexto de área seleccionada: {area_context if area_context else 'Determinar por imagen'}.

    REGLAS DE DISTRIBUCION ESPACIAL PARA SALON:
    - En planillas de Salón, la tabla esta dividida horizontalmente en dos bloques:
      1. BLOQUE SUPERIOR: Cargo 'GARZON'.
      2. BLOQUE INFERIOR: Cargo 'RUNNER'.
    - Clasifica a cada colaborador segun su posicion vertical respecto a este eje divisor.

    REGLAS DE EXTRACCION:
    - Extrae SOLAMENTE el rango horario (ej. '19:00-02:30', '10:30-18:30'). Ignora horas totales.
    - Para ausencias o descansos usa 'LIBRE', 'DM', 'SG', 'LICENCIA' o 'COMPENSADO'.
    
    Retorna UNICAMENTE un arreglo JSON valido con esta estructura:
    [
      {{
        "nombre": "NOMBRE DETECTADO",
        "seccion": "SALON",
        "cargo": "GARZON",
        "lunes": "19:00-02:30",
        "martes": "LIBRE",
        "miercoles": "19:00-02:30",
        "jueves": "19:00-02:30",
        "viernes": "19:00-02:30",
        "sabado": "19:00-02:30",
        "domingo": "13:00-20:30"
      }}
    ]
    """
    
    try:
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_bytes}
        ])
        text_resp = response.text.strip()
        text_resp = re.sub(r'```json\s*', '', text_resp)
        text_resp = re.sub(r'```\s*$', '', text_resp)
        return json.loads(text_resp)
    except Exception as e:
        render_notice("error", "Falla de inferencia", str(e))
        return None

# -----------------------------------------------------------------------------
# BARRA LATERAL
# -----------------------------------------------------------------------------
st.sidebar.markdown("<span class='badge-mono'>ENTORNO Y CREDENCIALES</span>", unsafe_allow_html=True)
st.sidebar.markdown("### Configuración")

buk_file = st.sidebar.file_uploader(
    "Plantilla BUK Base (.xls / .xlsx)", 
    type=["xls", "xlsx"],
    help="Cargue el archivo consolidado ReporteTurnosColaboradores.xls"
)

secret_key = st.secrets.get("GEMINI_API_KEY", "")
if secret_key:
    gemini_api_key = secret_key
    render_notice("success", "Credencial Activa", "API Key cargada via secrets.toml")
else:
    gemini_api_key = st.sidebar.text_input(
        "API Key de Gemini",
        type="password",
        help="Ingrese clave si no está configurada en secrets"
    )

st.sidebar.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<span class='badge-mono'>CALENDARIO OPERATIVO</span>", unsafe_allow_html=True)
lunes_fecha = st.sidebar.date_input("Fecha Lunes de inicio")

dias_semana = [lunes_fecha + pd.Timedelta(days=i) for i in range(7)]
columnas_fechas_buk = [d.strftime("%d-%m-%Y") for d in dias_semana]
dias_keys = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

st.sidebar.markdown(f"""
<div style="background: var(--surface-base); border: 1px solid var(--border-subtle); border-radius: 6px; padding: 0.75rem; margin-top: 0.5rem;">
    <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">Semana BUK</div>
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--text-primary); margin-top: 0.2rem;">
        {columnas_fechas_buk[0]} &rarr; {columnas_fechas_buk[-1]}
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FLUJO PRINCIPAL
# -----------------------------------------------------------------------------
if buk_file is not None:
    try:
        xl = pd.ExcelFile(buk_file)
        tc_df = xl.parse('turnosColaboradores')
    except Exception as err:
        render_notice("error", "Error de lectura", "Verifique que el archivo sea un Excel valido o instale xlrd.")
        st.stop()
        
    shift_map, valid_siglas = build_shift_dictionary(buk_file)
    nombres_buk_oficiales = sorted(tc_df['Nombre del Colaborador'].dropna().unique().tolist())
    
    raw_schedules = st.session_state.get('extracted_schedules', [])
    
    render_telemetry(
        total_buk=len(tc_df),
        extraidos=len(raw_schedules),
        ambiguos=st.session_state.get('cant_ambiguos', 0),
        desconocidos=st.session_state.get('cant_desconocidos', 0)
    )
    
    st.markdown("<span class='badge-mono badge-accent'>CANAL DE INGRESO</span>", unsafe_allow_html=True)
    metodo = st.radio(
        label="Canal de entrada",
        options=["Imágenes de Cuadrantes (Vision IA)", "Planilla de Datos (Excel / CSV)"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if metodo == "Imágenes de Cuadrantes (Vision IA)":
        tab_barra, tab_salon, tab_cocina, tab_general = st.tabs([
            "Barra",
            "Salón",
            "Cocina / Producción",
            "Documentos Mixtos"
        ])

        with tab_barra:
            render_area_spec("Área Barra", "Asignacion directa a Bartenders.", "BARRA")
            uploaded_barra = st.file_uploader("Barra", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="up_barra", label_visibility="collapsed")

        with tab_salon:
            render_area_spec("Regla Espacial de Salón", "Division por franja: Garzones arriba, Runners abajo.", "SALON")
            tipo_carga_salon = st.radio("Modalidad de carga", ["Pliego completo (Deteccion espacial)", "Recortes separados"], horizontal=True)

            if tipo_carga_salon == "Pliego completo (Deteccion espacial)":
                uploaded_salon_completo = st.file_uploader("Salón completo", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="up_salon_comp", label_visibility="collapsed")
                uploaded_garzones, uploaded_runners = [], []
            else:
                col_g, col_r = st.columns(2)
                with col_g:
                    st.caption("Bloque Superior: Garzones")
                    uploaded_garzones = st.file_uploader("Garzones", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="up_garz", label_visibility="collapsed")
                with col_r:
                    st.caption("Bloque Inferior: Runners")
                    uploaded_runners = st.file_uploader("Runners", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="up_runn", label_visibility="collapsed")
                uploaded_salon_completo = []

            uploaded_salon = (uploaded_salon_completo or []) + (uploaded_garzones or []) + (uploaded_runners or [])

        with tab_cocina:
            render_area_spec("Área Cocina y Producción", "Caliente, Frio, Nikkei, Cooperia y Jefaturas.", "COCINA")
            uploaded_cocina = st.file_uploader("Cocina", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="up_cocina", label_visibility="collapsed")

        with tab_general:
            render_area_spec("Documentos Multizona", "Planillas con encabezados mixtos explicitos.", "MIXTO")
            uploaded_general = st.file_uploader("Mixtos", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="up_general", label_visibility="collapsed")

        lotes_carga = [
            (uploaded_barra, "BARRA", "BARTENDER"),
            (uploaded_cocina, "COCINA", "COCINA"),
            (uploaded_salon_completo, "SALON", None),
            (uploaded_garzones, "SALON", "GARZON"),
            (uploaded_runners, "SALON", "RUNNER"),
            (uploaded_general, None, None)
        ]

        conteos_por_area = {
            "Barra": len(uploaded_barra) if uploaded_barra else 0,
            "Salón": len(uploaded_salon) if uploaded_salon else 0,
            "Cocina": len(uploaded_cocina) if uploaded_cocina else 0,
            "Mixtos": len(uploaded_general) if uploaded_general else 0,
        }
        
        if sum(conteos_por_area.values()) > 0:
            render_queue_status(conteos_por_area)
            
            if not gemini_api_key:
                render_notice("warning", "Credencial Requerida", "Ingrese la clave API de Gemini para iniciar.")
            else:
                col_btn, _ = st.columns([1, 2])
                with col_btn:
                    if st.button("Iniciar Extracción Visual", use_container_width=True):
                        with st.spinner("Procesando matriz visual con reglas espaciales..."):
                            all_extracted = []
                            for archivos, area_f, cargo_f in lotes_carga:
                                if not archivos:
                                    continue
                                for img_file in archivos:
                                    img_bytes = img_file.read()
                                    extracted = process_image_with_gemini(img_bytes, gemini_api_key, area_context=area_f)
                                    if extracted:
                                        for item in extracted:
                                            if area_f:
                                                item["seccion"] = area_f
                                            if cargo_f:
                                                item["cargo"] = cargo_f
                                        all_extracted.extend(extracted)

                            st.session_state['extracted_schedules'] = all_extracted
                            st.rerun()

        if 'extracted_schedules' in st.session_state:
            raw_schedules = st.session_state['extracted_schedules']

    elif metodo == "Planilla de Datos (Excel / CSV)":
        uploaded_excel = st.file_uploader("Subir cuadrante (.xlsx / .csv)", type=["xlsx", "xls", "csv"])
        if uploaded_excel:
            df_in = pd.read_csv(uploaded_excel) if uploaded_excel.name.endswith('.csv') else pd.read_excel(uploaded_excel)
            st.dataframe(df_in.head(5), use_container_width=True)
            cols = df_in.columns.tolist()
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                c_nom = st.selectbox("Columna: Nombre", cols, index=0)
            with col_m2:
                has_sec = st.checkbox("Incluye columna de Sección")
                c_sec = st.selectbox("Columna: Sección", cols) if has_sec else None

            st.caption("Asignación de días")
            col_grid = st.columns(7)
            d_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
            c_dias = [col_grid[i].selectbox(d_names[i], cols, index=min(i + 1, len(cols) - 1), key=f"ec_{i}") for i in range(7)]
            
            if st.button("Procesar Datos Tabulares"):
                parsed_list = []
                for _, r in df_in.iterrows():
                    item = {"nombre": str(r[c_nom]).strip(), "seccion": str(r[c_sec]).strip() if c_sec else ""}
                    for idx_d, dk in enumerate(dias_keys):
                        item[dk] = str(r[c_dias[idx_d]]).strip()
                    parsed_list.append(item)
                st.session_state['extracted_schedules'] = parsed_list
                raw_schedules = parsed_list

    # -------------------------------------------------------------------------
    # HOMOLOGACIÓN Y RESOLUCIÓN DE AMBIGÜEDAD
    # -------------------------------------------------------------------------
    if raw_schedules:
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("<span class='badge-mono badge-accent'>AUDITORÍA DE ASIGNACIÓN</span>", unsafe_allow_html=True)
        
        mapped_entries = []
        unresolved_names = []
        
        for item in raw_schedules:
            raw_n = str(item.get("nombre", "")).strip().upper()
            sec_str = str(item.get("seccion", "")).strip()
            norm_a = normalize_area(sec_str)
            
            if not raw_n or raw_n == 'NAN':
                continue
            
            exact_match = None
            alias_target = ALIAS_PREDEFINIDOS.get(raw_n, None)
            if alias_target:
                matches_alias = tc_df[tc_df['Nombre del Colaborador'].str.upper() == alias_target.upper()]
                if not matches_alias.empty:
                    exact_match = matches_alias.iloc[0]['Nombre del Colaborador']
                else:
                    raw_n = alias_target.upper()
            
            if not exact_match:
                candidates_df = tc_df[tc_df['Nombre del Colaborador'].str.upper().str.contains(raw_n, regex=False)]
                if len(candidates_df) == 1:
                    exact_match = candidates_df.iloc[0]['Nombre del Colaborador']
                elif len(candidates_df) > 1:
                    if norm_a:
                        area_filtered = candidates_df[candidates_df['Área'].str.upper() == norm_a]
                        if len(area_filtered) == 1:
                            exact_match = area_filtered.iloc[0]['Nombre del Colaborador']
                        else:
                            unresolved_names.append((raw_n, sec_str, area_filtered['Nombre del Colaborador'].tolist() if not area_filtered.empty else candidates_df['Nombre del Colaborador'].tolist(), item))
                            continue
                    else:
                        unresolved_names.append((raw_n, sec_str, candidates_df['Nombre del Colaborador'].tolist(), item))
                        continue
                else:
                    fuzzy = difflib.get_close_matches(raw_n, [n.upper() for n in nombres_buk_oficiales], n=5, cutoff=0.4)
                    if fuzzy:
                        cand_df = tc_df[tc_df['Nombre del Colaborador'].str.upper().isin(fuzzy)]
                        if norm_a:
                            area_filtered = cand_df[cand_df['Área'].str.upper() == norm_a]
                            if len(area_filtered) == 1:
                                exact_match = area_filtered.iloc[0]['Nombre del Colaborador']
                            else:
                                unresolved_names.append((raw_n, sec_str, cand_df['Nombre del Colaborador'].tolist(), item))
                                continue
                        else:
                            unresolved_names.append((raw_n, sec_str, cand_df['Nombre del Colaborador'].tolist(), item))
                            continue
                    else:
                        unresolved_names.append((raw_n, sec_str, nombres_buk_oficiales, item))
                        continue

            if exact_match:
                mapped_entries.append((exact_match, item))

        st.session_state['cant_ambiguos'] = len(unresolved_names)
        resolved_ambiguous = {}
        
        if unresolved_names:
            render_notice("warning", "Discrepancia de Nombres", f"Se detectaron {len(unresolved_names)} nombres que requieren confirmación.")
            for idx_un, (raw_n, sec_s, candidates, item_obj) in enumerate(unresolved_names):
                col_i, col_s = st.columns([1, 2])
                with col_i:
                    st.markdown(f"<span class='badge-mono'>{sec_s or 'GENERAL'}</span> <strong>{raw_n}</strong>", unsafe_allow_html=True)
                with col_s:
                    sel = st.selectbox(f"Asignar {raw_n}", ["-- Omitir --"] + candidates, key=f"amb_{idx_un}", label_visibility="collapsed")
                    if sel != "-- Omitir --":
                        resolved_ambiguous[idx_un] = (sel, item_obj)

        for idx_un, (selected_name, item_obj) in resolved_ambiguous.items():
            mapped_entries.append((selected_name, item_obj))

        # ---------------------------------------------------------------------
        # VALIDACIÓN DE SIGLAS
        # ---------------------------------------------------------------------
        unknown_shifts_found = set()
        for buk_name, schedule_obj in mapped_entries:
            for dk in dias_keys:
                raw_time = schedule_obj.get(dk, "")
                sigla, is_unk = convert_time_to_buk_sigla(raw_time, shift_map)
                if is_unk and sigla not in valid_siglas:
                    unknown_shifts_found.add((raw_time, sigla))

        st.session_state['cant_desconocidos'] = len(unknown_shifts_found)
        sigla_replacements = {}
        
        if unknown_shifts_found:
            render_notice("error", "Horarios no Catalogados", f"Se encontraron {len(unknown_shifts_found)} horarios sin equivalencia en turnosSemanales.")
            for raw_t, estimated_sig in unknown_shifts_found:
                col_t1, col_t2 = st.columns([1, 2])
                with col_t1:
                    st.markdown(f"<span class='badge-mono'>{raw_t}</span>", unsafe_allow_html=True)
                with col_t2:
                    matches = difflib.get_close_matches(estimated_sig, valid_siglas, n=1)
                    def_idx = valid_siglas.index(matches[0]) if matches else 0
                    chosen = st.selectbox(f"Reemplazo {raw_t}", valid_siglas, index=def_idx, key=f"unk_{raw_t}", label_visibility="collapsed")
                    sigla_replacements[raw_t] = chosen

        # ---------------------------------------------------------------------
        # VISTA PREVIA Y AUDITORÍA POR ÁREA (BARRA / SALÓN / COCINA)
        # -----------------------------------------------------------------------------
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("<span class='badge-mono badge-accent'>CONTROL DE CALIDAD PRE-EXPORTACIÓN</span>", unsafe_allow_html=True)
        st.markdown("### Vista Previa y Verificación por Área")

        # Construcción reactiva del dataset consolidado
        df_staging = tc_df.copy()
        for f_col in columnas_fechas_buk:
            if f_col not in df_staging.columns:
                df_staging[f_col] = 'BASE'

        assigned_names_set = set()
        for buk_name, schedule_obj in mapped_entries:
            if buk_name in df_staging['Nombre del Colaborador'].values:
                assigned_names_set.add(buk_name)
                idx = df_staging[df_staging['Nombre del Colaborador'] == buk_name].index[0]
                for idx_d, dk in enumerate(dias_keys):
                    raw_time = schedule_obj.get(dk, "")
                    sigla, is_unk = convert_time_to_buk_sigla(raw_time, shift_map)
                    sigla = sigla_replacements.get(raw_time, sigla_replacements.get(sigla, sigla))
                    df_staging.at[idx, columnas_fechas_buk[idx_d]] = sigla

        # Segmentador y Controles de Filtro
        col_filtro_area, col_filtro_modo = st.columns([2, 2])
        with col_filtro_area:
            areas_disponibles = ["TODAS"] + sorted([str(a) for a in df_staging['Área'].dropna().unique() if str(a).strip() != ""])
            area_seleccionada = st.selectbox("Filtrar por Área operativa:", areas_disponibles, index=0)
        
        with col_filtro_modo:
            solo_asignados = st.toggle("Mostrar únicamente colaboradores con turnos asignados", value=False)

        # Aplicación del filtrado
        df_display = df_staging.copy()
        if area_seleccionada != "TODAS":
            df_display = df_display[df_display['Área'].str.upper() == area_seleccionada.upper()]

        total_area = len(df_display)
        con_turno_area = sum(df_display['Nombre del Colaborador'].isin(assigned_names_set))
        turnos_base_area = total_area - con_turno_area

        if solo_asignados:
            df_display = df_display[df_display['Nombre del Colaborador'].isin(assigned_names_set)]

        # Render de métricas de cobertura
        render_preview_metrics(
            area_label=area_seleccionada,
            total_colabs=total_area,
            con_turno=con_turno_area,
            turnos_base=turnos_base_area
        )

        columnas_a_mostrar = ['Nombre del Colaborador', 'Área']
        if 'Cargo' in df_display.columns:
            columnas_a_mostrar.append('Cargo')
        columnas_a_mostrar += columnas_fechas_buk

        st.dataframe(
            df_display[columnas_a_mostrar],
            use_container_width=True,
            hide_index=True
        )

        # ---------------------------------------------------------------------
        # CONSOLIDACIÓN Y DESCARGA
        # ---------------------------------------------------------------------
        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
        col_act, _ = st.columns([1, 2])
        with col_act:
            btn_generar = st.button("Consolidar Matriz BUK Final", use_container_width=True)
            
        if btn_generar:
            render_notice("success", "Compilación Exitosa", "Planilla validada y lista para descarga.")

            out = io.BytesIO()
            with pd.ExcelWriter(out, engine='openpyxl') as writer:
                df_staging.to_excel(writer, sheet_name='turnosColaboradores', index=False)
                for sheet_n in xl.sheet_names:
                    if sheet_n != 'turnosColaboradores':
                        xl.parse(sheet_n).to_excel(writer, sheet_name=sheet_n, index=False)

            col_down, _ = st.columns([1, 2])
            with col_down:
                st.download_button(
                    label="Descargar Reporte BUK (.xlsx)",
                    data=out.getvalue(),
                    file_name=f"ReporteTurnosColaboradores_{columnas_fechas_buk[0]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
else:
    render_notice("warning", "Esperando archivo base", "Suba la plantilla ReporteTurnosColaboradores.xls en el panel lateral.")