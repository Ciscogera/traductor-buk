import streamlit as st
import pandas as pd
import io
import re
import json
import time
import difflib

from utils.ui import (
    load_theme,
    render_ops_header,
    render_bento_telemetry,
    render_area_banner,
    render_shift_audit_table
)

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y CARGA DE TEMA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Traductor de Horarios BUK - Bar El Bajo",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_theme("styles/theme.css")

render_ops_header(
    title="Carga Masiva de Horarios BUK",
    subtitle="Procesamiento automatizado de cuadrantes semanales y homologación de turnos",
    badge_text="Bar El Bajo · Ops"
)

# -----------------------------------------------------------------------------
# DICCIONARIOS DE HOMOLOGACIÓN Y ALIAS PREDEFINIDOS
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
# FUNCIONES DE NEGOCIO Y PROCESAMIENTO
# -----------------------------------------------------------------------------
def normalize_area(section_str):
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
        st.error(f"Error al leer la hoja 'turnosSemanales': {e}")
        return {}, []

def convert_time_to_buk_sigla(val, shift_map):
    if pd.isna(val):
        return 'BASE', False
    
    clean_val = str(val).strip().upper()
    
    if clean_val in ['LIBRE', 'L', 'DM', 'SG', 'SIN GOCE']:
        return 'L', False
    if clean_val in ['COMPENSADO', 'C']:
        return 'C', False
    if clean_val in ['LICENCIA', 'LIC', 'VACACION', 'VACACIONES', 'V', 'BASE', 'TURNO BASE']:
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
            return clean_val, True

    return clean_val, True

def process_image_with_gemini(image_bytes, mime_type, api_key, area_context=None):
    if not HAS_GEMINI:
        return None, "Librería google-generativeai no instalada."
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name='gemini-3.5-flash-lite',
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.1
        }
    )
    
    prompt = f"""
    Analiza esta imagen que contiene un cuadrante de horarios de personal.
    Contexto de área seleccionada por el usuario: {area_context if area_context else 'No especificado (determinar por imagen)'}.

    REGLAS DE DISTRIBUCION ESPACIAL PARA SALON:
    - En planillas de Salón, la tabla está dividida horizontalmente en dos bloques principales por una franja o eje divisor:
      1. BLOQUE SUPERIOR (arriba del eje): Corresponde estrictamente al cargo "GARZON".
      2. BLOQUE INFERIOR (abajo del eje): Corresponde estrictamente al cargo "RUNNER".
    - Aunque la imagen no tenga escrita la palabra "GARZON" o "RUNNER" como encabezado, clasifica a cada colaborador según su posición vertical respecto a este eje.

    REGLAS DE EXTRACCION:
    - Extrae SOLAMENTE el rango horario (ej. '19:00-02:30', '10:30-18:30'). Ignora por completo el total de horas que aparece debajo (ignora '7', '8', '7,5', etc.).
    - Para descansos o ausencias escribe 'LIBRE', 'DM', 'SG', 'LICENCIA' o 'COMPENSADO'.
    
    Retorna UNICAMENTE un arreglo JSON válido con esta estructura:
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
            {"mime_type": mime_type or "image/jpeg", "data": image_bytes}
        ])
        text_resp = response.text.strip()
        text_resp = re.sub(r'^```json\s*', '', text_resp)
        text_resp = re.sub(r'\s*```$', '', text_resp)
        return json.loads(text_resp), None
    except Exception as e:
        return None, str(e)

# -----------------------------------------------------------------------------
# PANEL LATERAL DE CONFIGURACIÓN
# -----------------------------------------------------------------------------
st.sidebar.header("Configuración Base")

buk_file = st.sidebar.file_uploader(
    "1. Plantilla BUK Base (.xls / .xlsx)", 
    type=["xls", "xlsx"],
    help="Sube el archivo 'ReporteTurnosColaboradores.xls' de BUK"
)

secret_key = st.secrets.get("GEMINI_API_KEY", "")
if secret_key:
    gemini_api_key = secret_key
    st.sidebar.success("API Key activa desde secretos")
else:
    gemini_api_key = st.sidebar.text_input(
        "2. API Key de Gemini",
        type="password",
        help="Ingresa tu clave si no está configurada en secrets.toml"
    )

st.sidebar.subheader("3. Rango Semanal")
lunes_fecha = st.sidebar.date_input("Lunes de inicio de semana")

dias_semana = [lunes_fecha + pd.Timedelta(days=i) for i in range(7)]
columnas_fechas_buk = [d.strftime("%d-%m-%Y") for d in dias_semana]
dias_keys = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

st.sidebar.info(f"Semana BUK: {columnas_fechas_buk[0]} al {columnas_fechas_buk[-1]}")

# -----------------------------------------------------------------------------
# FLUJO PRINCIPAL
# -----------------------------------------------------------------------------
if buk_file is not None:
    try:
        xl = pd.ExcelFile(buk_file)
        tc_df = xl.parse('turnosColaboradores')
    except Exception:
        st.error("Error leyendo archivo .xls. Instala 'xlrd' o convierte la plantilla a .xlsx.")
        st.stop()
        
    shift_map, valid_siglas = build_shift_dictionary(buk_file)
    nombres_buk_oficiales = sorted(tc_df['Nombre del Colaborador'].dropna().unique().tolist())
    
    # Telemetría inicial desde el estado de sesión
    raw_schedules = st.session_state.get('extracted_schedules', [])
    render_bento_telemetry(
        total_dotacion=len(tc_df),
        asignados=len(raw_schedules),
        ambiguos=st.session_state.get('count_ambiguos', 0),
        siglas_unk=st.session_state.get('count_siglas_unk', 0)
    )
    
    st.subheader("Entrada de Horarios")
    metodo = st.radio(
        "Fuente de los horarios:",
        ["Imágenes de Cuadrantes (Visión IA)", "Excel / CSV del Cuadrante"],
        horizontal=True
    )
    
    if metodo == "Imágenes de Cuadrantes (Visión IA)":
        tab_barra, tab_salon, tab_cocina, tab_general = st.tabs([
            "Barra",
            "Salón (Garzones / Runners)",
            "Cocina / Producción",
            "General (Con encabezados)"
        ])

        with tab_barra:
            render_area_banner(
                "Cuadrantes de Barra",
                "Asignación directa a Bartenders y resolución automática de homónimos de barra.",
                "AREA: BARRA"
            )
            uploaded_barra = st.file_uploader(
                "Imágenes de Barra",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="up_barra",
                label_visibility="collapsed"
            )

        with tab_salon:
            render_area_banner(
                "Regla Espacial de Salón",
                "Bloque superior asignado a Garzones; bloque inferior asignado a Runners.",
                "AREA: SALON"
            )
            tipo_carga_salon = st.radio(
                "Modalidad de imagen para Salón:",
                ["Imagen completa de Salón (Bloque superior Garzones / Inferior Runners)", "Imágenes recortadas por cargo"],
                horizontal=True
            )

            if tipo_carga_salon == "Imagen completa de Salón (Bloque superior Garzones / Inferior Runners)":
                uploaded_salon_completo = st.file_uploader(
                    "Subir imagen completa de Salón",
                    type=["jpg", "jpeg", "png"],
                    accept_multiple_files=True,
                    key="up_salon_comp",
                    label_visibility="collapsed"
                )
                uploaded_garzones, uploaded_runners = [], []
            else:
                col_g, col_r = st.columns(2)
                with col_g:
                    uploaded_garzones = st.file_uploader(
                        "Bloque Garzones (Superior)",
                        type=["jpg", "jpeg", "png"],
                        accept_multiple_files=True,
                        key="up_garz"
                    )
                with col_r:
                    uploaded_runners = st.file_uploader(
                        "Bloque Runners (Inferior)",
                        type=["jpg", "jpeg", "png"],
                        accept_multiple_files=True,
                        key="up_runn"
                    )
                uploaded_salon_completo = []

        with tab_cocina:
            render_area_banner(
                "Cuadrantes de Cocina y Producción",
                "Incluye línea caliente, frío, nikkei, producción, coopería y jefaturas.",
                "AREA: COCINA / PRODUCCION"
            )
            uploaded_cocina = st.file_uploader(
                "Imágenes de Cocina",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="up_cocina",
                label_visibility="collapsed"
            )

        with tab_general:
            render_area_banner(
                "Documentos Mixtos",
                "Planillas que incluyen encabezados explícitos visibles en la imagen.",
                "DETECCION MIXTA"
            )
            uploaded_general = st.file_uploader(
                "Imágenes Mixtas",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="up_general",
                label_visibility="collapsed"
            )

        lotes_carga = [
            (uploaded_barra, "BARRA", "BARTENDER"),
            (uploaded_cocina, "COCINA", "COCINA"),
            (uploaded_salon_completo, "SALON", None),
            (uploaded_garzones, "SALON", "GARZON"),
            (uploaded_runners, "SALON", "RUNNER"),
            (uploaded_general, None, None)
        ]

        total_archivos = sum(len(archivos) for archivos, _, _ in lotes_carga if archivos)

        if total_archivos > 0:
            if not gemini_api_key:
                st.warning("Ingresa tu API Key de Gemini en el panel lateral.")
            else:
                if st.button("Leer y Analizar Todas las Imágenes"):
                    with st.spinner("Procesando imágenes con IA Vision..."):
                        all_extracted = []
                        extraction_logs = []
                        
                        for archivos, area_f, cargo_f in lotes_carga:
                            if not archivos:
                                continue
                            for img_file in archivos:
                                img_bytes = img_file.getvalue()
                                extracted, err_msg = process_image_with_gemini(
                                    image_bytes=img_bytes,
                                    mime_type=img_file.type,
                                    api_key=gemini_api_key,
                                    area_context=area_f
                                )
                                
                                area_label = area_f if area_f else "MIXTO"
                                if err_msg:
                                    extraction_logs.append({
                                        "archivo": img_file.name,
                                        "area": area_label,
                                        "ok": False,
                                        "detalle": err_msg
                                    })
                                elif extracted:
                                    for item in extracted:
                                        if area_f:
                                            item["seccion"] = area_f
                                        if cargo_f:
                                            item["cargo"] = cargo_f
                                    all_extracted.extend(extracted)
                                    extraction_logs.append({
                                        "archivo": img_file.name,
                                        "area": area_label,
                                        "ok": True,
                                        "detalle": f"{len(extracted)} colaboradores detectados"
                                    })
                                else:
                                    extraction_logs.append({
                                        "archivo": img_file.name,
                                        "area": area_label,
                                        "ok": False,
                                        "detalle": "La IA devolvió una lista vacía"
                                    })
                                
                                time.sleep(1.5)

                        st.session_state['extracted_schedules'] = all_extracted
                        st.session_state['extraction_logs'] = extraction_logs
                        st.rerun()

        if 'extraction_logs' in st.session_state and st.session_state['extraction_logs']:
            for log in st.session_state['extraction_logs']:
                if log["ok"]:
                    st.success(f"[{log['area']}] {log['archivo']}: {log['detalle']}")
                else:
                    st.error(f"[{log['area']}] {log['archivo']} falló: {log['detalle']}")

        if 'extracted_schedules' in st.session_state:
            raw_schedules = st.session_state['extracted_schedules']

    else:
        uploaded_excel = st.file_uploader("Sube la planilla Excel o CSV del cuadrante semanal", type=["xlsx", "xls", "csv"])
        if uploaded_excel:
            df_in = pd.read_csv(uploaded_excel) if uploaded_excel.name.endswith('.csv') else pd.read_excel(uploaded_excel)
            st.dataframe(df_in.head(5), use_container_width=True)
            cols = df_in.columns.tolist()
            c_nom = st.selectbox("Columna de Nombre del Trabajador:", cols, index=0)
            
            has_sec_col = st.checkbox("¿El archivo incluye columna de Área/Sección?")
            c_sec = st.selectbox("Columna de Área/Sección:", cols) if has_sec_col else None

            c_dias = []
            col_grid = st.columns(7)
            d_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
            for i in range(7):
                def_idx = min(i + 1, len(cols) - 1)
                cd = col_grid[i].selectbox(d_names[i], cols, index=def_idx, key=f"e_col_{i}")
                c_dias.append(cd)
            
            if st.button("Procesar Cuadrante"):
                parsed_list = []
                for _, r in df_in.iterrows():
                    item = {
                        "nombre": str(r[c_nom]).strip(),
                        "seccion": str(r[c_sec]).strip() if c_sec else ""
                    }
                    for idx_d, dk in enumerate(dias_keys):
                        item[dk] = str(r[c_dias[idx_d]]).strip()
                    parsed_list.append(item)
                st.session_state['extracted_schedules'] = parsed_list
                raw_schedules = parsed_list

    # -------------------------------------------------------------------------
    # HOMOLOGACIÓN E INSPECCIÓN DE DATOS
    # -------------------------------------------------------------------------
    if raw_schedules:
        st.divider()
        st.subheader("Homologación e Inspección de Datos")
        
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
                        if norm_a == 'COCINA':
                            area_filtered = candidates_df[candidates_df['Área'].str.upper().isin(['COCINA', 'PRODUCCION', 'PRODUCCIÓN'])]
                        else:
                            area_filtered = candidates_df[candidates_df['Área'].str.upper() == norm_a]
                            
                        if len(area_filtered) == 1:
                            exact_match = area_filtered.iloc[0]['Nombre del Colaborador']
                        elif len(area_filtered) > 1:
                            unresolved_names.append((raw_n, sec_str, area_filtered['Nombre del Colaborador'].tolist(), item))
                            continue
                        else:
                            unresolved_names.append((raw_n, sec_str, candidates_df['Nombre del Colaborador'].tolist(), item))
                            continue
                    else:
                        unresolved_names.append((raw_n, sec_str, candidates_df['Nombre del Colaborador'].tolist(), item))
                        continue
                else:
                    fuzzy = difflib.get_close_matches(raw_n, [n.upper() for n in nombres_buk_oficiales], n=5, cutoff=0.4)
                    if fuzzy:
                        cand_df = tc_df[tc_df['Nombre del Colaborador'].str.upper().isin(fuzzy)]
                        if norm_a:
                            if norm_a == 'COCINA':
                                area_filtered = cand_df[cand_df['Área'].str.upper().isin(['COCINA', 'PRODUCCION', 'PRODUCCIÓN'])]
                            else:
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

        st.session_state['count_ambiguos'] = len(unresolved_names)

        resolved_ambiguous = {}
        if unresolved_names:
            st.warning("Se detectaron nombres ambiguos o duplicados en distintas áreas. Selecciona la persona correcta de BUK:")
            for idx_un, (raw_n, sec_s, candidates, item_obj) in enumerate(unresolved_names):
                col_a, col_b = st.columns([1, 2])
                sec_disp = f" (Sección: {sec_s})" if sec_s else ""
                col_a.write(f"**Nombre detectado:** `{raw_n}`{sec_disp}")
                sel_buk = col_b.selectbox(
                    "Asignar a colaborador BUK:",
                    options=["-- Ignorar --"] + candidates,
                    key=f"amb_name_{idx_un}"
                )
                if sel_buk != "-- Ignorar --":
                    resolved_ambiguous[idx_un] = (sel_buk, item_obj)

        for idx_un, (selected_buk_name, item_obj) in resolved_ambiguous.items():
            mapped_entries.append((selected_buk_name, item_obj))

        # ---------------------------------------------------------------------
        # VALIDACIÓN DE SIGLAS
        # ---------------------------------------------------------------------
        st.subheader("Validación de Siglas y Horarios")
        
        unknown_shifts_found = set()
        for buk_name, schedule_obj in mapped_entries:
            for dk in dias_keys:
                raw_time = schedule_obj.get(dk, "")
                sigla, is_unk = convert_time_to_buk_sigla(raw_time, shift_map)
                if is_unk and sigla not in valid_siglas:
                    unknown_shifts_found.add((raw_time, sigla))

        st.session_state['count_siglas_unk'] = len(unknown_shifts_found)

        sigla_replacements = {}
        if unknown_shifts_found:
            st.error("Se detectaron horarios no registrados en la tabla oficial BUK (`turnosSemanales`). Selecciona la sigla oficial correspondiente:")
            for raw_t, estimated_sig in unknown_shifts_found:
                col_s1, col_s2 = st.columns([1, 2])
                col_s1.write(f"Horario no registrado: `{raw_t}`")
                
                closest_matches = difflib.get_close_matches(estimated_sig, valid_siglas, n=1)
                default_idx = valid_siglas.index(closest_matches[0]) if closest_matches else 0
                
                chosen_sig = col_s2.selectbox(
                    f"Seleccionar Sigla BUK para '{raw_t}':",
                    options=valid_siglas,
                    index=default_idx,
                    key=f"unk_shift_{raw_t}"
                )
                sigla_replacements[raw_t] = chosen_sig

        assigned_names = [e[0] for e in mapped_entries]
        duplicates = [name for name in set(assigned_names) if assigned_names.count(name) > 1]
        if duplicates:
            st.error(f"Alerta de Duplicados: Hay colaboradores asignados más de una vez: {', '.join(duplicates)}.")

        # ---------------------------------------------------------------------
        # GENERACIÓN Y AUDITORÍA VISUAL DE PLANILLA BUK
        # ---------------------------------------------------------------------
        st.divider()
        if st.button("Generar y Exportar Planilla BUK Final"):
            resultado_buk = tc_df.copy()
            
            for fecha_col in columnas_fechas_buk:
                if fecha_col not in resultado_buk.columns:
                    resultado_buk[fecha_col] = 'BASE'

            for buk_name, schedule_obj in mapped_entries:
                if buk_name in resultado_buk['Nombre del Colaborador'].values:
                    idx = resultado_buk[resultado_buk['Nombre del Colaborador'] == buk_name].index[0]
                    for idx_d, dk in enumerate(dias_keys):
                        raw_time = schedule_obj.get(dk, "")
                        sigla, is_unk = convert_time_to_buk_sigla(raw_time, shift_map)
                        
                        if raw_time in sigla_replacements:
                            sigla = sigla_replacements[raw_time]
                        elif sigla in sigla_replacements:
                            sigla = sigla_replacements[sigla]
                            
                        resultado_buk.at[idx, columnas_fechas_buk[idx_d]] = sigla

            st.success("Planilla generada con éxito.")
            
            # Renderizado de tabla nocturna en vidrio esmerilado con badges por turno
            render_shift_audit_table(resultado_buk, columnas_fechas_buk)

            output_bytes = io.BytesIO()
            with pd.ExcelWriter(output_bytes, engine='openpyxl') as writer:
                resultado_buk.to_excel(writer, sheet_name='turnosColaboradores', index=False)
                for sheet_n in xl.sheet_names:
                    if sheet_n != 'turnosColaboradores':
                        xl.parse(sheet_n).to_excel(writer, sheet_name=sheet_n, index=False)

            st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="Descargar ReporteTurnosColaboradores.xlsx para BUK",
                data=output_bytes.getvalue(),
                file_name=f"ReporteTurnosColaboradores_{columnas_fechas_buk[0]}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

else:
    st.info("Sube el archivo base `ReporteTurnosColaboradores.xls` en el panel lateral para iniciar.")