import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import json
import difflib

# Cargar librería de Gemini si está disponible
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Traductor de Horarios BUK - Bar El Bajo",
    page_icon="🍹",
    layout="wide"
)

st.title("🍹 Carga Masiva de Horarios a BUK - Bar El Bajo")
st.markdown("""
Esta aplicación procesa imágenes de los cuadrantes semanales de turnos (vía **IA Vision**) 
o archivos de datos, traduciéndolos al **formato exacto exigido por BUK** (`ReporteTurnosColaboradores.xls`).
""")

# -----------------------------------------------------------------------------
# DICCIONARIOS DE HOMOLOGACIÓN Y ALIAS PREDEFINIDOS
# (Nota: Se omiten nombres ambiguos de pila única como JESUS, DIEGO, CARLOS, etc.
#  para que la app use el filtro de Área o pida confirmación mediante menú desplegable).
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
    "MICHELL": "DENISS MICHAEL LABAN AREVALO",
    "ROXANA": "ROXANA XIOMARA VILLACRESES ALBAN",
    "NICOLAS": "NICOLAS ALEJANDRO TORO TOBAR",
    "JORGE": "JORGE ANDRÉS DELGADO SEGUEL",
    "DYLAN": "DYLAN MÉNDEZ DINAMARCA",
    "GASPAR": "GASPAR SEBASTIÁN IGNACIO SILVA MARTÍNEZ"
}

# -----------------------------------------------------------------------------
# FUNCIONES AUXILIARES DE CONVERSIÓN Y PROCESAMIENTO
# -----------------------------------------------------------------------------

def normalize_area(section_str):
    """Mapea el nombre de sección/área del cuadrante al Área oficial de BUK (BARRA, SALON, COCINA)."""
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
    """Extrae el mapeo (Entrada-Salida -> Sigla) y la lista de siglas válidas desde turnosSemanales."""
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
    """
    Convierte un valor de horario/texto a Sigla BUK.
    Retorna (sigla, es_desconocido)
    """
    if pd.isna(val):
        return 'BASE', False
    
    clean_val = str(val).strip().upper()
    
    # Reglas de Negocio Específicas
    if clean_val in ['LIBRE', 'L', 'DM', 'SG', 'SIN GOCE']:
        return 'L', False  # Libres, DM y SG quedan como 'L'
    if clean_val in ['COMPENSADO', 'C']:
        return 'C', False  # Compensado queda como 'C'
    if clean_val in ['LICENCIA', 'LIC']:
        return 'BASE', False  # Licencia previamente ingresada en BUK
    if clean_val in ['VACACION', 'VACACIONES', 'V']:
        return 'BASE', False  # Vacaciones previamente ingresadas en BUK
    if clean_val in ['BASE', 'TURNO BASE']:
        return 'BASE', False

    # Limpiar espacios internos (ej: "10:30 - 18:30" -> "10:30-18:30")
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

def process_image_with_gemini(image_bytes, api_key):
    """Envía la imagen a Gemini Vision API y retorna una lista de dicts estructurados incluyendo la sección/área."""
    if not HAS_GEMINI:
        st.error("Librería `google-generativeai` no instalada. Instálala con: `pip install google-generativeai`")
        return None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.6-flash')
    
    prompt = """
    Analiza esta imagen que contiene un cuadrante de horarios de personal de restaurant/bar.
    Extrae la lista de todos los colaboradores con sus respectivos turnos de Lunes a Domingo y la sección/área en la que están ubicados.
    
    Retorna ÚNICAMENTE un objeto JSON válido con la siguiente estructura (sin Markdown de código adicional):
    [
      {
        "nombre": "JESUS",
        "seccion": "GARZONES",
        "lunes": "19:00-02:30",
        "martes": "LIBRE",
        "miercoles": "19:00-02:30",
        "jueves": "19:00-02:30",
        "viernes": "19:00-02:30",
        "sabado": "19:00-02:30",
        "domingo": "13:00-20:30"
      }
    ]

    Reglas de extracción:
    - Identifica la sección o subtítulo donde aparece el trabajador (ej. "BARTENDERS", "GARZONES", "RUNNER", "NIKKEI", "FRIO", "PIZZAS", "CALIENTE", "PRODUCCION", "COOPERIA").
    - Conserva las horas de entrada y salida exactas ej "19:00-02:30", "09:30-18:00", "18:30-02:30", "10:30 - 18:30".
    - Para días sin turno o descansos escribe "LIBRE", "DM", "SG", "LICENCIA" o "COMPENSADO" según corresponda.
    - Captura a todos los trabajadores listados en las imágenes.
    """
    
    try:
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_bytes}
        ])
        text_resp = response.text.strip()
        text_resp = re.sub(r'```json\s*', '', text_resp)
        text_resp = re.sub(r'```\s*$', '', text_resp)
        parsed_data = json.loads(text_resp)
        return parsed_data
    except Exception as e:
        st.error(f"Error procesando la imagen con IA Vision: {e}")
        return None

# -----------------------------------------------------------------------------
# PANEL LATERAL DE CONFIGURACIÓN
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuración")

# 1. Archivo Plantilla BUK
buk_file = st.sidebar.file_uploader(
    "1. Plantilla BUK Base (.xls / .xlsx)", 
    type=["xls", "xlsx"],
    help="Sube el archivo 'ReporteTurnosColaboradores.xls' de BUK"
)

# 2. Clave API de Gemini (Manejo Seguro de Secretos)
secret_key = st.secrets.get("GEMINI_API_KEY", "")

if secret_key:
    # Si la clave existe en secretos, se usa internamente sin mostrar el texto en pantalla
    gemini_api_key = secret_key
    st.sidebar.success("🔑 API Key activa (Cargada de forma segura)")
else:
    # Solo si NO hay clave en secrets.toml, se muestra el cuadro para ingresarla manualmente
    gemini_api_key = st.sidebar.text_input(
        "2. API Key de Gemini (para Imágenes)",
        type="password",
        help="Ingresa tu clave si no la tienes configurada en secrets.toml"
    )

# 3. Selección de la Fecha del Lunes
st.sidebar.subheader("3. Rango de Fechas")
lunes_fecha = st.sidebar.date_input("Selecciona el Lunes de inicio de la semana")

# Calcular los 7 días de la semana
dias_semana = [lunes_fecha + pd.Timedelta(days=i) for i in range(7)]
columnas_fechas_buk = [d.strftime("%d-%m-%Y") for d in dias_semana]
dias_keys = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

st.sidebar.info(f"Semana BUK:\n**{columnas_fechas_buk[0]}** al **{columnas_fechas_buk[-1]}**")

# -----------------------------------------------------------------------------
# FLUJO PRINCIPAL DE LA APLICACIÓN
# -----------------------------------------------------------------------------
if buk_file is not None:
    try:
        xl = pd.ExcelFile(buk_file)
        tc_df = xl.parse('turnosColaboradores')
    except Exception as err:
        st.error("⚠️ Si obtienes un error con archivos .xls, instala 'xlrd' en tu entorno con `pip install xlrd` o convierte el archivo a .xlsx")
        st.stop()
        
    shift_map, valid_siglas = build_shift_dictionary(buk_file)
    nombres_buk_oficiales = sorted(tc_df['Nombre del Colaborador'].dropna().unique().tolist())
    
    st.success(f"Plantilla BUK cargada correctamente: **{len(tc_df)} colaboradores** en base de datos.")
    
    st.subheader("📸 Entrada de Horarios (Opción A: Imágenes / Opción B: Excel)")
    
    metodo = st.radio("Fuente de los horarios:", ["Imágenes de Cuadrantes (Visión IA)", "Excel / CSV del Cuadrante"])
    
    raw_schedules = []
    
    if metodo == "Imágenes de Cuadrantes (Visión IA)":
        uploaded_images = st.file_uploader("Sube una o varias imágenes de los horarios (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if uploaded_images:
            if not gemini_api_key:
                st.warning("⚠️ Ingresa tu API Key de Gemini en el panel lateral para procesar las imágenes.")
            else:
                if st.button("🔍 Leer y Analizar Imágenes con IA"):
                    with st.spinner("Procesando imágenes con IA Vision..."):
                        all_extracted = []
                        for img_file in uploaded_images:
                            img_bytes = img_file.read()
                            extracted = process_image_with_gemini(img_bytes, gemini_api_key)
                            if extracted:
                                all_extracted.extend(extracted)
                        st.session_state['extracted_schedules'] = all_extracted
                        st.success(f"¡Extracción exitosa! Se detectaron {len(all_extracted)} trabajadores.")

        if 'extracted_schedules' in st.session_state:
            raw_schedules = st.session_state['extracted_schedules']

    else:
        uploaded_excel = st.file_uploader("Sube la planilla Excel o CSV del cuadrante semanal", type=["xlsx", "xls", "csv"])
        if uploaded_excel:
            if uploaded_excel.name.endswith('.csv'):
                df_in = pd.read_csv(uploaded_excel)
            else:
                df_in = pd.read_excel(uploaded_excel)
            
            st.dataframe(df_in.head(5), use_container_width=True)
            cols = df_in.columns.tolist()
            c_nom = st.selectbox("Columna de Nombre del Trabajador:", cols, index=0)
            
            has_sec_col = st.checkbox("¿El archivo incluye columna de Área/Sección?")
            c_sec = None
            if has_sec_col:
                c_sec = st.selectbox("Columna de Área/Sección:", cols)

            st.markdown("**Asignación de columnas por día:**")
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
    # HOMOLOGACIÓN INTELIGENTE CON FILTRO POR ÁREA Y DETECCIÓN DE AMBIGÜEDAD
    # -------------------------------------------------------------------------
    if raw_schedules:
        st.divider()
        st.subheader("🛠️ Homologación e Inspección de Datos Extraídos")
        
        mapped_entries = [] # tuples of (nombre_buk_oficial, schedule_item_dict)
        unresolved_names = [] # list of (raw_n, section_str, candidate_list, schedule_item_dict)
        
        for item in raw_schedules:
            raw_n = str(item.get("nombre", "")).strip().upper()
            sec_str = str(item.get("seccion", "")).strip()
            norm_a = normalize_area(sec_str)
            
            if not raw_n or raw_n == 'NAN':
                continue
            
            # 1. Alias predefinido (si existe en ALIAS_PREDEFINIDOS)
            exact_match = ALIAS_PREDEFINIDOS.get(raw_n, None)
            
            if not exact_match:
                # 2. Buscar candidatos en BUK que contengan la subcadena
                candidates_df = tc_df[tc_df['Nombre del Colaborador'].str.upper().str.contains(raw_n, regex=False)]
                
                if len(candidates_df) == 1:
                    exact_match = candidates_df.iloc[0]['Nombre del Colaborador']
                elif len(candidates_df) > 1:
                    # Aplicar filtro por Área
                    if norm_a:
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
                    # 3. Coincidencia difusa (Fuzzy Match)
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

        # Interfaz interactiva para resolver nombres ambiguos o múltiples matches
        resolved_ambiguous = {}
        if unresolved_names:
            st.warning("⚠️ Se detectaron nombres ambiguos o duplicados en distintas áreas. Selecciona la persona correcta de BUK:")
            for idx_un, (raw_n, sec_s, candidates, item_obj) in enumerate(unresolved_names):
                col_a, col_b = st.columns([1, 2])
                sec_disp = f" (Sección: {sec_s})" if sec_s else ""
                col_a.write(f"**Nombre detectado:** `{raw_n}`{sec_disp}")
                sel_buk = col_b.selectbox(
                    f"Asignar a colaborador BUK:",
                    options=["-- Ignorar --"] + candidates,
                    key=f"amb_name_{idx_un}"
                )
                if sel_buk != "-- Ignorar --":
                    resolved_ambiguous[idx_un] = (sel_buk, item_obj)

        for idx_un, (selected_buk_name, item_obj) in resolved_ambiguous.items():
            mapped_entries.append((selected_buk_name, item_obj))

        # ---------------------------------------------------------------------
        # VALIDACIÓN Y ALERTAS DE HORARIOS Y SIGLAS
        # ---------------------------------------------------------------------
        st.subheader("⏱️ Validación de Siglas y Horarios de Turno")
        
        unknown_shifts_found = set()
        for buk_name, schedule_obj in mapped_entries:
            for dk in dias_keys:
                raw_time = schedule_obj.get(dk, "")
                sigla, is_unk = convert_time_to_buk_sigla(raw_time, shift_map)
                if is_unk and sigla not in valid_siglas:
                    unknown_shifts_found.add((raw_time, sigla))

        sigla_replacements = {}
        if unknown_shifts_found:
            st.error("⚠️ Se detectaron horarios no registrados en la tabla oficial BUK (`turnosSemanales`). Selecciona la sigla oficial correspondiente:")
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

        # Detección de duplicados en la asignación final
        assigned_names = [e[0] for e in mapped_entries]
        duplicates = [name for name in set(assigned_names) if assigned_names.count(name) > 1]
        if duplicates:
            st.error(f"⚠️ **Alerta de Duplicados:** Hay colaboradores asignados más de una vez: {', '.join(duplicates)}. Por favor verifica los nombres resueltos arriba.")

        # ---------------------------------------------------------------------
        # GENERACIÓN Y EXPORTACIÓN DEL ARCHIVO FINAL BUK
        # ---------------------------------------------------------------------
        st.divider()
        if st.button("🚀 Generar y Exportar Planilla BUK Final"):
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

            st.success("¡Planilla generada con éxito!")
            st.dataframe(resultado_buk[['Nombre del Colaborador', 'RUT', 'Área'] + columnas_fechas_buk], use_container_width=True)

            output_bytes = io.BytesIO()
            with pd.ExcelWriter(output_bytes, engine='openpyxl') as writer:
                resultado_buk.to_excel(writer, sheet_name='turnosColaboradores', index=False)
                for sheet_n in xl.sheet_names:
                    if sheet_n != 'turnosColaboradores':
                        xl.parse(sheet_n).to_excel(writer, sheet_name=sheet_n, index=False)

            st.download_button(
                label="📥 Descargar ReporteTurnosColaboradores.xlsx para BUK",
                data=output_bytes.getvalue(),
                file_name=f"ReporteTurnosColaboradores_{columnas_fechas_buk[0]}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

else:
    st.info("Por favor, sube el archivo base `ReporteTurnosColaboradores.xls` en el panel lateral para iniciar.")
