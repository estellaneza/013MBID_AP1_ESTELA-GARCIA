import streamlit as st
import requests

# ============================
# Configuración de la página
# ============================

st.set_page_config(
    page_title="Predicción de Mora en Créditos",
    page_icon="💳",
    layout="wide"
)

# ============================
# Sidebar
# ============================

with st.sidebar:
    st.header("Instrucciones")
    st.write(
        "1. Introduce los datos del cliente.\n"
        "2. Pulsa **Predecir**.\n"
        "3. Consulta el resultado de la predicción."
    )

    st.divider()

    st.header("Configuración de la API")
    api_url = st.text_input(
        "URL de la API",
        value="http://127.0.0.1:8000"
    )

    if st.button("Probar conexión"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            if r.status_code == 200 and r.json().get("status") == "ok":
                st.success("API operativa ✅")
            else:
                st.error("La API no está disponible ❌")
        except Exception:
            st.error("No se puede conectar con la API")

# ============================
# Título
# ============================

st.title("Predicción de Mora en Créditos")
st.write(
    "Formulario para predecir la probabilidad de mora usando el modelo entrenado."
)

# ============================
# Formulario
# ============================

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        edad = st.number_input("Edad", 18, 100, 30)
        ingresos = st.number_input("Ingresos", value=9585)
        pct_ingreso = st.number_input("Pct. ingreso", value=0.10)
        tasa_interes = st.number_input("Tasa de interés", value=11.14)
        estado_credito = st.selectbox("Estado crédito", [0, 1])

    with col2:
        situacion_vivienda = st.selectbox("Vivienda", ["PROPIA", "ALQUILER"])
        objetivo_credito = st.selectbox("Objetivo", ["EDUCACION", "VIVIENDA", "CONSUMO"])
        estado_civil = st.selectbox("Estado civil", ["CASADO", "SOLTERO", "DIVORCIADO"])
        estado_cliente = st.selectbox("Estado cliente", ["ACTIVO", "INACTIVO"])
        genero = st.selectbox("Género", ["M", "F"])

    with col3:
        antiguedad_cliente = st.number_input("Antigüedad cliente", value=39.0)
        limite_credito_tc = st.number_input("Límite TC", value=12691.0)
        nivel_educativo = st.selectbox(
            "Nivel educativo", ["SECUNDARIO_COMPLETO", "UNIVERSITARIO"]
        )
        nivel_tarjeta = st.selectbox("Nivel tarjeta", ["Blue", "Silver", "Gold"])
        personas_a_cargo = st.number_input("Personas a cargo", value=4)

    st.divider()

    col4, col5 = st.columns(2)

    with col4:
        capacidad_pago = st.number_input("Capacidad de pago", value=0.10)
        estabilidad_laboral = st.number_input("Estabilidad laboral", value=0.24)

    with col5:
        operaciones_mensuales = st.number_input("Operaciones mensuales", value=3.5)
        gasto_medio_mensual = st.number_input("Gasto medio mensual", value=95.33)
        gasto_promedio_operacion = st.number_input(
            "Gasto promedio por operación", value=27.24
        )

    submit = st.form_submit_button("Predecir", type="primary", use_container_width=True)

# ============================
# Resultado
# ============================

if submit:
    payload = {
        "edad": edad,
        "situacion_vivienda": situacion_vivienda,
        "ingresos": ingresos,
        "objetivo_credito": objetivo_credito,
        "pct_ingreso": pct_ingreso,
        "tasa_interes": tasa_interes,
        "estado_credito": estado_credito,
        "antiguedad_cliente": antiguedad_cliente,
        "estado_civil": estado_civil,
        "estado_cliente": estado_cliente,
        "genero": genero,
        "limite_credito_tc": limite_credito_tc,
        "nivel_educativo": nivel_educativo,
        "nivel_tarjeta": nivel_tarjeta,
        "personas_a_cargo": personas_a_cargo,
        "capacidad_pago": capacidad_pago,
        "estabilidad_laboral": estabilidad_laboral,
        "operaciones_mensuales": operaciones_mensuales,
        "gasto_medio_mensual": gasto_medio_mensual,
        "gasto_promedio_operacion": gasto_promedio_operacion,
    }

    try:
        r = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        r.raise_for_status()
        result = r.json()

        # Datos de la API
        prediction = result["prediction"]          # 'N' o 'Y'
        prob = result["probability"]               # dict con N / Y

        prob_mora = prob.get("Y", 0.0)
        prob_no_mora = prob.get("N", 0.0)

        st.divider()
        st.subheader("📊 Resultado de la predicción")

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            if prediction == "Y":
                st.error("**Predicción: Entra en mora**")
            else:
                st.success("**Predicción: No entra en mora**")

        with col_r2:
            st.metric("Probabilidad de mora", f"{prob_mora * 100:.1f}%")
            st.metric("Probabilidad de no mora", f"{prob_no_mora * 100:.1f}%")

        with st.expander("Ver respuesta completa de la API"):
            st.json(result)

    except Exception as e:
        st.error(f"Error llamando a la API: {e}")
        