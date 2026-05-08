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
# Sidebar (igual que el profe, adaptado)
# ============================
with st.sidebar:
    st.header("Instrucciones")
    st.write("""
    1. Ingrese los datos del cliente en el formulario.
    2. Haga clic en el botón **Predecir**.
    3. Revise el resultado de la predicción.
    """)
    st.divider()

    st.header("Configuración de la API")
    api_url = st.text_input(
        "URL de la API",
        value="http://127.0.0.1:8000",
        help="URL donde se está ejecutando la API FastAPI."
    )

    st.divider()
    if st.button("Probar conexión con la API"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            if r.status_code == 200:
                st.success("Conexión correcta con la API.")
            else:
                st.error("La API respondió con error.")
        except Exception as e:
            st.error(f"No se pudo conectar con la API: {e}")

# ============================
# Título y descripción
# ============================
st.title("Predicción de Mora en Créditos")
st.write(
    "Ingrese los datos del cliente para predecir la probabilidad de mora "
    "utilizando el modelo de machine learning."
)

# ============================
# Formulario (variables EXACTAS de tu API)
# ============================
with st.form("prediction_form"):
    st.subheader("Datos del cliente")
    col1, col2, col3 = st.columns(3)

    with col1:
        edad = st.number_input("Edad", min_value=18, max_value=100, value=21)
        ingresos = st.number_input("Ingresos", min_value=0, value=9600)
        pct_ingreso = st.number_input("Porcentaje del ingreso", value=0.10)
        tasa_interes = st.number_input("Tasa de interés", value=11.14)
        estado_credito = st.selectbox("Estado del crédito", [0, 1])

    with col2:
        situacion_vivienda = st.selectbox(
            "Situación de la vivienda",
            ["PROPIA", "ALQUILER"]
        )
        objetivo_credito = st.selectbox(
            "Objetivo del crédito",
            ["EDUCACIÓN", "VIVIENDA", "CONSUMO"]
        )
        estado_civil = st.selectbox(
            "Estado civil",
            ["CASADO", "SOLTERO", "DIVORCIADO"]
        )
        estado_cliente = st.selectbox(
            "Estado del cliente",
            ["ACTIVO", "INACTIVO"]
        )
        genero = st.selectbox("Género", ["M", "F"])

    with col3:
        antiguedad_cliente = st.number_input("Antigüedad del cliente", value=39.0)
        limite_credito_tc = st.number_input("Límite crédito TC", value=12691.0)
        nivel_educativo = st.selectbox(
            "Nivel educativo",
            ["SECUNDARIO_COMPLETO", "UNIVERSITARIO"]
        )
        nivel_tarjeta = st.selectbox(
            "Nivel de tarjeta",
            ["Blue", "Silver", "Gold"]
        )
        personas_a_cargo = st.number_input("Personas a cargo", value=3.0)

    st.divider()
    st.subheader("Información adicional")

    col4, col5 = st.columns(2)

    with col4:
        capacidad_pago = st.number_input("Capacidad de pago", value=0.104167)
        estabilidad_laboral = st.number_input("Estabilidad laboral", value=0.238095)

    with col5:
        operaciones_mensuales = st.number_input("Operaciones mensuales", value=3.5)
        gasto_medio_mensual = st.number_input("Gasto medio mensual", value=95.333333)
        gasto_promedio_operacion = st.number_input(
            "Gasto promedio por operación", value=27.238095
        )

    submit_button = st.form_submit_button(
        "Predecir",
        use_container_width=True,
        type="primary"
    )

# ============================
# Llamada a la API
# ============================
if submit_button:
    input_data = {
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
        "gasto_promedio_operacion": gasto_promedio_operacion
    }

    try:
        resp = requests.post(f"{api_url}/predict", json=input_data, timeout=10)
        resp.raise_for_status()
        result = resp.json()

        st.divider()
        st.subheader("📊 Resultado de la predicción")

        prediction = result["prediction"]
        prob = result.get("probability", {})
        labels = result.get(
            "class_labels",
            {"N": "No entra en mora", "Y": "Entra en mora"}
        )

        label_text = labels.get(str(prediction), prediction)

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            if str(prediction) == "Y":
                st.error(f"**Predicción: {label_text}**")
            else:
                st.success(f"**Predicción: {label_text}**")

        with col_r2:
            prob_mora = prob.get("Y", 0)
            prob_no_mora = prob.get("N", 1 - prob_mora)
            st.metric("Probabilidad de mora", f"{prob_mora*100:.1f}%")
            st.metric("Probabilidad de no mora", f"{prob_no_mora*100:.1f}%")

        with st.expander("Ver respuesta completa de la API"):
            st.json(result)

    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar con la API. Verifique la URL.")
    except requests.exceptions.HTTPError as e:
        st.error(f"Error de la API: {e}")
    except Exception as e:
        st.error(f"Error inesperado: {e}")
