# Importar las librerías
import streamlit as st
from PIL import Image

# Configurar la aplicación
st.set_page_config(page_title="Clasificación de residuos",page_icon="♻️",layout="wide")

# Personalizar el diseño
st.markdown("""
<style>
    .stApp {
        background-color: #F5F8F4;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(120deg, #174631, #2E8054);
        padding: 35px 40px;
        border-radius: 20px;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 2.6rem;
        margin: 12px 0;
    }

    .hero p {
        color: #E3F1E8;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .etiqueta {
        color: #C6E8D1;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 2px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #DFE9E0;
        border-radius: 15px;
        padding: 20px;
    }

    div[data-testid="stImage"] img {
        border-radius: 15px;
    }

    div[data-testid="stFileUploader"] {
        background-color: white;
        border-radius: 12px;
    }

    h2, h3 {
        color: #1B4631;
    }
</style>
""",unsafe_allow_html=True)

# Encabezado de la aplicación
st.markdown("""
<div class="hero">
    <div class="etiqueta">RECICLAJE INTELIGENTE</div>
    <h1>♻ Clasificación de residuos</h1>
    <p>Identifica materiales mediante inteligencia artificial y aprende a separarlos correctamente.</p>
</div>
""",unsafe_allow_html=True)

# Crear las pestañas
tab1,tab2=st.tabs(["Clasificar residuo","Sobre el modelo"])

with tab1:
    st.subheader("Analiza tu residuo")
    st.write("Sube una fotografía para conocer el material y obtener recomendaciones de reciclaje.")

    # Permitir cargar una fotografía
    archivo=st.file_uploader("Selecciona una imagen",type=["jpg","jpeg","png"])

    if archivo is not None:
        # Mostrar la imagen junto a los resultados
        imagen=Image.open(archivo).convert("RGB")
        col_imagen,col_resultado=st.columns([1,1.2],gap="large")

        with col_imagen:
            st.image(imagen,caption="Fotografía cargada",width="stretch")
# Importar YOLO
from ultralytics import YOLO

# Cargar el modelo entrenado
@st.cache_resource
def cargar_modelo():
    return YOLO("modelos/yolo_residuos.pt")

# Clasificar la fotografía
if archivo is not None:
    modelo=cargar_modelo()
    resultado=modelo(imagen,verbose=False)[0]

    # Obtener la clase y su confianza
    clase=resultado.names[resultado.probs.top1]
    confianza=resultado.probs.top1conf.item()

    # Traducir las categorías al español
    nombres={
        "Cardboard":"Cartón",
        "Glass":"Vidrio",
        "Metal":"Metal",
        "Paper":"Papel",
        "Plastic":"Plástico"
    }

    # Mostrar los resultados a la derecha
    with col_resultado:
        st.subheader("Resultado de la clasificación")
        st.metric("Material identificado",nombres[clase])
        st.metric("Confianza",f"{confianza:.1%}")
import pandas as pd

with tab1:
    if archivo is not None:
        # Obtener las probabilidades de las cinco clases
        probabilidades=resultado.probs.data.cpu().numpy()

        # Organizar los resultados en español
        tabla=pd.DataFrame({
            "Material":[nombres[resultado.names[i]] for i in range(len(probabilidades))],
            "Probabilidad (%)":(probabilidades*100).round(1)
        })

        # Mostrar el gráfico de probabilidades
        st.subheader("Probabilidades por material")
        st.bar_chart(tabla,x="Material",y="Probabilidad (%)",
                     color="#2E8054",horizontal=True,height=300,sort=False)

        # Mostrar los porcentajes exactos
        with st.expander("Ver tabla de probabilidades"):
            st.dataframe(tabla,hide_index=True,width="stretch")

with tab1:
    if archivo is not None:
        # Advertir cuando la confianza sea baja
        if confianza < 0.70:
            st.warning("La confianza de esta predicción es baja. Se recomienda verificar el material antes de separarlo para reciclar.")

        # Recordar las limitaciones de la clasificación
        st.caption("Esta herramienta es de apoyo educativo. Sus resultados pueden contener errores y no reemplazan la revisión humana.")
# Importar Gemini
from google import genai

with tab1:
    if archivo is not None:
        # Generar una recomendación al presionar el botón
        if st.button("Obtener recomendación de reciclaje"):
            prompt=f"""
            Eres un asistente especializado en reciclaje de residuos.

            YOLO identificó el material: {clase}.
            Confianza de la clasificación: {confianza:.1%}.

            Entrega recomendaciones breves y prácticas en español sobre
            cómo separar y preparar este material para reciclarlo.
            Si la confianza es inferior al 70%, advierte que es
            importante verificar el material.
            No asegures que el residuo es reciclable sin conocer
            su composición ni las condiciones locales de reciclaje.
            """

            # Consultar la API de Gemini
            try:
                with st.spinner("Generando recomendación..."):
                    cliente=genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                    respuesta=cliente.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=prompt
                    )

                st.subheader("Recomendación de reciclaje")
                if respuesta.text:
                    st.write(respuesta.text)
                else:
                    st.warning("No se obtuvo una recomendación.")

            except Exception:
                st.error("No fue posible consultar Gemini. Revisa la conexión y la API Key.")
import os
import pandas as pd
with tab2:
    # Información del dataset
    st.subheader("Dataset RealWaste")
    st.write(
        "RealWaste fue desarrollado por Single, Iranmanesh y Raad (2023), "
        "utilizando fotografías de residuos recibidos en una instalación "
        "de gestión de residuos en Wollongong, Australia."
    )
    st.write(
        "El dataset original contiene 4.752 imágenes de nueve categorías. "
        "Para nuestro proyecto seleccionamos cinco materiales."
    )

    # Resumen de los datos
    col1,col2,col3=st.columns(3)
    with col1:
        st.metric("Imágenes seleccionadas","3.092")
    with col2:
        st.metric("Entrenamiento","2.472")
    with col3:
        st.metric("Validación","620")

    st.caption("80% de entrenamiento y 20% de validación.")
    st.markdown(
        "**Fuente:** [RealWaste en Hugging Face]"
        "(https://huggingface.co/datasets/shahzaibvohra/realwaste) · "
        "[Autores originales](https://github.com/sam-single/realwaste)"
    )
    st.divider()

    # Distribución de las categorías
    st.subheader("Categorías del dataset")

    datos_clases=pd.DataFrame({
        "Material":["Cartón","Vidrio","Metal","Papel","Plástico"],
        "Imágenes":[461,420,790,500,921],
        "Entrenamiento":[368,336,632,400,736],
        "Validación":[93,84,158,100,185]
    })

    st.bar_chart(datos_clases,x="Material",y="Imágenes",
                 color="#2E8054",horizontal=True,height=280,sort=False)

    with st.expander("Ver distribución de entrenamiento y validación"):
        st.dataframe(datos_clases,hide_index=True,width="stretch")

    # Mostrar fotografías del dataset
    st.markdown("**Ejemplos de las categorías**")

    clases=["Cardboard","Glass","Metal","Paper","Plastic"]
    materiales=["Cartón","Vidrio","Metal","Papel","Plástico"]

    if all(os.path.exists(f"ejemplos/{c}.jpg") for c in clases):
        columnas=st.columns(5)

        for i,clase in enumerate(clases):
            with columnas[i]:
                st.image(f"ejemplos/{clase}.jpg",width="stretch")
                st.caption(materiales[i])
    else:
        st.info("Las fotografías de ejemplo todavía no están disponibles.")

    st.divider()

    # Comparación de los modelos
    st.subheader("Comparación de modelos")
    col1,col2=st.columns(2)

    with col1:
        st.metric("EfficientNet-B0","87,3%")

    with col2:
        st.metric("YOLO26n-cls","91,5%")

    st.caption("Accuracy balanceada obtenida con las mismas 620 imágenes de validación.")
    st.success("Seleccionamos YOLO por obtener el mejor rendimiento en validación.")
    st.divider()

    # Resultados generales de YOLO
    st.subheader("Evaluación de YOLO")
    col1,col2,col3=st.columns(3)

    with col1:
        st.metric("Accuracy","91,0%")

    with col2:
        st.metric("Accuracy balanceada","91,5%")

    with col3:
        st.metric("Imágenes correctas","564 / 620")

    # Métricas para cada material
    st.markdown("**Rendimiento por categoría**")
    metricas=pd.DataFrame({
        "Material":["Cartón","Vidrio","Metal","Papel","Plástico"],
        "Precision":[0.93,0.92,0.89,0.94,0.89],
        "Recall":[0.88,0.95,0.90,0.95,0.89],
        "F1-score":[0.91,0.94,0.90,0.95,0.89]
    })
    st.dataframe(metricas,hide_index=True,width="stretch")

    # Matriz de confusión
    with st.expander("Ver matriz de confusión"):
        matriz=pd.DataFrame(
            [[82,0,1,3,7],
             [0,80,2,1,1],
             [1,3,142,1,11],
             [2,0,2,95,1],
             [3,4,12,1,165]],
            index=materiales,
            columns=materiales
        )

        st.caption("Filas: material real. Columnas: material predicho.")
        st.dataframe(matriz,width="stretch")
        st.write(
            "Una de las principales confusiones ocurre entre plástico "
            "y metal: 12 imágenes de plástico se clasificaron como metal "
            "y 11 de metal como plástico."
        )

        # Limitaciones del modelo
        st.info(
            "El modelo puede confundirse con residuos de apariencia similar o compuestos "
            "por varios materiales. En la validación, las principales confusiones fueron "
            "entre plástico y metal: 12 plásticos se clasificaron como metal y 11 metales "
            "como plástico."
        )
