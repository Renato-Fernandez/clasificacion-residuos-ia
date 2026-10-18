# Importar las librerías
import streamlit as st
from PIL import Image

# Configurar la aplicación
st.set_page_config(page_title="Clasificación de residuos",page_icon="♻️",layout="wide")

st.title("Clasificación de residuos")
st.write("Sube una fotografía para identificar el material y recibir recomendaciones de reciclaje.")

# Crear las pestañas
tab1,tab2=st.tabs(["Clasificar residuo","Sobre el modelo"])

with tab1:
    # Permitir cargar una fotografía
    archivo=st.file_uploader("Selecciona una imagen",type=["jpg","jpeg","png"])

    if archivo is not None:
        imagen=Image.open(archivo).convert("RGB")
        st.image(imagen,caption="Fotografía cargada",width=400)
# Importar YOLO
from ultralytics import YOLO

# Cargar el modelo entrenado
@st.cache_resource
def cargar_modelo():
    return YOLO("modelos/yolo_residuos.pt")

with tab1:
    if archivo is not None:
        # Clasificar la imagen
        modelo=cargar_modelo()
        resultado=modelo(imagen,verbose=False)[0]

        # Obtener la clase y su confianza
        clase=resultado.names[resultado.probs.top1]
        confianza=resultado.probs.top1conf.item()

        # Mostrar los resultados
        st.subheader("Resultado de la clasificación")
        col1,col2=st.columns(2)

        with col1:
            st.metric("Material identificado",clase)

        with col2:
            st.metric("Confianza",f"{confianza:.1%}")
import pandas as pd

with tab1:
    if archivo is not None:
        # Obtener las probabilidades de las cinco clases
        probabilidades=resultado.probs.data.cpu().numpy()

        # Organizar los resultados en una tabla
        tabla=pd.DataFrame({
            "Material":[resultado.names[i] for i in range(len(probabilidades))],
            "Probabilidad (%)":(probabilidades*100).round(1)
        })

        # Mostrar las probabilidades
        st.subheader("Probabilidades por material")
        st.bar_chart(tabla.set_index("Material"))
        st.dataframe(tabla,hide_index=True,use_container_width=True)

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
with tab2:
    # Información del dataset
    st.subheader("Datos utilizados")
    st.write("Utilizamos el dataset RealWaste, seleccionando 3.092 imágenes de cinco materiales.")
    st.write("El 80% se utilizó para entrenamiento y el 20% para validación.")

    # Mostrar el rendimiento de los modelos
    st.subheader("Comparación de modelos")
    col1,col2=st.columns(2)

    with col1:
        st.metric("EfficientNet-B0", "87,3%")

    with col2:
        st.metric("YOLO26n-cls", "91,5%")
    st.caption("Accuracy balanceada obtenida durante la validación.")

    # Resultados del modelo seleccionado
    st.subheader("Rendimiento de YOLO")
    st.write("Accuracy: 91,0%")
    st.write("Accuracy balanceada: 91,5%")
    st.write("Imágenes clasificadas correctamente: 564 de 620.")
    st.info("Estos resultados corresponden a validación. El rendimiento puede variar al utilizar fotografías nuevas.")
