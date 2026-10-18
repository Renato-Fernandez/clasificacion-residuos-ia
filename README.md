# Clasificación de residuos con IA

**Asignatura:** IA Generativa para la Gestión Industrial  
**Grupo 8:** Renato Fernández, Vicente Vargas y Matías González

**Aplicación web:** https://clasificacion-residuos-ia-cuzkftoprnh6an4wgxxwce.streamlit.app/

## 1. Descripción del proyecto

Desarrollamos una aplicación web que permite identificar residuos a partir de fotografías y entregar recomendaciones para su separación y reciclaje.

El objetivo es facilitar el reconocimiento de materiales mediante visión artificial y apoyar al usuario con información práctica generada por inteligencia artificial. La aplicación funciona como una herramienta educativa y no reemplaza la revisión humana.

## 2. Datos utilizados

Utilizamos el dataset **RealWaste**, desarrollado por Single, Iranmanesh y Raad (2023), que contiene fotografías de residuos reales obtenidas en Australia.

De sus 4.752 imágenes originales, seleccionamos **3.092 fotografías de cinco categorías**:

| Material | Imágenes |
|---|---:|
| Cardboard (cartón) | 461 |
| Glass (vidrio) | 420 |
| Metal | 790 |
| Paper (papel) | 500 |
| Plastic (plástico) | 921 |
| **Total** | **3.092** |

Dividimos los datos en **80% para entrenamiento (2.472 imágenes)** y **20% para validación (620 imágenes)**, manteniendo la proporción de cada categoría.

**Fuente utilizada:** https://huggingface.co/datasets/shahzaibvohra/realwaste

**Licencia:** Según los autores originales, el dataset RealWaste se distribuye bajo la licencia CC BY-NC-SA 4.0.

**Fuente:** https://github.com/sam-single/realwaste

Una de las principales limitaciones de los datos es que algunos residuos presentan características visuales similares o están compuestos por varios materiales. Además, las categorías tienen distintas cantidades de imágenes y no representan todos los residuos existentes.

## 3. Entrenamiento y comparación de modelos

Utilizamos **transfer learning**, adaptando dos modelos preentrenados para clasificar nuestras cinco categorías:

- **EfficientNet-B0:** entrenado durante 10 épocas utilizando PyTorch y `timm`.
- **YOLO26n-cls:** entrenado durante 10 épocas utilizando Ultralytics.

Para compararlos utilizamos las mismas 620 imágenes de validación y la accuracy balanceada, una métrica que considera el rendimiento de todas las categorías.

| Modelo | Accuracy balanceada |
|---|---:|
| EfficientNet-B0 | 87,3% |
| **YOLO26n-cls** | **91,5%** |

Seleccionamos **YOLO26n-cls**, ya que obtuvo un rendimiento superior por 4,2 puntos porcentuales y permite realizar clasificaciones directamente desde nuestra aplicación.

## 4. Evaluación y análisis de errores

Al evaluar YOLO con las imágenes de validación obtuvimos:

| Métrica | Resultado |
|---|---:|
| Accuracy | 91,0% |
| Accuracy balanceada | 91,5% |
| Imágenes correctas | 564 de 620 |
| Precision promedio (macro) | 0,92 |
| Recall promedio (macro) | 0,91 |
| F1-score promedio (macro) | 0,91 |

La matriz de confusión mostró que **vidrio y papel** fueron las categorías con mayor recall, alcanzando un 95%.

Por otro lado, identificamos dificultades para diferenciar plástico y metal: 12 imágenes de plástico fueron clasificadas como metal y 11 de metal como plástico.

Este error es relevante porque podría provocar una separación incorrecta de los residuos y contaminar los materiales destinados al reciclaje, generando trabajo adicional para corregirlos.

## 5. Aplicación web

Desarrollamos la aplicación mediante **Streamlit** y la publicamos en Streamlit Community Cloud.

La interfaz permite:

1. Subir fotografías en formato JPG, JPEG o PNG.
2. Identificar el material mediante YOLO.
3. Mostrar la confianza y las probabilidades de las cinco categorías.
4. Consultar las métricas del modelo seleccionado.
5. Recibir una advertencia cuando la confianza sea inferior al 70%.
6. Solicitar recomendaciones de reciclaje generadas por Gemini.

La integración con **Gemini** funciona mediante API. Al solicitar una recomendación, se envían la categoría identificada y la confianza del modelo para generar consejos sobre la separación del residuo.

El prompt considera las clasificaciones de baja confianza y evita asegurar que un objeto es reciclable sin conocer su composición ni las condiciones locales.

La API Key se mantiene protegida mediante los secretos de Streamlit y no se incluye en el repositorio.

## 6. Limitaciones

Aunque obtuvimos buenos resultados, nuestro proyecto presenta algunas limitaciones:

- El modelo reconoce únicamente cinco categorías y puede clasificar incorrectamente materiales no incluidos.
- Los residuos compuestos por varios materiales pueden generar confusiones.
- Las recomendaciones de Gemini dependen de la categoría predicha y no permiten comprobar la composición exacta del objeto ni las condiciones locales de reciclaje.

Por estos motivos, la aplicación está pensada como una **herramienta de apoyo educativo**, cuyos resultados necesitan verificación humana cuando exista incertidumbre.

## 7. Uso de inteligencia artificial

Durante el desarrollo del proyecto utilizamos inteligencia artificial generativa como herramienta de apoyo en distintas etapas. Inicialmente, nos permitió explorar posibles ideas de proyectos, entre las cuales seleccionamos la clasificación de residuos por su utilidad y propuesta de valor. También nos ayudó a organizar la estructura de los notebooks, definir títulos y presentación de la información.

En cuanto al código, tomamos como principal referencia los ejemplos y metodologías enseñados en clases, realizando las adaptaciones necesarias para nuestro proyecto. La IA nos sirvió para verificar estas adaptaciones, resolver errores puntuales y apoyar el desarrollo de la aplicación web y su integración con Gemini, incluyendo la elaboración de un prompt más breve y preciso. Asimismo, mantuvimos la metodología de entrenamiento y validación trabajada en clases.

También la utilizamos para revisar la redacción y ortografía, mejorar la presentación de tablas, ordenar el código e incorporar comentarios que facilitaran su comprensión. Por último, nos apoyamos en ella para mejorar algunos aspectos visuales de la aplicación y estructurar este README, revisando posteriormente la información, los resultados y las fuentes utilizadas.