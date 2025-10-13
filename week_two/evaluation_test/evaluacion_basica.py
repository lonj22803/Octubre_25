import pandas as pd  # Para manipular DataFrames y leer CSVs
import re  # Para normalización de texto (remover puntuación)
from difflib import SequenceMatcher  # Para similitud textual simple (built-in)
import warnings  # Para suprimir warnings específicos de NLTK

# Opcional: Para BLEU score (instala con pip install nltk)
NLTK_AVAILABLE = False
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.tokenize import word_tokenize
    from nltk.translate.bleu_score import SmoothingFunction  # Para suavizado y evitar warnings/BLEU=0

    NLTK_AVAILABLE = True

    # Descarga automática silenciosa (solo la primera vez; requiere internet)
    try:
        nltk.download('punkt_tab', quiet=True)  # Para tokenización multi-idioma (incluye español)
        nltk.download('punkt', quiet=True)  # Backup para español/oraciones
        print("NLTK recursos descargados exitosamente para español.")
    except Exception as dl_error:
        print(
            f"Advertencia: No se pudieron descargar recursos NLTK automáticamente ({dl_error}). BLEU será 0. Ejecuta manualmente: nltk.download('punkt_tab')")
        NLTK_AVAILABLE = False  # Desactiva si falla descarga

except ImportError:
    print("Advertencia: NLTK no está instalado. BLEU score será 0. Instala con 'pip install nltk'.")
    NLTK_AVAILABLE = False

# Cargar los datos
# - 'resultados_experimento_one.csv': Respuestas del modelo (columnas esperadas: 'Pregunta', 'Respuesta', 'Modelo', 'Prompt', 'Tiempo_Respuesta').
# - 'preguntas_respuestas.csv': Ground truth (columnas esperadas: 'pregunta', 'respuesta_esperada').
try:
    resultados_df = pd.read_csv('resultados_experimentos_two.csv')
    preguntas_df = pd.read_csv('preguntas_respuestas.csv')

    # Validar columnas básicas (evita errores downstream)
    required_cols_resultados = ['Pregunta', 'Respuesta', 'Modelo', 'Prompt', 'Tiempo_Respuesta']
    required_cols_esperadas = ['pregunta', 'respuesta_esperada']
    if not all(col in resultados_df.columns for col in required_cols_resultados):
        raise ValueError(
            f"Columnas faltantes en resultados_df: {set(required_cols_resultados) - set(resultados_df.columns)}")
    if not all(col in preguntas_df.columns for col in required_cols_esperadas):
        raise ValueError(
            f"Columnas faltantes en preguntas_df: {set(required_cols_esperadas) - set(preguntas_df.columns)}")

    print("Datos cargados exitosamente.")
except FileNotFoundError as e:
    print(f"Error: Archivo no encontrado - {e}. Asegúrate de que los CSVs estén en el directorio actual.")
    exit(1)
except ValueError as e:
    print(f"Error en columnas: {e}")
    exit(1)


def normalizar_texto(texto):
    """
    Normaliza texto: minúsculas, remueve puntuación y espacios extra.
    Adaptado para español (mantiene acentos básicos).
    """
    if pd.isna(texto) or texto == "":
        return ""
    texto = str(texto).lower().strip()
    texto = re.sub(r'[^\w\sáéíóúüñÁÉÍÓÚÜÑ]', ' ', texto)  # Remueve puntuación, mantiene acentos y ñ
    texto = re.sub(r'\s+', ' ', texto)  # Normaliza espacios
    return texto


def evaluacion_automatica_mejorada(df_resultados, df_esperadas, umbral_longitud=50):
    """
    Función mejorada para evaluación automática de respuestas de LLM (optimizada para español).

    Mejoras para BLEU:
    - Usa SmoothingFunction para evitar BLEU=0 y warnings en overlaps bajos (común en QA corta).
    - Suprime warnings específicos de NLTK BLEU con catch_warnings.
    - Calcula BLEU solo si textos >10 chars (evita procesar vacíos).
    - Keywords expandidos para español/transporte.
    - Score total: Keywords (0.3), Ruta (0.3), Longitud (0.2), Similitud (0.2) + BLEU (0.1 opcional).

    Parámetros:
    - df_resultados (pd.DataFrame): Como antes.
    - df_esperadas (pd.DataFrame): Como antes.
    - umbral_longitud (int): Umbral para "respuesta completa" (default: 50 caracteres).

    Retorna:
    - pd.DataFrame: Con métricas.
    """
    resultados = []  # Lista para resultados

    # Normalizar columnas de preguntas para matching (una vez, para eficiencia)
    df_esperadas['pregunta_norm'] = df_esperadas['pregunta'].apply(normalizar_texto)

    for _, fila in df_resultados.iterrows():
        pregunta_original = fila['Pregunta']
        respuesta_modelo_original = fila['Respuesta']

        # Normalizar para métricas
        pregunta_norm = normalizar_texto(pregunta_original)
        respuesta_modelo_norm = normalizar_texto(respuesta_modelo_original)

        # Buscar respuesta esperada: Matching normalizado (insensible)
        match = df_esperadas[df_esperadas['pregunta_norm'] == pregunta_norm]
        if len(match) > 0:
            respuesta_esperada = match.iloc[0]['respuesta_esperada']
            respuesta_esperada_norm = normalizar_texto(respuesta_esperada)
        else:
            print(
                f"Advertencia: No se encontró respuesta esperada para la pregunta normalizada: '{pregunta_norm[:50]}...'")
            respuesta_esperada = ""
            respuesta_esperada_norm = ""
            similitud = 0.0
            bleu_score = 0.0

        # Si respuesta modelo está vacía, score=0
        if respuesta_modelo_norm == "":
            print(f"Advertencia: Respuesta vacía para pregunta: '{pregunta_norm[:50]}...'")
            contiene_estaciones = False
            menciona_ruta = False
            longitud_respuesta = 0
            similitud = 0.0
            bleu_score = 0.0
            score_total = 0.0
        else:
            # Métricas heurísticas (normalizadas, adaptadas para español)
            longitud_respuesta = len(respuesta_modelo_original)  # Usa original para longitud real
            # Keywords expandidos para transporte en español (más sinónimos)
            contiene_estaciones = any(keyword in respuesta_modelo_norm for keyword in
                                      ['estacion', 'linea', 'transbordo', 'sentido', 'parada', 'metro', 'estaciones',
                                       'tren', 'subte'])
            menciona_ruta = any(keyword in respuesta_modelo_norm for keyword in
                                ['ruta', 'tomar', 'hasta', 'desde', 'camino', 'direccion', 'itinerario', 'viaje',
                                 'pasos'])

            # Similitud simple (SequenceMatcher: 0-1, funciona bien en español)
            similitud = SequenceMatcher(None, respuesta_modelo_norm, respuesta_esperada_norm).ratio()

            # BLEU score (opcional, con manejo de errores, smoothing y supresión de warnings)
            bleu_score = 0.0
            if NLTK_AVAILABLE and len(respuesta_esperada_norm) > 10 and len(respuesta_modelo_norm) > 10:
                try:
                    # Suprime warnings específicos de BLEU durante el cálculo
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="The hypothesis contains 0 counts")
                        # Tokenización para español
                        ref_tokens = word_tokenize(respuesta_esperada_norm, language='spanish')
                        hyp_tokens = word_tokenize(respuesta_modelo_norm, language='spanish')
                        # Aplica smoothing para evitar 0 y warnings en QA corta
                        smooth = SmoothingFunction().method1
                        bleu_score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)
                except Exception as bleu_error:
                    print(f"Advertencia en BLEU para pregunta '{pregunta_norm[:50]}...': {bleu_error}. Usando 0.0.")
                    bleu_score = 0.0

            # Score total mejorado (ponderado, escala 0-1)
            score_heuristicas = 0
            if contiene_estaciones:
                score_heuristicas += 0.3
            if menciona_ruta:
                score_heuristicas += 0.3
            if longitud_respuesta > umbral_longitud:
                score_heuristicas += 0.2

            score_total = score_heuristicas + (similitud * 0.2) + (bleu_score * 0.1)  # Agrega BLEU con bajo peso
            score_total = min(1.0, score_total)  # Cap a 1.0

        # Diccionario de resultados (expandido)
        resultado = {
            'Modelo': fila['Modelo'],
            'Prompt': fila['Prompt'],
            'Pregunta': pregunta_original,
            'Respuesta_Esperada': respuesta_esperada,  # Para trazabilidad
            'Longitud_Respuesta': longitud_respuesta,
            'Contiene_Estaciones': contiene_estaciones,
            'Menciona_Ruta': menciona_ruta,
            'Similitud_Simple': round(similitud, 4),  # Redondea para legibilidad
            'BLEU_Score': round(bleu_score, 4),
            'Score_Total': round(score_total, 4),
            'Tiempo_Respuesta': fila['Tiempo_Respuesta']
        }
        resultados.append(resultado)

    df_resultados_final = pd.DataFrame(resultados)
    # Limpia columna temporal si existe
    if 'pregunta_norm' in df_esperadas.columns:
        df_esperadas.drop('pregunta_norm', axis=1, inplace=True)
    return df_resultados_final


# Si NLTK no está disponible, informa al usuario
if not NLTK_AVAILABLE:
    print("Nota: BLEU no disponible. El score usa solo heurísticas y similitud simple (suficiente para español).")

# Ejecutar la evaluación mejorada
print("Ejecutando evaluación mejorada (sin warnings de BLEU)...")
resultados_mejorados = evaluacion_automatica_mejorada(resultados_df, preguntas_df, umbral_longitud=50)

# Guardar resultados en CSV
resultados_mejorados.to_csv('evaluacion_resultados_basica_two.csv', index=False, encoding='utf-8')
print("Resultados guardados en 'evaluacion_resultados_basica_two.csv'.")
