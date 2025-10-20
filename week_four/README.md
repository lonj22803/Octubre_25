# Revision reunion 20/10/2025

Después de la revision realizada para el resultado de la evaluación de los
prompts, se sugirieron cambios sustanciales en el prompt que indica las estaciones
metro, explicación del tipo de información, etc. Estos cambios serán detallados más 
adelante en este documento, asi como estandarizar la evaluación de los resultados de 
los modelos usando estos prompts, con métricas objetivas.

Respecto al borrador de la propuesta del artículo, se realizaron algunas modificaciones 
que enriquecen la proyección del llamado "Benchmark" para la evaluación de cruce de 
información entre modelos, con información estructurada y no estructurada, cambiante,
sinténica y verificable en cada iteración. Esto causa un cambio que será realizado al 
sistema que aún está en fase de prueba y desarrollo.

## Estudio de ablación de prompts

Dentro de las modificaciones sugeridas para el prompt, para que este se perfile y lo entienda 
un poco mejor el modelo, se sugirieron los siguientes cambios:

1. ¿Qué hacer si no hay información suficiente?  
    - Cambiar la frase "No puedo ayudarte con eso, solo puedo ayudarte a guiarte en el sistema de metro" a 
ayudarle al modelo a identificar la diferencia entre un tema sobre el cual no puede responder, a cuando se necesita
solicitar búsqueda de información adicional o aclaración de la petición, identificando cuando la pregunta está
fuera de contexto, cuando está dentro del contexto, pero no hay suficiente información, o cuando la pregunta es ambigua y necesita
ser reformulada o aclarada.

2. Crear una explicación detallada del tipo de información que está procesando:

    - Incluir una sección en el prompt que explique al modelo qué información se proporciona, cómo es y
de qué tipo es.

    - Mejorar la forma de presentar la información para que no existan redundancias o ambigüedades en la interpretación
de la información y en los conceptos del diálogo. Cambiar "sentido de ida y vuelta" por "sentido 1" y "sentido 2".

    - Mejorar la codificación de las estaciones, para que crezca según la cantidad de líneas que pasan por una
estación, en función de la estación con mayor cantidad de líneas. Ejemplo: si la estación con mayor cantidad de líneas tiene 3
o incluso 4, que la codificación incluya en su nombre las 4 letras de cada línea. Ejemplo: Estación "A1B2C3D4" para una estación que conecta 4 líneas.

    - Que la información que se le entrega en el prompt sea la primera y vaya por delante, antes de cualquier otra instrucción o contexto,
para mejorar la comprensión del modelo sobre la información que se le entrega.

    - Contextualizar sobre los conceptos de estación, línea, sentido y transbordo que están en la repetición de
estaciones dentro de las líneas.

    - Depurar errores en los ejemplos, primando la factualidad y no la creatividad o el enfoque emocional.


3. Mejorar la evaluacion de los resultados y estandarizar metricas para el mismo.

    - Definir metricas objetivas ya enteriormente usadas en articulos como Human Last Exam y diferentes 
benchmarks de evaluación de modelos de lenguaje, para evaluar la calidad de las respuestas, esto con el fin no
solo se analizar los resultados del experimento, si no sentar una testbed, para el benchmark de evaluacion que
se planea realizar.

   - Estudiar las investigaciones anteriores de sobre evaluacion de respuestas centrado en que el conepto sea 
claro y ademas de ello el resultado haya acertado en el valor:
     1. Genracion automarica de marco semantico para evaluacion de respuestas
     2. Marco semantico, concepto claro, Accuracy, 
     3. Metricas de clasicas comprension automatica del habla, tales como BLEU, ROUGE, METEOR, BERTScore, entre otras.
     4. Extractor de concepto de valor y su comparacion con el valor esperado.
     5. Sistema basado en reglas para evaluacion de respuestas.
     6. Conocimiento en marco de evaluacion de respuestas.
     7. Value Accuracy
     8. Analisis de sistemas de evaluacion de respuestas automatica.

4. Puntos tratados sobre el articulo:
 
 - Ademas de tener en cuenta los cambios subrayados en el PDF por el profesor Javier, los cuales seran modificados
en la medida que se modifique el codigo y logremos guiar la generacion de datos a algo mas tangible y contreto, 
se resaltan algunos puntos importantes que deben cambiar.
   
   1. Nunca trabajar con una solo distribución de probabilidad para una variable, si no que esta sea, la comnbinacion
 de diferentes Gaussianas, o incluso otro tipo de distribuciones, para lograr un comportamiento mas realista.

Ejemplo de esto: Un hotel esta dado el centro de una ciudad, pero dada tmabien una probabildad mas alta de esta cerca
de una estacion por la cual cruzan varias lineas de metro.

