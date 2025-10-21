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


## Mejorar la evaluación de los resultados y estandarizar métricas para el mismo

Definir métricas objetivas ya anteriormente usadas en artículos como Human Last Exam y diferentes
benchmarks de evaluación de modelos de lenguaje, para evaluar la calidad de las respuestas. Esto con el fin no
solo de analizar los resultados del experimento, sino de sentar una testbed para el benchmark de evaluación que
se planea realizar.

- Estudiar las investigaciones anteriores sobre evaluación de respuestas, centradas en que el concepto sea
claro y, además, que el resultado haya acertado en el valor:

  1. Generación automática de marco semántico para evaluación de respuestas.

  2. Marco semántico, concepto claro, Accuracy.

  3. Métricas clásicas de comprensión automática del habla, tales como BLEU, ROUGE, METEOR, BERTScore, entre otras.

  4. Extractor de concepto de valor y su comparación con el valor esperado.

  5. Sistema basado en reglas para evaluación de respuestas.

  6. Conocimiento en marco de evaluación de respuestas.

  7. Value Accuracy.

  8. Análisis de sistemas de evaluación de respuestas automáticas.

## Puntos tratados sobre el artículo

Además de tener en cuenta los cambios subrayados en el PDF por el profesor Javier, los cuales serán modificados
en la medida en que se modifique el código y logremos guiar la generación de datos a algo más tangible y concreto,
se resaltan algunos puntos importantes que deben cambiar.

1.  Nunca trabajar con una sola distribución de probabilidad para una variable, sino que esta sea la combinación
de diferentes gaussianas, o incluso de otros tipos de distribuciones, para lograr un comportamiento más realista.

    **Ejemplo:** Un hotel está dado en el centro de una ciudad, pero también dada una probabilidad más alta de estar cerca
        de una estación por la cual cruzan varias líneas de metro. Ahora bien, la combinación de un hotel, dada una estación central,
        da más probabilidad de que exista un restaurante cerca; a su vez, la combinación de estos puntos de interés sugiere que debe
        existir una atracción turística próxima, y así sucesivamente.


2.  Los precios deben girar en función de una distribución de probabilidad, pero estos también deben ser una
probabilidad conjunta con puntos turísticos y paradas de metro, para que el modelo pueda aprender a relacionar qué zonas
son más caras y cuáles son más económicas.

3.  Hay que revisar el algoritmo de construcción del grafo: si hacemos que crezca cada vez o si se tienen grafos predefinidos,
para evitar que ocurran errores en la generación de las rutas, como construcciones erráticas o que, en el mundo real, no tengan
sentido. Es mejor que en cada iteración se genere una codificación diferente (tiempos entre tramos, coste, etc.).

4. En las preguntas, hay varios temas a tratar:

   1. Por ahora, solo analicemos el componente factual.

   2. El nivel emocional se debe dejar para una segunda prueba.

   3. Revisar cómo es el nivel de construcción.

   4. Hacer un análisis semántico y generar un marco semántico que podamos llenar con cada una de las preguntas.

   5. Tener en cuenta los dos tipos de confianza sobre las respuestas: confianza en términos conversacionales, es decir,
   que el modelo se exprese de forma natural y fluida, qué tan seguro está de su respuesta, y la confianza general
   que proviene de la prueba de Monte Carlo.

