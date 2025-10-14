# Trabajo tercera semana de Octubre

## Objetivos
1. Diseñar y organizar una metodología para la construcción de un TestBed o Benchmark orientado a la evaluación 
integral de Grandes Modelos de Lenguaje (LLMs) bajo condiciones controladas. Esta metodología buscará garantizar 
el uso de información completamente nueva y verificable, no presente en los procesos de entrenamiento ni de ajuste
fino de los modelos, con el fin de medir su capacidad de razonamiento, comprensión, confianza e interpretabilidad
de manera objetiva y reproducible.

2. Evaluar el estudio de ablación aplicado a diferentes técnicas de prompt engineering, analizando el impacto del 
uso de rutas en formato JSON, en texto, en combinación de ambas, y la incorporación de ejemplos few-shot en los 
prompts. El propósito es identificar las mejores prácticas y estrategias que permitan maximizar el rendimiento y 
la efectividad de los LLMs, así como determinar las limitaciones en su capacidad para responder preguntas complejas,
primando la reduccion de alucinaciones.

3. Investigar e implementar una técnica y un marco de evaluación de la confianza en las respuestas generadas por Grandes
Modelos de Lenguaje (LLMs). El marco explorará y comparará métodos como autoevaluación del modelo, métricas de 
incertidumbre (p. ej. entropía, calibración de probabilidades, intervalos de confianza), y verificación frente 
a fuentes externas confiables. El propósito es desarrollar una medida cuantificable de fiabilidad que permita evaluar 
si las técnicas de prompt engineering (incluyendo few-shot, formatos JSON vs. texto, etc.) mejoran la confianza de las 
respuestas en distintos contextos y aplicaciones, y proporcionar criterios claros y reproducibles para su uso práctico.

4. Realizar dos implementaciones experimentales adicionales —una utilizando un LLM con capacidades explícitas de razonamiento 
y otra integrando una técnica de extracción de información (RAG — Retrieval Augmented Generation)— para comparar su rendimiento, 
precisión y capacidad en tareas controladas. El propósito es evaluar y contrastar las fortalezas y limitaciones de cada enfoque 
en escenarios prácticos, identificando para qué tipos de problemas o aplicaciones cada uno resulta más adecuado.

## Metodologia para la construcción del TestBed o Benchmark

![](images/sistema_generico.png)
Sistema de metro

La semana anterior se creó un sistema genérico y a su misma vez en conjunto con el sistema metro, se construyo un sistema 
hotelero al rededor del punto central del sistema metro, ahora bien para que estos datos parezcan factibles y reales se
deben centrar en un plano no de coordenadas cartesianas sino en un plano geográfico como longitud y latitud, para esto se hizo una 
conversion lineal a fin de que los datos generados por el sistema metro se asemejen a datos reales, para esto se tómo como referencia
la ciudad de Madrid, España, y se tomó como punto central la estacion de Sol y se hizo la conversion lineal a partir de este punto.

> [!NOTE]  
> Para más detalles mirar el archivo de generación de datos de la semana pasada.

### Estandarizacion del codigo

Con el fin de poder escalar y escalar el proyecto se ha decidido estandarizar el código, con programacion orientada a objetos,
de esta manera se puede tener un código más limpio y ordenado, además de que se permite agregar objeto como:
- Sistema metro 
- Sistema hotelero
- Sistema de restaurantes
- Aeropuerto 
- Sistema Turístico
- etc
 
#### ¿Para que?
El objetivo es construir un conjunto de pruebas que sea evolutivo y coherente, en otras palabras, que permita crecer en función de la complejidad
y capacidad del modelo para resolver el problema, lo suficientemente robusto y flexible para adaptarse a diferentes escenarios y tipos de modelos,
con el fin de evaluar el cruce de datos de los modelos de lenguaje, además de que se pueda verificar su veracidad y que no se encuentre
en los datos de entrenamiento de los modelos, para esto se ha decidido crear un sistema genérico que permita crear diferentes tipos de sistemas
cada vez que se necesite y a su vez se tenga información de como se creó, además que la pregunta se puedan generar cada vez que se necesite asi 
como las repuestas, para evaluar el modelo. Además de la posibilidad de trasladar el sistema a diferentes ciudades del mundo, asi el modelo podra 
confundirse con datos geográficos, de entrenamiento y no los datos generados para ser evaluado.

De la semana pasada el sistema de metro y hotelero ya se encuentra sobre la ciudad de Madrid. 

![](images/hoteles_concentrados_centro_y_metro_madrid.png)


