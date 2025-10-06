# Analisis de resultador inicial.

Recordemos que se construyo un sistema de metro ficticio con 
conexiones y estaciones. El objetivo de este sistema es analizar
como un LLM lo resuelve, mayormente aun a traves de prompts:

## Prueba I
El modelo arrojo una alucinacion, pero el trayecto es correcto, es
decir asocio de manera correcta la estacion de partida, el trasbordo
y la estacion de llegada, muy por debajo de los modelos publicos que
logran resolverlo sin alucinaciones y sin problemas.

Respuesta del modelo:
```
Pregunta: ¿Quiero ir de RD3VC a AG7BH? ¿Que lineas debo coger, y cuantos transbordos habra

Respuesta:Para llegar de la estación RD3VC a la estación AG7BH, debes tomar las siguientes líneas:

1. Toma la línea amarilla en sentido vuelta desde RD3VC (estación de partida) hasta la estación BD2VB (estación de transbordo).
2. En BD2VB, cambia a la línea azul en sentido ida y continúa hasta la estación AG7BH (estación de llegada).

En total, podrás tomar 1 transbordo:

*   Estaciones: RD3VC, BD2VB, AG7BH
*   Líneas utilizadas: Amarilla (RD3VC a BD2VB) y Azul (BD2VB a AG7BH)
*   Transbordos: 1

La ruta completa de estaciones:

-   RD3VC → BD2VB (transbordo)
-   BD2VB → AG7BH (estación de llegada)

Espero que esta información te sea útil.
```

Cabe destacar que le modelo aunque alucino, identifico correctamente las 
estaciones más allá que uso líneas incorrectas.

La pregunta generica para todos los mdoelo fue:

```
{ "lineas": { "amarilla": 
{ "estaciones": [ "AA1SC", "AB2SC", "AC3SC", "AD4RF", "AE5VE", "AF6SC", "AG7BH" ], 
"sentido_ida": [ "AA1SC", "AB2SC", "AC3SC", "AD4RF", "AE5VE", "AF6SC", "AG7BH" ], 
"sentido_vuelta": [ "AG7BH", "AF6SC", "AE5VE", "AD4RF", "AC3SC", "AB2SC", "AA1SC" ] },
 "azul": { "estaciones": [ "BA1SC", "BB2OC", "BC3SC", "BD2VB", "BE4RC", "BF5SC", "BG6SC", "AG7BH" ], 
 "sentido_ida": [ "BA1SC", "BB2OC", "BC3SC", "BD2VB", "BE4RC", "BF5SC", "BG6SC", "AG7BH" ], ...

¿Quiero ir de RD3VC a AG7BH? ¿Qué líneas debo coger, y cuantos transbordos habrá?
```
### Respuesta ChatGPT-5

Tuvo alucinacion completa, trasbordos inexistentes y lineas incorrectas.

```
Para ir de RD3VC a AG7BH, tienes varias opciones de líneas, pero necesitarás hacer al menos un transbordo. Aquí están las líneas y los transbordos necesarios:

Línea Roja (sentido ida)

Desde RD3VC, puedes ir en la Línea Roja hasta RG6SC.

Luego, puedes hacer transbordo a la Línea Azul en AG7BH.

Transbordos:

1 transbordo en RG6SC de la Línea Roja a la Línea Azul.

Resumen:

Líneas a coger: Roja → Azul.

Transbordos: 1 (en RG6SC).

Este es el camino más directo para ir de RD3VC a AG7BH.
```

### Respuestas Mistral

Respuesta Correcta, sin alucinaciones:
```
¡Perfecto! Vamos a planificar tu ruta de **RD3VC** a **AG7BH** usando las líneas de metro disponibles.

### **Ruta recomendada:**
1. **Línea Verde:** Toma la línea verde en **RD3VC** con dirección a **VF6SC** (sentido ida).
   - Bájate en **AE5VE** (2 paradas después).

2. **Transbordo a Línea Amarilla:** En **AE5VE**, cambia a la línea amarilla con dirección a **AG7BH** (sentido ida).
   - Bájate en **AG7BH** (2 paradas después).

### **Resumen:**
- **Transbordos:** **1** (en AE5VE)
- **Líneas a usar:** **Verde → Amarilla**
- **Trayecto:** RD3VC → AE5VE (verde) → AG7BH (amarilla)

¿Te gustaría que te indique también el tiempo estimado o prefieres otra alternativa de ruta?
```

### Claude con Sonnet 4.5

Respuesta correcta sub alucinaciones.
```
Para ir de **RD3VC a AG7BH**, aquí está tu ruta:

## Ruta óptima:

1. **Línea Roja** (sentido ida): RD3VC → RE5SC → AD4RF
2. **Transbordo en AD4RF**
3. **Línea Amarilla** (sentido ida): AD4RF → AE5VE → AF6SC → AG7BH

## Resumen:
- **Líneas a coger**: Roja → Amarilla
- **Número de transbordos**: 1 (en la estación AD4RF)
- **Estaciones totales**: 6 estaciones (3 en cada línea)

Esta es la ruta más directa, ya que AD4RF es una estación de intercambio entre la línea Roja y la Amarilla.
```


### Gemini
Respuesta correcta sin alucinaciones:
```
Para ir de **RD3VC** a **AG7BH**, la mejor ruta requiere **1 transbordo** y usarás las líneas **Verde** y **Amarilla** (o **Roja** y **Amarilla**).

## Opción 1: Verde $\rightarrow$ Amarilla (1 Transbordo)

Esta es la ruta más directa con un solo transbordo.

1.  **Línea Verde** (Desde **RD3VC**):
    * Sube en **RD3VC** (estación de origen).
    * Viaja 2 paradas en sentido\_vuelta hasta la estación **AE5VE**.
2.  **Transbordo** en **AE5VE**:
    * Cambia de la Línea Verde a la **Línea Amarilla**.
3.  **Línea Amarilla** (Hasta **AG7BH**):
    * Sube en **AE5VE**.
    * Viaja 2 paradas en sentido\_ida hasta la estación **AG7BH** (estación de destino).

---

## Opción 2: Roja $\rightarrow$ Azul (1 Transbordo)

Otra opción con un solo transbordo utilizando la conexión de la Línea Roja con la Azul en **BE4RC**.

1.  **Línea Roja** (Desde **RD3VC**):
    * Sube en **RD3VC** (estación de origen).
    * Viaja 1 parada en sentido\_vuelta hasta la estación **BE4RC**.
2.  **Transbordo** en **BE4RC**:
    * Cambia de la Línea Roja a la **Línea Azul**.
3.  **Línea Azul** (Hasta **AG7BH**):
    * Sube en **BE4RC**.
    * Viaja 4 paradas en sentido\_ida hasta la estación **AG7BH** (estación de destino).

---

## Opción 3: Verde $\rightarrow$ Azul (1 Transbordo)

Otra ruta que también te lleva con un solo transbordo.

1.  **Línea Verde** (Desde **RD3VC**):
    * Sube en **RD3VC** (estación de origen).
    * Viaja 1 parada en sentido\_vuelta hasta la estación **BD2VB**.
2.  **Transbordo** en **BD2VB**:
    * Cambia de la Línea Verde a la **Línea Azul**.
3.  **Línea Azul** (Hasta **AG7BH**):
    * Sube en **BD2VB**.
    * Viaja 5 paradas en sentido\_ida hasta la estación **AG7BH** (estación de destino).

```
| **Ruta más corta**              | **Líneas**          | **Transbordos** | **Estaciones de Transbordo** |
| :------------------------------ | :------------------ | :-------------- | :--------------------------- |
| **Verde → Amarilla**            | Verde, Amarilla     | **1**           | **AE5VE**                    |
| **Roja → Azul**                 | Roja, Azul          | **1**           | **BE4RC**                    |
| **Verde → Azul**                | Verde, Azul         | **1**           | **BD2VB**                    |

## Prueba 2 corrigiendo errores de ortografía en la pregunta

Las Alucinaciones empeoraron
```
Relevant Chunks:
Similarity: 0.6926
Chunk: Línea Azul del metro:
        - Estaciones: BA1SC, BB2OC, BC3SC, BD2VB, BE4RC, BF5SC, BG6SC, AG7BH
        - Sentido ida: BA1SC, BB2OC, BC3SC, BD2VB, BE4RC, BF5SC, BG6SC, AG7BH
        - Sentido vuelta: AG7BH, BG6SC, BF5SC, BE4RC, BD2VB, BC3SC, BB2OC, BA1SC
        - Total estaciones: 8

Similarity: 0.6736
Chunk: Línea Amarilla del metro:
        - Estaciones: AA1SC, AB2SC, AC3SC, AD4RF, AE5VE, AF6SC, AG7BH
        - Sentido ida: AA1SC, AB2SC, AC3SC, AD4RF, AE5VE, AF6SC, AG7BH
        - Sentido vuelta: AG7BH, AF6SC, AE5VE, AD4RF, AC3SC, AB2SC, AA1SC
        - Total estaciones: 7

Similarity: 0.6724
Chunk: Línea Roja del metro:
        - Estaciones: RA1SC, RB2SC, BE4RC, RD3VC, RE5SC, AD4RF, RG6SC
        - Sentido ida: RA1SC, RB2SC, BE4RC, RD3VC, RE5SC, AD4RF, RG6SC
        - Sentido vuelta: RG6SC, AD4RF, RE5SC, RD3VC, BE4RC, RB2SC, RA1SC
        - Total estaciones: 7

Chatbot response:
Para ir de RD3VC a AG7BH, debes tomar la línea Roja del metro en sentido ida.

En este caso, tienes que tomar la línea Roja (RD3VC) en sentidos hacia el centro de la ciudad. La ruta más directa es:

1. Toma la línea Roja (RD3VC) desde RD3VC.
2. Caminando 1 kilómetro hacia el centro, llegas a la estación BB2OC.
3. En BB2OC, cambia a la línea Azul del metro en sentido vuelta y continúa hasta la estación de transbordo AG7BH.

De esta forma, hay un total de 1 transbordo (a partir de RD3VC hasta BB2OC) y 1 línea utilizada en la ruta.

Process finished with exit code 0
```
## Prueba 3: Centrados solo en solucionar la alucinacion y el RAG
Lon hizo bien, errores de prompting corregibles
```
Toma Roja (ida) de RD3VC a AD4RF. Transborda a Amarilla (ida) hasta AG7BH. Total: 1 transbordo.  Ruta sugerida: 
RD3VC → RE5SC → AD4RF (Roja ida), transbordo → AE5VE → AF6SC → AG7BH (Amarilla ida). 1 línea. 1 transbordo. 

Nota: La respuesta se basa en el razonamiento proporcionado en el ejemplo 1. La ruta sugerida es una posible 
solución para llegar de RD3VC a AG7BH. Si hay otras rutas más eficientes, no se mencionan en la respuesta proporcionada. 
Ruta sugerida: RD3VC → RE5SC → AD4RF (Roja ida), transbordo → AE5VE → AF6SC → AG7BH (Amarilla ida). 
1 línea. 1 transbordo. 1 transbordo. 1 transbordo. 1 transbordo. 1 transbordo. 1 transbordo. 1 transbordo. 1 transbordo. 1 transb
```