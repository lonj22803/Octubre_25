# Semana II  Octubre de 2025 

## Temas relevantes reunion 6/Oct/25

- **Cruce de informacion**
    - Se verifico que Mistral, Gemini, Claude y ChatGPT (el ultimo con errores), son capaces de resolver un grafo de 
manera simple, organizada y correcta, entregando rutas correctas y optimizadas.
    - Se desea integrar informacion de hoteleria, restaurantes, incluso un punto donde se crucen tres lineas diferentes de un
metro ficticio. Para eso se extendera el experimiento generico de un escenario controlado y sobre el cual tendremos informacion veridica y
verificable.

- **Mejora al sistema de transporte**
    - Se agregaron dos estaciones a linea naranja, con una particularidad y es que llega a la estacion RD3VC, es decir, dentro de su
codificacion no esta el OC que indicaria que esa linea pasa por ahi, lo cual rompe con patrones, es un caso que puede ocurrir en lña vida real, pero ayuda
para que el modelo no se confie en patrones y pueda razonar.

![Sistema Generico](week_two/sistema_generico.png "Sistema Generico")
Sistema Generico, con las dos nuevas estaciones creadas para la linea naranja.

- **Simulacion de longitud y latitud**
    - Para generar el gráfico anterior se usaron coordenadas ficticias, las cuales ya predefinían la distribucion, pero esas coordenadas 
eran completamente arbitrarias, y no se podían comparar con un caso realista. Por ende se procedió a hacer una conversión a coordenadas más realistas 
para una ciudad cuyo tamaño fuera considerable. Se seleccionó Madrid como ciudad de referencia y se aplicó una transformación lineal donde cada unidad 
en las coordenadas originales equivale a 0.0001 grados geográficos (aproximadamente 11,13 metros), resultando en un sistema que cubre un área de 55×62 km 
con distancias promedio de 11 km entre estaciones, representativo de una red de transporte metropolitana de gran escala.
    - *Calculo*:  
  Transformacion lineal:  
  La conversión se realizó mediante una transformación afín que preserva la distribución geométrica original mientras escala las coordenadas a un sistema 
de referencia geográfico real:
      $$ Latitud = Latitud_central + y * s $$
      $$ Longitud = Longitud_central + x * s $$
        - Donde (x, y) coordenadas cartesianas originales
        - Latitud_central y Longitud_central = (40.4168, -3.7038) (coordenadas de Madrid)
        - s = 0.0001 (factor de escala en grados por unidad)
    - *Parámetros de Escala y Equivalencias Físicas*
  El facto de escala s=0.001 se seleccionó considerando que:
  $$
\begin{align*}
1^\circ \text{de latitud} &\approx 111,3 \text{km} \\
1 \text{unidad cartesiana} &= s^\circ = 0.0001^\circ \approx 11,13 \text{m}
\end{align*}
$$
    - *El resultado fue el siguiente*:

        ```
        AB2SC,40.4128,-3.6988,amarilla
        AC3SC,40.4168,-3.6918,amarilla
        AD4RF,40.4208,-3.6858,"amarilla, roja"
        AE5VE,40.4228,-3.6798,"amarilla, verde"
        AF6SC,40.4268,-3.6718,amarilla
        AG7BH,40.4288,-3.6638,"amarilla, azul"
        BA1SC,40.4348,-3.7158,azul
        BB2OC,40.4368,-3.7098,"azul, naranja"
        BC3SC,40.4388,-3.7038,azul
        BD2VB,40.4408,-3.6978,"azul, verde"
        BE4RC,40.4388,-3.6918,"azul, roja"
        BF5SC,40.4348,-3.6838,azul
        BG6SC,40.4328,-3.6748,azul
        RA1SC,40.4428,-3.6798,roja
        RB2SC,40.4468,-3.6858,roja
        RD3VC,40.4328,-3.6938,"roja, verde, naranja"
        RE5SC,40.4248,-3.6918,roja
        RG6SC,40.4048,-3.6918,roja
        VA1SC,40.4528,-3.7038,verde
        VD4SC,40.4288,-3.6858,verde
        VF6SC,40.4168,-3.6778,verde
        OA1SC,40.4548,-3.7198,naranja
        OB2SC,40.4448,-3.7158,naranja
        OC3SC,40.4248,-3.7038,naranja
        ```
    - *Visualización Geográfica*
  Si graficaramos las estaciones en un mapa de Madrid, obtenemos la siguiente distribución:
    ![Sistema Generico Coordenadas Reales](week_two/estaciones_sobre_mapa_madrid.png "Sistema Generico Coordenadas Reales")
    Sistema Generico con coordenadas geográficas simuladas, representando una red de transporte metropolitana en Madrid.

  
