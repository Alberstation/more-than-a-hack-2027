# Imputación del balón (y de agentes) con información faltante, y su uso para posesión/eventos (2020-2026)

## 1. Métodos que imputan balón/agentes en tracking deportivo y errores frente a la longitud del hueco

### Takeaway
Hay cuatro familias: (a) inferencia del balón solo desde jugadores (Ball Radar, TranSPORTmer, multimodal 2025); (b) imputadores multiagente generales (Graph Imputer, NAOMI, MIDAS/DBHP, UniTraj); (c) difusión con incertidumbre (U2Diff/U2Diffine); (d) filtros de estados híbridos para el balón (Yandex 2025). Sin nada de balón, el mejor ADE publicado es ~1.2-1.9 m con vídeo, o ~1.7-2.0 m solo con trayectorias. Con un 10-20 % de balón observado, el error baja a 1.3-2.1 m incluso en modelos antiguos. El error de los jugadores crece casi linealmente con la longitud del hueco.

### Cited Findings
- **Ball Radar (KDD'23, Kim et al.)**: Set Transformer y Bi-LSTM jerárquico. Predice primero quién posee el balón (paso intermedio) y después la trayectoria. Sin balón: PE 3.69 m, PPA (jugador) 64.0 %, TPA (equipo) 85.3 % (H-LSTM). Con RL: 3.66 m, 64.7 %, 85.9 %. El postprocesado por reglas sube el PE a 4.07 m pero hace posible detectar eventos. Datos: 15 partidos de GPS de Fitogether (K League 2020) más las 3 muestras de Metrica. Código: github.com/pientist/ballradar — [arXiv 2306.08206](https://arxiv.org/pdf/2306.08206)
- **Ball Radar con balón parcial (Tabla 3, antes del PP)**. Se entrena enmascarando el 80 % del balón en la mitad de los batches y el 100 % en la otra mitad:

  | Balón enmascarado | PE | PPA | TPA |
  |---|---|---|---|
  | 100 % | 5.35 m | 61.9 % | 84.0 % |
  | 95 % | 3.00 m | 79.5 % | 91.5 % |
  | 90 % | 2.09 m | 87.1 % | 95.0 % |
  | 80 % | 1.31 m | 93.1 % | 97.4 % |

  — [arXiv 2306.08206](https://arxiv.org/pdf/2306.08206)
- **Multimodal con pre-entrenamiento enmascarado (arXiv 2512.19528, dic-2025)**. Entradas: trayectorias de jugadores, tipo (equipo) y recortes de imagen por jugador. No usa balón en inferencia. Arquitectura: set-attention temporal y social, con cabezas para trayectoria, estado del balón y poseedor. CropDrop enmascara las features visuales de jugadores (ratio 0.5). Resultados:
  - Mejor modelo: ADE 1.17 m, MaxErr 4.77 m, estado del balón 85.85 %, poseedor 89.56 %.
  - Ball Radar reentrenado: ADE 1.86 m, poseedor 78.12 %. TranSPORTmer: ADE 1.72 m, estado 79.77 %, poseedor 83.79 %.
  - Ablación: solo trayectorias da ADE 1.97 m, estado 77.8 %, poseedor 82.1 %. Con tipo de jugador: 1.78 m, 79.6 %, 83.7 %.
  - Datos propietarios: 283 partidos, 6.25 Hz, ventanas de 9.6 s. No se indica ni código ni cómputo. — [arXiv 2512.19528](https://arxiv.org/html/2512.19528v1)
- **MIDAS/DBHP (Choi, Kim, … Ko; arXiv 2408.10878)**. Set Transformer equivariante a permutaciones más Bi-LSTM por jugador. Predicción inicial, más acumulación de derivadas hacia delante y hacia atrás (velocidad y aceleración), con un ensemble ponderado aprendido. Imputa jugadores, no el balón. — [arXiv 2408.10878](https://arxiv.org/pdf/2408.10878v4)
  - Fútbol (Metrica, 3 partidos, 10 Hz, ventanas de 20 s). PE en m de MIDAS frente a interpolación lineal (LI) y spline cúbica (CS):
    - uniforme: 1.32 frente a LI 3.84 y CS 2.21;
    - por agente: 1.98 frente a LI 3.11 y CS 1.92;
    - cámara de TV simulada: 1.23 frente a LI 3.52 y CS 2.21.
  - BRITS, NAOMI y CSDI a menudo no superan a LI/CS. Graph Imputer queda peor que LI/CS en muchos casos, "como ya reportaba su artículo original".
  - Error según la longitud del hueco (fútbol, por agente): 33 frames (3.3 s) da 0.05 m; 90 frames (9 s) da 0.70 m; 173 frames (17 s) da 2.61 m.
  - Masking dinámico del 10-90 % en train y del 50 % en test.
  - Coste: 3.9 M parámetros, 24 ms por ventana de 20 s en TITAN RTX, 8.2 s por media parte.
  - Código: github.com/gkswns95/midas. Licencia del paper: CC BY-NC-ND 4.0.
- **UniTraj / Sports-Traj (arXiv 2405.17680)**. CVAE con Transformer, *Ghost Spatial Masking* (max-pool invariante al orden) y Mamba bidireccional temporal. Ejemplos de masking: huecos centrales de 25-40 frames, 5 agentes completos ocultos, 50-80 % discreto.
  - minADE20 sobre todos los agentes, balón incluido. No se reporta el error del balón por separado.
  - Soccer-U usa SoccerTrack, en píxeles: UniTraj 94.6, GC-VRNN 105.9, NAOMI 145.2.
  - Entrenado en 1 A100, 100 épocas, batch 128.
  - Código, pesos y datos: github.com/colorfulfuture/UniTraj-pytorch. — [arXiv 2405.17680](https://arxiv.org/html/2405.17680v2)
- **U2Diff / U2Diffine (Capellera et al., IRI-UPC y Kognia; arXiv 2503.18589 y 2605.10717)**. Difusión estilo CSDI con Mamba y Social Transformer. Predice una varianza heteroscedástica por estado (x, y), y RankNN asigna una probabilidad de error a cada uno de los 20 modos. — [arXiv 2503.18589](https://arxiv.org/html/2503.18589v2); [arXiv 2605.10717](https://arxiv.org/html/2605.10717)
  - minADE20: Football-U 2.38 yd frente a 3.55 de UniTraj; Soccer-U 52.9 px frente a 94.6.
  - Cobertura Mahalanobis del 95 % superior al 95 % en Football-U y Soccer-U.
  - Spearman entre la incertidumbre media y el error: 0.32-0.41. Con RankNN: 0.58-0.79.
  - No hay métricas separadas para el balón.
  - Cómputo: RTX A6000, 100 épocas, 50 pasos de difusión, batch 16 en fútbol. No se encontró código enlazado.
- **Graph Imputer (DeepMind y Liverpool FC, Sci. Rep. 2022)**. Bi-VLSTM con graph networks. Aprende una distribución de trayectorias de jugadores fuera de cámara sobre 105 partidos, y extiende pitch control a escenarios parcialmente observados. — [Sci. Rep.](https://www.nature.com/articles/s41598-022-12547-0); [arXiv 2106.04219](https://arxiv.org/abs/2106.04219)
- **Filtro híbrido para el balón (Yandex, arXiv 2506.07981)**. Estados discretos: vuelo (J), posesión de un jugador (P), espera tras la posesión (W) y fuera del campo (O). Guarda los K≈1000 mejores hipótesis por log-verosimilitud, con un buffer de latencia de L frames.
  - Si el balón no se ve, J se extrapola con una varianza creciente y P se genera a partir de las posiciones de los jugadores.
  - Resultados: Acc@1 m 0.66 y F1 de patada 0.74, frente a 0.66 de la MIP de Maksai.
  - Más de 50 fps en un solo núcleo de CPU. Es determinista por construcción, pero no se ha publicado código. — [arXiv 2506.07981](https://arxiv.org/html/2506.07981v1)
- **TranSPORTmer (ACCV'24)**. Set Attention Blocks con máscara de entrada. Cubre inferencia e imputación del balón en fútbol y baloncesto. — [ACCV 2024](https://openaccess.thecvf.com/content/ACCV2024/papers/Capellera_TranSPORTmer_A_Holistic_Approach_to_Trajectory_Understanding_in_Multi-Agent_Sports_ACCV_2024_paper.pdf)

### Inferences
- La interpolación lineal o spline es una línea base fuerte para jugadores en huecos de menos de 5 s. Lo que aportan las redes está en huecos de más de 9 s (MIDAS: 0.7-2.6 m frente a errores mucho mayores de LI).
- Para el balón, la curva de Ball Radar muestra que lo que más vale son unos pocos anclajes reales. Pasar del 0 % al 10 % de balón observado sube la TPA del 84 % al 95 %. Nuestro balón está observado en el 59 % de los frames, así que estamos en el régimen de "balón parcial". El valor marginal de imputar se concentra en los huecos largos, que en nuestro caso son sobre todo de balón muerto.

### Gaps
- No encontré curvas de error del **balón** frente a la longitud del hueco (solo de jugadores, en MIDAS). UniTraj y U2Diff mezclan el balón con todos los agentes.
- No pude extraer las tablas numéricas de Graph Imputer: PMC estaba bloqueado.

## 2. ¿Alguien usa el balón imputado, su incertidumbre o una distribución sobre el portador como entrada para posesión o eventos? ¿Cuánto ayuda?

### Takeaway
La evidencia directa apunta a que "primero el balón, después la posesión" **es peor** que inferir directamente la posesión/portador como variable latente. PathCRF, sobre los mismos datos DFL que usamos, da 75.7 % de F1 de eventos con inferencia directa de la arista de posesión, frente al 48.4 % de la ruta "regresar el balón y luego heurísticas de Ball Radar". El balón ayuda mucho cuando hay anclajes **reales**, no imputados.

### Cited Findings
- **PathCRF (Kim et al., arXiv 2602.12080, feb-2026)**. Datos: Sportec Open DFL (7 partidos; 5 de train, 1 de val y 1 de test). Solo jugadores en inferencia, a 5 fps con ventanas de 10 s. Modela la posesión como la arista (emisor, receptor) entre 676 candidatas, con un CRF enmascarado dinámico y decodificación de Viterbi. — [arXiv 2602.12080](https://arxiv.org/html/2602.12080)
  - Precisión de arista 69.64 % (emisor 80.4 %, receptor 82.3 %). F1 de eventos 75.69 % (precisión 73.2 %, recall 78.4 %).
  - **Ball TP** regresa el balón (2.74 m) y luego aplica heurísticas: precisión de arista 47.0 % y F1 de eventos 48.4 %.
  - Sin CRF: 65.3 %. Sin CRF pero con Viterbi: 73.1 %. Con decodificación voraz: 70.9 %.
  - Posesión de equipo por frame: **92.40 %** (cuota local real 62.1 %, predicha 62.6 %).
  - Backbone TranSPORTmer: 68.2 %. Backbone Ball Radar: 75.2 %.
  - Coste: RTX 4090, 565 s por época a 5 fps, 23 s de inferencia por partido. Código: github.com/hyunsungkim-ds/pathcrf.
- **Ball Radar, detección de pases sobre el balón predicho** (regla: toque de A, balón en tránsito, toque de B; tolerancia de ±2 s). F1 de pase según el balón enmascarado:

  | Balón enmascarado | F1 de pase | F1 de emisor | F1 de receptor |
  |---|---|---|---|
  | 100 % | 0.39 | 0.60 | 0.57 |
  | 95 % | 0.60 | – | – |
  | 90 % | 0.74 | – | – |
  | 80 % | **0.87** | 0.91 | 0.90 |

  — [arXiv 2306.08206](https://arxiv.org/pdf/2306.08206)
- Ball Radar suma las probabilidades de posesión por jugador para obtener la del equipo (TPA). Es decir, una distribución sobre el portador alimenta directamente la posesión de equipo. — [arXiv 2306.08206](https://arxiv.org/pdf/2306.08206)
- **Multimodal 2025**. La posesión es una salida conjunta con la trayectoria y el estado del balón (vivo/muerto, etc.): el poseedor pasa de 82.1 % (solo trayectorias) a 89.6 % (con recortes de vídeo y CropDrop). El estado del balón pasa de 77.8 % a 85.9 %. — [arXiv 2512.19528](https://arxiv.org/html/2512.19528v1)
- MIDAS usa las trayectorias imputadas en tareas posteriores (pitch control y distancia/sprints), no en posesión. — [arXiv 2408.10878](https://arxiv.org/pdf/2408.10878v4)
- En el filtro Yandex, la posesión es un modo discreto del estado del balón. La detección de patadas sale de las transiciones P→J (F1 0.74). — [arXiv 2506.07981](https://arxiv.org/html/2506.07981v1)

### Inferences
- Encaja con nuestro +0.026 de spellF1 al simular balón recuperado de vídeo. Imputar una posición puntual del balón y dársela al clasificador añade poco, y en PathCRF la ruta "balón primero" es claramente peor.
- La palanca con evidencia es tratar el **portador/arista** como el estado latente del decodificador. En nuestro caso eso sería ampliar los 3 estados H/A/D a estados por jugador-tracklet, o por pares emisor→receptor, con restricciones de transición. PathCRF gana unos 10 puntos de F1 al pasar de no-CRF a CRF enmascarado.
- Para posesiones de menos de 2 s (pérdidas y despejes), lo que discrimina en estos papers es la transición de arista: quién está cerca en el momento del toque. Una distribución sobre el portador, con el balón observado como anclaje cuando existe, es lo que más se parece a lo que funciona.

### Gaps
- No encontré trabajos que den la **varianza** del balón imputado (U2Diff) como entrada a un clasificador de posesión y midan la ganancia.
- No hay F1 por duración de posesión: ninguna fuente informa del recall en posesiones cortas.

## 3. Métodos con jugadores anónimos (invariantes a permutación) y sin identificar al portero

### Takeaway
Casi todos los métodos recientes son invariantes o equivariantes a permutación dentro de cada equipo (Set Transformer, set-attention o max-pooling). Pero todos usan la **etiqueta de equipo** y trayectorias de 10-20 s por agente. Ninguno trabaja con tracklets que aparecen y desaparecen de forma nativa: lo resuelven con máscaras.

### Cited Findings
- Ball Radar es "parcialmente invariante a permutación" (dentro del equipo) para la trayectoria, y parcialmente equivariante para el poseedor. Los porteros se trataron aparte: el GPS de Fitogether no mide al portero, así que entrenaron un regresor de trayectoria del portero en Metrica (error de 5.4 m en el test de Metrica) y lo aplicaron a Fitogether. — [arXiv 2306.08206](https://arxiv.org/pdf/2306.08206)
- MIDAS: Set Transformer equivariante. Los baselines no equivariantes (BRITS, NAOMI, CSDI) se tuvieron que ordenar por la suma de la x y la y medias para no depender del orden. — [arXiv 2408.10878](https://arxiv.org/pdf/2408.10878v4)
- UniTraj: el token "ghost" se obtiene por max-pool sobre la máscara de agentes. La identidad solo se codifica como balón, ataque o defensa. El tipo de pooling apenas cambia el resultado (max 4.77, sum 4.79, mean 4.80). — [arXiv 2405.17680](https://arxiv.org/html/2405.17680v2)
- PathCRF: grafo completo dinámico sobre los jugadores. Las 676 aristas (26²) incluyen nodos especiales como fuera de juego. — [arXiv 2602.12080](https://arxiv.org/html/2602.12080)
- El modelo multimodal usa el tipo de jugador (one-hot de equipo), que mejora el poseedor de 82.1 % a 83.7 %. — [arXiv 2512.19528](https://arxiv.org/html/2512.19528v1)

### Inferences
- Nuestro encoder DeepSets/TCN ya va en esa línea. Lo que no está resuelto en la literatura es la identidad temporal: como todos asumen agentes persistentes, con tracklets cortos haría falta una ventana corta o atención por frame con enlace débil.
- Lo del portero se puede sustituir por rasgos geométricos (el jugador más profundo de cada equipo). Ball Radar muestra que incluso imputar al portero desde otro proveedor es viable, aunque con unos 5 m de error.

### Gaps
- No encontré evaluaciones con tracklets fragmentados (IDs que cambian) en tareas de posesión.

## 4. Entrenar en tracking limpio externo con masking artificial y transferir a otro proveedor

### Takeaway
El masking artificial durante el entrenamiento es el procedimiento estándar en todos los métodos (MIDAS, UniTraj, Ball Radar). Las pruebas entre proveedores son escasas. Ball Radar mezcla Fitogether y Metrica, y transfiere un modelo de portero de Metrica a GPS. PathCRF menciona generalización a un dataset no público en un apéndice que no pude leer.

### Cited Findings
- MIDAS aplica en entrenamiento una tasa de faltantes dinámica entre el 10 y el 90 %, y evalúa con un 50 % fijo, para generalizar entre tasas. También simula una cámara de TV que sigue al balón. — [arXiv 2408.10878](https://arxiv.org/pdf/2408.10878v4)
- MIDAS es eficiente con pocos datos: en baloncesto compara 3 partidos frente a 70, y lo motiva así: "en fútbol y fútbol americano los datos de entrenamiento son limitados". — [arXiv 2408.10878](https://arxiv.org/pdf/2408.10878v4)
- Ball Radar entrena con 10 partidos de GPS de Fitogether más 2 de Metrica, y evalúa en otros 3 de Fitogether más la segunda parte de la Sample Game 3 de Metrica. Remuestrea Metrica a la frecuencia de Fitogether. — [arXiv 2306.08206](https://arxiv.org/pdf/2306.08206)
- PathCRF: el apéndice B evalúa robustez y generalización en un dataset no público. Las cifras no estaban en el texto que leí. El texto principal habla de "rendimiento fuerte con solo cinco partidos". — [arXiv 2602.12080](https://arxiv.org/html/2602.12080)
- No encontré ningún paper que entrene en un proveedor (p. ej. DFL) y evalúe sin ajuste en otro (p. ej. SkillCorner o tracking de vídeo). La página de SkillCorner afirma que usa un modelo de extrapolación de jugadores fuera de cámara, pero eso es marketing, no una evaluación. — [SkillCorner](https://skillcorner.com/products/football/xy-tracking-data)

### Inferences
- Los papers no modelan nuestro tipo de degradación: balón suavizado (unas 3 veces), huecos de balón concentrados en balón muerto y tracklets anónimos. Hay que imitarla en la generación de máscaras, con huecos condicionados al estado del juego en lugar de uniformes.
- PathCRF está entrenado en exactamente los mismos 7 partidos de DFL que tenemos. Es un punto de referencia directo: 92.4 % de posesión de equipo por frame sin balón, con tracking limpio de jugadores e identidades.

### Gaps
- No hay cifras publicadas de transferencia entre proveedores de un imputador de balón o de posesión.

## 5. Coste de entrenamiento en una GPU de 8 GB e inferencia determinista en CPU

### Takeaway
Los modelos de tipo Set Transformer con LSTM (Ball Radar, MIDAS, PathCRF) son pequeños (unos 4 M de parámetros). Encajan en 8 GB con batches moderados y en horas de entrenamiento con unos pocos partidos. La difusión (U2Diff) y NRTSI son más caros en inferencia. La única alternativa nativa de CPU determinista es el filtro de estados discretos (estilo Yandex/HMM).

### Cited Findings
- MIDAS: 3.95 M de parámetros, 24 ms por ventana de 20 s y 8.2 s por media parte en TITAN RTX (24 GB). CSDI: 0.41 M de parámetros pero 394 ms por ventana. NRTSI: 84 M de parámetros y 893 ms por ventana. — [arXiv 2408.10878](https://arxiv.org/pdf/2408.10878v4)
- PathCRF: RTX 4090, 565 s por época a 5 fps con 5 partidos, y 23 s de inferencia por partido. — [arXiv 2602.12080](https://arxiv.org/html/2602.12080)
- UniTraj: 1 A100, 100 épocas, batch 128. — [arXiv 2405.17680](https://arxiv.org/html/2405.17680v2)
- U2Diffine: unas 4 veces más lento que U2Diff (31 ms frente a 8 ms por modo con batch 128 en A6000), con 50 pasos de difusión en entrenamiento y 6 en muestreo. — [arXiv 2605.10717](https://arxiv.org/html/2605.10717); [arXiv 2503.18589](https://arxiv.org/html/2503.18589v2)
- Filtro Yandex: más de 50 fps en un núcleo de Xeon E5-2660, sin GPU. — [arXiv 2506.07981](https://arxiv.org/html/2506.07981v1)

### Inferences
- Un Set Transformer pequeño con Bi-GRU (unos 4 M de parámetros), a 5 fps y en ventanas de 10 s, sobre 10 partidos (7 de DFL y 3 de Metrica) debería caber en 8 GB. Por las cifras de PathCRF serían unos minutos por época, aunque no lo he medido. Su forward se puede reexportar a numpy en float64 para tener inferencia determinista en CPU, igual que ya hacemos.
- Las LSTM y las atenciones son reproducibles en numpy si se fijan los pesos y el orden de las operaciones. La difusión requiere muestreo, que es reproducible con semilla, pero cuesta K×S forwards.

### Gaps
- Ningún paper informa de memoria máxima ni de un entrenamiento en 8 GB. Las cifras de arriba son en GPU de 24-48 GB.
- Licencias del código: MIDAS, Ball Radar, PathCRF y UniTraj tienen repositorio público, pero no verifiqué su licencia de software. Los papers son CC BY-NC-ND o CC BY-NC-SA.
