# Adaptación entre proveedores de tracking y robustez por degradación (posesión/eventos en fútbol, 2021-2026)

Contexto propio: se entrena solo con datos externos (7 partidos IDSSE/DFL con TRACAB Gen5 y 3 partidos de muestra de Metrica). Se predice un partido con tracker automático: el balón falta en el 41 % de los frames evaluados, en huecos largos, y hay unos 20,5 de 22 jugadores visibles. Medido internamente: IDSSE→Metrica pierde unos 0,1 de spellF1, y añadir Metrica al encoder recupera +0,01-0,02. No se permite TTA ni self-training sobre el partido objetivo. La normalización por partido en inferencia sí está permitida.

## 1. Caídas reportadas entre proveedores y qué las reduce

### Takeaway
La única cifra publicada y directamente comparable es la de PathCRF (2026). En detección de eventos con balón solo a partir de jugadores, el F1 cae de 73,9 a 49,2 (-24,7 pt) al pasar de Sportec/DFL a K League (GPS), y de 79,8 a 66,3 en la dirección contraria. Los autores no proponen ningún remedio. Nuestra caída (~0,1 spellF1) es menor que la suya, y la mejora por entrenar con varias fuentes es la única con evidencia, aunque sea indirecta.

### Cited Findings
- PathCRF, Event F1 (media±SD), según la Tabla 6 del apéndice B — [PathCRF, arXiv 2602.12080](https://arxiv.org/html/2602.12080):
  - K League→K League: 79,83±2,60 %
  - Sportec→Sportec: 73,88±1,90 %
  - K League→Sportec: 66,25±7,65 %
  - Sportec→K League: 49,16±3,37 %
  - La precisión de la arista de posesión del modelo entrenado en Sportec baja de 66,75±2,76 % a 41,65±2,34 %.
- PathCRF usa K League 1 con GPS a 10 Hz (15 partidos, split 10:2:3) y Sportec con leave-one-match-out sobre 7 partidos. Atribuye la caída a que "the two providers differ in collection modality (wearable GPS vs. camera-based tracking) and annotation conventions", lo que produce "distinct noise patterns and ground-truth labels". No prueba fine-tuning, normalización ni augmentación. Lo justifica en que el usuario previsto, el proveedor, entrena con sus propios datos — [PathCRF](https://arxiv.org/html/2602.12080).
- El preprocesado de PathCRF baja de 25 a 5 FPS, usa ventanas de 10 s con stride de 5 frames y sincroniza eventos con ELASTIC. No incluye ninguna augmentación de robustez — [PathCRF](https://arxiv.org/html/2602.12080).
- La asimetría que muestra PathCRF (entrenar en el dominio más ruidoso, GPS, generaliza mejor hacia el limpio que al revés) es coherente con la idea de entrenar con datos degradados — [PathCRF](https://arxiv.org/html/2602.12080). Es una inferencia sobre sus números, no una afirmación de los autores.
- AdaSports-Traj (trayectorias multiagente, 2025) entrena con baloncesto, fútbol americano y fútbol fusionados y evalúa en cada test sin usar datos del objetivo (U2S). En minADE20, UniTraj empeora de 3,55 a 11,89 en Football-U al pasar de S2S a U2S, y AdaSports-Traj de 3,04 a 9,04. En Soccer-U, UniTraj va de 94,59 a 124,56 y AdaSports de 91,52 a 104,66. Su remedio combina un adapter con embeddings de rol y de dominio (cross-attention con gate sigmoide por token) y un contraste jerárquico con dos cabezas InfoNCE. Necesita etiquetas de dominio en el entrenamiento, no datos del objetivo — [AdaSports-Traj, arXiv 2509.16095](https://arxiv.org/html/2509.16095v1).
- Mills et al. (Sports Engineering 29(1):15, 2026) detectan eventos automáticamente con tracking derivado de broadcast en un partido del Mundial 2022 y comparan con óptico multicámara y con los eventos de FIFA. Concluyen que con la Camera 1 la precisión puede acercarse a la del óptico en algunos tipos de evento. Los errores vienen de la visibilidad de jugadores, de errores en el tracking del balón y de definiciones de evento subjetivas, sobre todo en balón parado y tiros — [Mills et al., SHU repo](https://shura.shu.ac.uk/37410/). No obtuve las cifras por tipo de evento porque el PDF está tras un bot-wall.
- Vidal-Codina et al. (arXiv 2202.00804) señalan que los errores de tracking (intercambios de jugador, flags in-play/dead-ball incorrectos) generan eventos erróneos — [arXiv 2202.00804](https://arxiv.org/pdf/2202.00804).

### Inferences
- Nuestra caída IDSSE→Metrica (~0,1) es del mismo orden que la de PathCRF K League→Sportec (-13,6 pt) y menor que Sportec→K League (-24,7 pt). La transferencia al tracker automático probablemente se parece más a este último caso, porque cambian a la vez el ruido, la modalidad y la cobertura del balón. Por eso la degradación sintética es la palanca principal.
- La lección operativa de AdaSports-Traj es que entrenar con varias fuentes ya reduce la caída, y que un embedding de dominio aprendido con varias fuentes ayuda algo más. Con 2 fuentes (IDSSE y Metrica) el embedding de dominio no tiene valor en inferencia, porque el dominio objetivo es desconocido. Lo aprovechable es la mezcla de fuentes y degradaciones.

### Gaps
- No encontré la cifra de Crang et al. 2025 traducida a impacto en modelos de posesión. Solo valida posición y velocidad (ver sección 3).
- No encontré ningún otro paper con cross-provider para posesión o estado (DFL↔Second Spectrum↔SkillCorner) con números.

## 2. Técnicas de generalización de dominio sin datos del objetivo

### Takeaway
En tracking deportivo no hay evidencia publicada y cuantificada de mixup, adversarial noise ni ensembles de degradaciones para posesión. La evidencia disponible es de dominios vecinos: multi-source (AdaSports-Traj), normalización por instancia (RevIN, con una crítica de 2026) y domain randomization (robótica y visión). El orden práctico es: aleatorizar los parámetros de degradación, aplicar dropout de canales o agentes, normalizar por partido y ensamblar.

### Cited Findings
- **Normalización por instancia.** RevIN (ICLR 2022) normaliza cada secuencia por su media y varianza, característica a característica, con un afín aprendible, y desnormaliza la salida. Es agnóstico al modelo y mejora el forecasting bajo distribution shift temporal — [ICLR 2022 poster](https://iclr.cc/virtual/2022/poster/6034); [repo oficial](https://github.com/ts-kim/RevIN).
- Una revisión de marzo de 2026 (arXiv 2603.11869) muestra con ablaciones que varios componentes de RevIN son redundantes o perjudiciales, y separa el shift temporal, el espacial y el condicional de la salida — [arXiv 2603.11869](https://arxiv.org/abs/2603.11869).
- **Multi-source.** Entrenar con varios dominios sin datos del objetivo, más embeddings de rol y dominio, rebaja la degradación cross-domain respecto a UniTraj. Por ejemplo, en Soccer-U U2S el minADE20 baja de 124,56 a 104,66 — [AdaSports-Traj](https://arxiv.org/html/2509.16095v1).
- **Domain randomization.** En una presentación de NASA, la DR ingenua sube la precisión de clasificación sim-to-real del 64 % al 79 %. Es una tarea de clasificación, no de tracking — [NASA NTRS 20250000242](https://ntrs.nasa.gov/citations/20250000242).
- En ProtoMotions se añade ruido de observación con niveles fijados empíricamente, no calibrados — [ProtoMotions DR](https://nvlabs.github.io/ProtoMotions/tutorials/workflows/domain_randomization.html).
- Un tracker activo entrenado en simulación con augmentación de entorno pasó a un robot real sin fine-tuning — [arXiv 1808.03405](https://arxiv.org/pdf/1808.03405).
- **Robustez a agentes ausentes y ruido en trayectorias.** Estos modelos tratan la observación parcial como tarea de entrenamiento enmascarando agentes o tramos:
  - GC-VRNN (CVPR 2023) imputa y predice a la vez, y empeora con huecos largos al inicio de la ventana — [slides CVPR 2023](https://cvpr.thecvf.com/media/cvpr-2023/Slides/23165.pdf).
  - UniMTD (ICCV 2025) cubre niveles arbitrarios de observación ausente — [UniMTD](https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_Unified_Multi-Agent_Trajectory_Modeling_with_Masked_Trajectory_Diffusion_ICCV_2025_paper.pdf).
  - RMP-YOLO añade un módulo de recuperación de historias incompletas delante del predictor — [arXiv 2409.11696](https://arxiv.org/html/2409.11696v2).
  - El modelo de difusión de Kognia/UPC hace forecasting, imputación e inferencia de agentes nunca vistos con incertidumbre por estado — [IRI-UPC](https://www.iri.upc.edu/download/scidoc/3003).
- Kim et al. (KDD 2023) infieren la trayectoria del balón solo a partir de jugadores (Set Transformer + Bi-LSTM jerárquico que primero predice el poseedor, con reality loss). Esto respalda modelar la posesión sin canal de balón en los huecos — [arXiv 2306.08206](https://arxiv.org/abs/2306.08206). No encontré resultados de generalización entre ligas en ese paper.

### Inferences
- **Aleatorizar parámetros en vez de valores fijos.** Hoy usamos 3 degradaciones × 2 patrones de huecos. Muestrear por episodio la amplitud y la constante temporal de la deriva, la tasa de dropout de jugadores (por ejemplo, de 0 a 3 ausentes) y la distribución de huecos (escalada ×0,5-2 sobre la del objetivo) es la versión DR de lo mismo. Usar la distribución de longitudes de hueco del objetivo como estadística de inputs no viola la regla, porque no son etiquetas, pero conviene confirmarlo con la aclaración de los organizadores y documentarlo.
- **Dropout del canal de balón.** Hay que entrenar con un flag explícito de "balón ausente" y con la posición rellenada a 0, no interpolada, para que la red aprenda la rama sin balón, en la línea de Kim 2023 y PathCRF.
- **Normalización por partido.** Estandarizar velocidad, aceleración y distancias por partido (z-score o percentiles) es análogo a RevIN, pero a nivel de partido. La crítica de 2026 sugiere probarlo como ablación y no darlo por bueno.
- **Ensembles.** Promediar probabilidades de modelos entrenados con semillas o mezclas de degradación distintas es barato en CPU. Suele dar mejoras pequeñas y estables. No encontré cifras en deportes.
- **Mixup y adversarial noise.** No encontré evidencia en tracking deportivo. Son de prioridad baja.

### Gaps
- No encontré ninguna publicación que mida mixup, AdvProp ni ensembles de degradaciones en modelos de posesión con tracking.
- No encontré ningún survey específico de domain generalisation en trayectorias deportivas.

## 3. Simulación realista de errores de tracking broadcast/automático

### Takeaway
Las fuentes describen errores con estas características:
- Jugadores fuera de cámara (en TV, por debajo del 50 % de la actividad visible).
- RMSE de posición muy variable (1,68-16,39 m entre proveedores broadcast).
- Fragmentación de tracklets e intercambios de identidad, que es la métrica clave en SoccerNet-GSR.
- Errores del balón y del flag dead-ball.
- Posiciones extrapoladas no observadas (flag is_detected de SkillCorner).

Un simulador fiel debería generar oclusiones correlacionadas espacialmente, como un rectángulo de cámara que sigue al balón, en vez de dropout independiente por jugador.

### Cited Findings
- "Broadcast analytics – an evaluation of video-based tracking systems with constrained player visibility" emula tracking a partir de un scouting feed y de la TV y lo compara con el tracking oficial. El scouting feed captura ~97 % de la actividad de los jugadores y la TV menos del 50 %. La visibilidad afecta mucho a las métricas físicas, mientras que el reconocimiento de formación casi no se ve afectado — [Sponet record](https://www.sponet.de/sponet/Record/4096118?lng=en).
- Crang et al. (2025) validan tres proveedores comerciales de tracking broadcast frente al óptico en un partido del Mundial 2022. La precisión es razonable cuando el jugador está detectado. El RMSE global de posición va de 1,68 m a 16,39 m. Recomiendan el tactical feed a 720p/1080p porque maximiza la detección — [arXiv 2508.19477](https://arxiv.org/pdf/2508.19477v1); [SAGE](https://journals.sagepub.com/doi/10.1177/22150218261445834). El rango de RMSE procede de un resumen secundario que no pude verificar en el PDF.
- SkillCorner extrapola a los jugadores fuera de cámara ("predicts their most likely positions") y los marca con is_detected=False — [SkillCorner XY](https://skillcorner.com/products/football/xy-tracking-data); [fast-forward docs](https://fast-forward.readthedocs.io/en/latest/providers/skillcorner/).
- El estudio de SkillCorner con 10 cámaras reporta identificación de jugadores del ~90-98 % y tracking del balón por encima del 92 % para la mayoría de posiciones de cámara. Es un estudio del propio proveedor — [SkillCorner article](https://skillcorner.com/fr/articles/does-camera-position-and-capture-style-affect-tracking-quality).
- SoccerNet-GSR (CVPRW 2024) mide tracking e identificación en minimapa con GS-HOTA: 200 clips de 30 s, baseline de 23,36 y ganador 2024 (Constructor Tech) con 63,81. La mejora vino de reducir la fragmentación de tracklets. La homografía es inestable cuando hay pocas líneas visibles — [SoccerNet GSR](https://arxiv.org/html/2404.11335v1); [arXiv 2504.06357](https://arxiv.org/pdf/2504.06357); [SoccerNet 2025 results](https://arxiv.org/pdf/2508.19182).
- Que algunos jugadores nunca son visibles en la emisión complica comparar tracking broadcast con la verdad — [arXiv 2110.11107](https://arxiv.org/pdf/2110.11107).
- Los errores de tracking que cambian eventos son los intercambios de jugadores y los flags in-play/dead-ball incorrectos — [arXiv 2202.00804](https://arxiv.org/pdf/2202.00804).

### Inferences (receta de simulación propuesta, priorizada)
1. **Oclusión por campo de visión.** Una ventana rectangular de ~40-60 m de ancho centrada en el balón, con suavizado temporal, hace desaparecer a los jugadores de fuera. Representa mejor el "<50 % visible" de la TV que el dropout aleatorio. En nuestro objetivo hay ~20,5 de 22 visibles, más cerca del scouting feed (~97 %), así que la ventana debe ser ancha. Conviene muestrear el ancho para que el número medio de ausentes quede entre 0,5 y 3.
2. **Fragmentación e intercambios de identidad.** Como el objetivo tiene tracklets anónimos, hay que cortar tracklets en segmentos (longitud geométrica) y permutar los IDs entre segmentos. Si el modelo es un set encoder invariante a permutaciones, esto ya está cubierto en gran parte. Habría que verificar que ninguna feature use la identidad a lo largo del tiempo.
3. **Duplicados y falsos positivos.** Añadir con baja probabilidad una pista duplicada a 0,5-2 m de un jugador real, o un árbitro o recogepelotas mal clasificado.
4. **Ruido de posición heteroscedástico.** El error crece con la distancia a la cámara y en los bordes del encuadre; el rango de RMSE de Crang muestra la cola larga. Se modela como deriva lenta (proceso AR(1)/OU) más saltos esporádicos.
5. **Balón.** Mantener los huecos con la distribución del objetivo y añadir errores de flag dead-ball (balón que "se mueve" en parado), que ya simulamos.

### Gaps
- No obtuve las tasas de identity switch por minuto de proveedores automáticos comerciales. No hay fuente abierta con esas cifras.
- No encontré el documento de "tracking data quality" de Bassek et al. con métricas de error de TRACAB. El paper de Scientific Data 2025 describe el dataset (7 partidos, 2022/23, TRACAB Gen5, CC-BY 4.0) pero no da cifras de error — [Bassek et al. 2025, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11787359/).

## 4. ¿Ayuda la estandarización por partido o la transformada de rangos en inferencia?

### Takeaway
No encontré evidencia publicada en tracking deportivo. La evidencia indirecta es RevIN, que mejora frente a distribution shift en series temporales, aunque una revisión de 2026 cuestiona parte de sus componentes. Es barato de probar con nuestro LOMO y está explícitamente permitido.

### Cited Findings
- RevIN normaliza por secuencia y por feature, y mejora diversos modelos de forecasting bajo shift — [ICLR 2022](https://iclr.cc/virtual/2022/poster/6034); [repo](https://github.com/ts-kim/RevIN).
- Ablación de 2026: algunos componentes de RevIN son redundantes o perjudiciales — [arXiv 2603.11869](https://arxiv.org/abs/2603.11869).
- PathCRF atribuye la caída a "distinct noise patterns", y AdaSports-Traj muestra escalas de error muy diferentes entre dominios — [PathCRF](https://arxiv.org/html/2602.12080); [AdaSports-Traj](https://arxiv.org/html/2509.16095v1). Ambos papers apuntan a un desajuste de escala y ruido, que es lo que la normalización por partido ataca.

### Inferences
- La estandarización por partido debe aplicarse solo a las magnitudes cuya escala depende del proveedor: velocidad, aceleración y jerk tras el suavizado, que dependen del filtro de cada tracker. Las magnitudes geométricas (distancias a balón y portería, en metros) deben quedar sin tocar, porque la escala del campo es la misma.
- Una transformada de rangos o percentiles por partido de la velocidad es más robusta a outliers (saltos del tracker) que el z-score.
- Riesgo: el partido objetivo tiene el 41 % de frames sin balón y el balón en parado se mueve. Las estadísticas por partido deberían calcularse sobre frames de jugadores (siempre presentes) y excluir los frames dead-ball detectados.
- Hay que aplicar la misma normalización en el entrenamiento, por partido y después de degradar. Si no, se introduce un desajuste train/test.

### Gaps
- No hay estudios sobre rank-transform por partido en modelos de posesión.

## 5. Qué es viable y validable en menos de 2-3 días con LOMO (IDSSE held-out y Metrica held-out)

### Takeaway
Hay cinco candidatos, ordenados por relación entre ganancia esperada y coste:
- (a) Aleatorizar los parámetros de degradación.
- (b) Oclusión por campo de visión con fragmentación de tracklets.
- (c) Normalización por partido de las features cinemáticas.
- (d) Ensemble de modelos entrenados con mezclas de degradación distintas.
- (e) Dropout explícito del canal de balón con flag.

El criterio de aceptación es la mejora en Metrica held-out (proxy cross-provider) sin pérdida en IDSSE held-out.

### Cited Findings
- La evidencia base de cada técnica está en las secciones 1-4: PathCRF (caída cross-provider), AdaSports-Traj (multi-source), RevIN y su crítica (normalización), Broadcast analytics y Crang (visibilidad y ruido), SoccerNet-GSR (fragmentación), y Kim 2023 y PathCRF (posesión sin balón) — [PathCRF](https://arxiv.org/html/2602.12080); [AdaSports-Traj](https://arxiv.org/html/2509.16095v1); [RevIN](https://iclr.cc/virtual/2022/poster/6034); [Sponet](https://www.sponet.de/sponet/Record/4096118?lng=en); [GSR](https://arxiv.org/html/2404.11335v1); [Kim 2023](https://arxiv.org/abs/2306.08206).

### Inferences (tabla de decisión; las ganancias son estimaciones razonadas, no medidas)
| Técnica | Coste | Ganancia esperada (spellF1, Metrica held-out) | Razonamiento | Riesgo |
|---|---|---|---|---|
| (a) DR de parámetros de degradación (deriva, dropout 0-3 jugadores, huecos escalados ×0,5-2) | 0,5-1 día (cambia el muestreador; reentrenar) | +0,01-0,03 | Amplía el soporte del entrenamiento; la evidencia DR genérica (NASA 64→79 %) es de otro dominio | Si el rango es demasiado ancho, puede bajar el rendimiento en limpio. Mitigación: mezclar un 20-30 % de muestras poco degradadas |
| (b) Oclusión por FOV centrada en el balón + fragmentación y permutación de tracklets + duplicados raros | 1 día | +0,01-0,02 en un proxy degradado; en Metrica limpio, ~0 | El tipo de ausencia real es espacial, no aleatorio (Sponet, Crang) | El FOV real del objetivo es desconocido. Muestrear el ancho para igualar los ~1,5 ausentes medios observados |
| (c) Estandarización o percentiles por partido de velocidad y aceleración (train e inferencia) | 0,5 día | 0 a +0,02 | Ataca el desajuste de escala del filtro de cada tracker (RevIN). La crítica de 2026 obliga a ablacionar | Las estadísticas se contaminan con frames dead-ball o con saltos. Usar estadísticos robustos (mediana/IQR) |
| (d) Ensemble de 3-5 modelos (semillas × mezclas de degradación) promediando probabilidades | 0,5-1 día (sobre todo tiempo de entrenamiento); la inferencia en CPU escala ×N | +0,005-0,015 | Reduce la varianza; complementa (a) | Coste de CPU y reproducibilidad. Fijar semillas y guardar los pesos |
| (e) Canal de balón con flag de ausencia y dropout de balón en tramos largos (ya parcialmente hecho con los huecos) | 0,5 día si falta el flag | Depende de lo ya hecho | Kim 2023 y PathCRF muestran que la posesión es inferible solo con jugadores | Ninguno relevante si ya existe |
| Embedding de dominio o contraste tipo AdaSports | 2+ días | Incierta con 2 fuentes | En inferencia, el dominio objetivo es desconocido | Coste alto y beneficio no medible con LOMO |
| Mixup o ruido adversarial | 1-2 días | Sin evidencia en deportes | — | Baja prioridad |

Protocolo sugerido:
- Cada técnica se prueba como una sola ablación sobre v6, en dos condiciones:
  - LOMO IDSSE (7 folds), con el held-out degradado según el perfil del objetivo.
  - Metrica held-out (3 folds), limpia y degradada.
- Se adopta si sube la media de Metrica degradado y no baja IDSSE degradado más de 0,005.
- Las combinaciones se prueban después: primero (a)+(c), y (d) al final porque es aditiva.

### Gaps
- Las ganancias de la tabla son estimaciones por analogía. No hay cifras publicadas para posesión con estas técnicas, así que solo el LOMO propio puede fijarlas.
- La regla sobre usar la distribución de huecos del objetivo (son estadísticas de inputs, no etiquetas) ya está en uso. Conviene dejar constancia de cómo se justifica frente a la aclaración de los organizadores.
