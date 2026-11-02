# Segmentación temporal y etiquetado secuencial para segmentos CORTOS (aplicado a posesión home/away/dead, spellF1)

Nota de alcance: casi todas las cifras medidas proceden de benchmarks de vídeo (Breakfast, 50Salads, GTEA, Assembly101). La única fuente medida en datos de tracking de fútbol es PathCRF. Ninguna fuente encontrada publica recall por bucket de duración de segmento; las mejoras de F1@k/Edit son mayoritariamente por **reducción de sobre-segmentación**, no por recuperar segmentos cortos. Los PDFs de ASRF y MS-TCN no se pudieron extraer (403 / binario), así que sus ablaciones aparecen solo con lo citado en fuentes secundarias.

## 1. Modelos semi-Markov / duración explícita / semi-CRF / priors de duración en decodificación: ¿ayudan a los cortos o los suprimen?

### Takeaway
Las variantes medidas usan la duración para **suprimir fragmentos** (duración mínima dura, techo de nº de segmentos). Suben F1@10/25 y Edit en vídeo (+1 a +5 puntos), pero no hay medición en segmentos cortos. Para nuestro problema la palanca útil es lo contrario de un mínimo duro: cambiar el coste constante de cambio por un log-prior de duración empírico (HSMM), que se puede ajustar para que los segmentos cortos frecuentes salgan baratos.

### Cited Findings
- **Constraint-Aware Decoding (CAD)**, Yeo Keat Ee, Debaditya Roy, Chen Li, Hao Zhang, Basura Fernando, 2026 (ICPR 2026). Viterbi modificado con priors estadísticos sacados de las anotaciones (confianza de transición, conjuntos de fronteras, duración min/max por clase normalizada por la longitud del vídeo). Se aplica sin reentrenar. Código: https://github.com/LUNAProject22/CAD (la página no indica licencia del código; el paper es CC BY 4.0) — [arXiv 2605.10149](https://arxiv.org/abs/2605.10149)
  - La duración actúa como **restricción dura**: solo se permite salir de una clase cuando el segmento alcanza la duración mínima, y quedarse solo mientras no supere la máxima. Una variante blanda (penalización tipo softmax) "consistently underperforms": Breakfast/ASFormer Edit 74.6 frente a 77.9 en la dura — [arXiv 2605.10149 HTML](https://arxiv.org/html/2605.10149)
  - ASFormer Breakfast: F1@10/25/50 74.1/68.7/55.5 → 78.8/73.1/58.0, Edit 72.8 → 77.9. 50Salads: 83.4/80.8/74.6 → 84.8/83.1/76.0, pero Acc 85.0 → 82.9. FACT Breakfast: F1@50 66.2 → 64.8 (−1.4). Semi-supervisado ICC 5 %: 50Salads F1@10 51.8 → 55.8 — [arXiv 2605.10149 HTML](https://arxiv.org/html/2605.10149)
  - Ablación (solo Breakfast/ASFormer): sin término de duración, F1@10 75.0 (frente a 78.8 completo) y F1@50 54.9 (frente a 58.0). Reponderar transición frente a duración (0.3/0.7 vs 0.7/0.3) cambia poco. No hay análisis de segmentos cortos ni varianza entre semillas — [arXiv 2605.10149 HTML](https://arxiv.org/html/2605.10149)
- **FIFA** (Fast Inference For Action segmentation, 2021). Sustituye la programación dinámica exacta por una energía diferenciable aproximada que se minimiza por descenso de gradiente. Inferencia más de 5x más rápida con rendimiento mantenido; "anytime". El abstract no detalla la forma del modelo de duración ni enlaza código — [arXiv 2108.03894](https://arxiv.org/abs/2108.03894)
- **Lea et al., Segmental Spatiotemporal CNNs** (ECCV 2016, v4). Modelo semi-Markov de transiciones entre acciones con inferencia segmental restringida "órdenes de magnitud" más rápida. El abstract no da cifras ni código oficial — [arXiv 1602.02995](https://arxiv.org/abs/1602.02995)
- Coste del Viterbi segmental (HSMM) con duración acotada: O(T·D·C²), con D la duración máxima — [Murphy, Hidden semi-Markov models](https://www.cs.ubc.ca/~murphyk/papers/segment.pdf)
- Ejemplo de prior de duración paramétrico en un semi-Markov de acciones: Poisson con media por etiqueta — [Fried et al., Learning to Segment Actions from Observation and Narration](https://arxiv.org/html/2005.03684)
- La motivación declarada de los modelos segmentales en TAS es eliminar "short spurious fragments", es decir, suprimir cortos — [resumen de búsqueda sobre semi-CRF / BiasFormer](https://www.mdpi.com/1424-8220/26/15/4854)

### Inferences
- Coste constante = prior de duración geométrico. Con 20 nats por cambio, un segmento intercalado paga 40 nats (dos cambios). Si el Viterbi corre a 25 fps sobre probabilidades interpoladas de 5 Hz, un segmento de 1 s (25 frames) necesita un log-ratio medio de unos 1.6 nats/frame sostenido, y uno de 0.5 s unos 3.2 nats/frame (ratio de probabilidad de ~25:1 en cada frame). Además, el upsampling copia la evidencia unas 5 veces, así que la penalización efectiva depende de la resolución a la que se decodifique. Esto encaja con el recall del 21 % en <2 s sin que el clasificador tenga que estar equivocado.
- Receta A (HSMM con duración empírica): decodificar a 5 Hz con un Viterbi semi-Markov donde coste(segmento) = −log P_dur(clase, d) − Σ log p. P_dur es el histograma suavizado de duraciones de train por clase (home, away, dead), con D_max de unos 60 s a 5 Hz (300 pasos, factible con O(T·D·C²)). Si la distribución real tiene masa en 0.5-2 s, esos segmentos quedan más baratos que con el coste geométrico de 20 nats. Hay que barrer un escalado λ·log P_dur.
- Receta B (CAD invertido): usar de CAD solo las restricciones de legalidad o máximo y no el mínimo duro, porque el mínimo duro es justo el mecanismo que elimina cortos.
- El coste de cambio podría depender del par de clases y de la duración del segmento previo (p. ej., away→dead→away corto barato), en lugar de 20/20 fijo.

### Gaps
- No encontré ninguna medición que separe el efecto de la duración explícita sobre segmentos cortos frente a largos. La RQ8 de Fried et al. ("semi-Markov vs Markov") aparece en el snippet, pero no se leyó la respuesta.
- No se recuperó el texto de Sarawagi & Cohen (semi-CRF, NeurIPS 2004) en esta sesión. Su relevancia (features a nivel de segmento) es conocida, pero sin cifras citables aquí.

## 2. Detección de fronteras, pérdidas de suavizado y de segmento, arquitecturas (ASRF, MS-TCN, DiffAct, ASFormer, LTContext, C2F-TCN)

### Takeaway
Las cabezas de frontera (ASRF, Mitsuoka & Hotta 2026) y las pérdidas de suavizado suben F1@k entre 1 y 5 puntos, sobre todo porque **fusionan fragmentos**. La parte reutilizable para cortos es la cabeza de frontera como señal independiente que localice los cambios, no el suavizado.

### Cited Findings
- **ASRF**, Yuchi Ishikawa, Seito Kasai, Yoshimitsu Aoki, Hirokatsu Kataoka, WACV 2021. Extractor de largo plazo con dos ramas, una de segmentación por frame y otra de regresión de fronteras. Las fronteras predichas refinan las predicciones por frame, y se añade una pérdida de suavizado de transiciones (GS-TMSE). Hasta +13.7 % de Edit y +16.1 % de F1 frente al SOTA de entonces — [arXiv 2007.06866](https://arxiv.org/abs/2007.06866). Código MIT: https://github.com/yiskw713/asrf — [GitHub](https://github.com/yiskw713/asrf)
  - 50Salads: F1@10/25/50 84.9/83.5/77.3, Edit 79.3, Acc 84.5 — [búsqueda; tabla replicada en Ahn & Lee ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Ahn_Refining_Action_Segmentation_With_Hierarchical_Video_Representations_ICCV_2021_paper.pdf)
- **Mitsuoka & Hotta, "Combining Boundary Supervision and Segment-Level Regularization"**, 2026. Añade (i) un canal de frontera agnóstico a clase con BCE, aplicado solo a ±5 frames de cada transición GT, y (ii) una pérdida de forma de segmento que compara la CDF de las probabilidades dentro de cada segmento GT con la CDF uniforme, en los frames no-frontera. Sin refinado en inferencia y sin código enlazado — [arXiv 2604.01859](https://arxiv.org/html/2604.01859)
  - MS-TCN 50Salads: 73.96/71.42/62.12 → 77.44/75.22/67.31, Edit 66.29 → 70.29. Breakfast: F1@50 36.58 → 40.78. GTEA: Edit 79.92 → 83.23. Las ganancias son menores sobre FACT, y con C2F-TCN hay caídas marginales de Acc — [arXiv 2604.01859](https://arxiv.org/html/2604.01859)
  - Aplicar ambas pérdidas a todos los frames empeora (GTEA Edit 81.61 frente a 83.23 con la asignación desacoplada) — [arXiv 2604.01859](https://arxiv.org/html/2604.01859)
- **MS-TCN** (Farha & Gall, CVPR 2019). La truncated MSE sobre log-probabilidades consecutivas produce "less over-segmentation errors" que la CE sola, con unos +5 puntos absolutos en Edit y F1 — [arXiv 1903.01945](https://arxiv.org/pdf/1903.01945)
- **DiffAct** (Diffusion Action Segmentation, Liu et al., ICCV 2023). Las predicciones se generan desde ruido condicionadas a las features, con máscaras de condicionamiento para el prior de posición, la ambigüedad de fronteras y la dependencia relacional. Código MIT: https://github.com/Finspire13/DiffAct. No pude extraer las cifras — [arXiv 2303.17959](https://arxiv.org/abs/2303.17959); [GitHub](https://github.com/Finspire13/DiffAct)
- **LTContext** (Bahrami et al., ICCV 2023). Breakfast 77.6/72.6/60.1, Edit 77.0. 50Salads 89.4/87.7/82.0, Edit 83.2. Assembly101 33.9/30.0/22.6. No analiza por longitud de segmento. Código: https://github.com/LTContext/LTContext (licencia no indicada en la página) — [arXiv 2308.11358](https://arxiv.org/html/2308.11358)
- Un trabajo de 2026 sobre active learning describe que en las regiones de transición las predicciones se vuelven inestables, con "boundary jitter and over-segmentation" — [arXiv 2604.15173](https://arxiv.org/html/2604.15173v1)
- En segmentación no supervisada se documenta el sesgo opuesto: ASOT genera muy pocos segmentos cortos aunque el GT tiene muchos, lo que baja F1 y recall pese a impactar poco en MoF — [arXiv 2412.17640](https://arxiv.org/pdf/2412.17640)

### Inferences
- Nuestro fallo es sub-segmentación (faltan cortos), no sobre-segmentación. Las pérdidas de suavizado (T-MSE, GS-TMSE) y la parte de forma de segmento empujan en la dirección que ya nos sobra.
- Receta C (cabeza de frontera): añadir al TCN un canal de "cambio de posesión" entrenado con BCE solo en ±k pasos de cada transición GT, como en Mitsuoka & Hotta. Nuestro logit de change-point ya probado restaba coste globalmente. La diferencia propuesta es (a) supervisarlo solo cerca de fronteras y (b) usarlo como propuesta de corte en un decodificador segmental: candidatos de frontera = picos de la cabeza, y luego elegir la etiqueta de cada intervalo entre picos (el refinado de ASRF), en lugar de restarlo al coste del Viterbi.
- Las arquitecturas (ASFormer, LTContext, DiffAct) suben F1 global en vídeo, pero no hay evidencia de que recuperen cortos. Su prioridad es menor que la del decodificador.

### Gaps
- No hay en las fuentes leídas un desglose de F1 o recall por longitud de segmento para ASRF, DiffAct, ASFormer, LTContext ni C2F-TCN.
- Las cifras de las ablaciones de ASRF (umbral de frontera, peso de GS-TMSE) y de MS-TCN (λ, τ) no se pudieron extraer de los PDFs.
- BCN (Wang et al., ECCV 2020) no se cubrió por límite de búsquedas.

## 3. Optimización directa de F1 de segmentos / Edit; decodificación MBR

### Takeaway
No encontré ninguna pérdida diferenciable de F1 segmental o de Edit publicada con resultados en TAS. Hay una "F1-adaptive loss" en VISAPP 2026 cuyo texto no pude leer. La vía práctica es decodificar por riesgo mínimo de Bayes (MBR) con muestreo de la posterior, que es especulativa porque no hay medición publicada en TAS.

### Cited Findings
- La búsqueda dirigida no encontró una aproximación diferenciable de F1 segmental o de edit distance para TAS. Lo más cercano es la T-MSE de MS-TCN — [MS-TCN](https://arxiv.org/pdf/1903.01945)
- "Hazard-Aware Duration Prior and F1-Adaptive Loss" (SciTePress/VISAPP 2026). El snippet indica que actúa a nivel de pérdida y que la mayor ganancia es en Edit. No pude extraer el texto (PDF binario) — [SciTePress](https://www.scitepress.org/Papers/2026/143962/143962.pdf)
- PathCRF entrena el CRF por NLL de secuencia (partition function con forward), lo que optimiza la verosimilitud de la secuencia completa en lugar de la CE por frame — [arXiv 2602.12080](https://arxiv.org/html/2602.12080)

### Inferences
- Receta D (MBR para spellF1, especulativa): con el modelo de transición actual, muestrear N trayectorias de la posterior (forward-filtering backward-sampling a 5 Hz), o usar un N-best del Viterbi con varios costes (5, 10, 20, 40 nats). Después, elegir la candidata que maximiza el spellF1 esperado frente a las demás muestras como pseudo-GT. Es barato de probar porque no requiere reentrenar.
- Receta E (barrido por umbral): spellF1 promedia IoU de 0.3 a 0.7. Un segmento GT de 1 s con frontera desplazada 0.2 s da IoU 0.8/1.2 = 0.67, que cuenta en 0.3-0.6 pero no en 0.7. Medir el spellF1 por umbral y por bucket de duración separa "no detectado" de "detectado pero mal alineado".
- Que el reescalado sin entrenar no funcionara no descarta MBR: este cambia el criterio de decisión, no las puntuaciones.

### Gaps
- No hay ninguna medición publicada de MBR para F1@IoU en segmentación de acciones encontrada en esta sesión.

## 4. Resolución temporal (5 vs 25 fps) y modelos multi-tasa

### Takeaway
La evidencia medida va en la dirección esperada: bajar la resolución mejora Edit y F1 por menos sobre-segmentación, pero los propios autores avisan de que se pierden fronteras precisas y "small action segments". FACT y BIT evitan un submuestreo fijo por ese motivo.

### Cited Findings
- MS-TCN a 1 fps frente a 15 fps en 50Salads: 1 fps mejora Edit y F1, y 15 fps da mejor accuracy. Cita: "this comes with the cost of losing the precise location of the boundaries … or even missing small action segments". La diferencia en F1@50 es pequeña (64.0 vs 64.5) — [MS-TCN arXiv 1903.01945](https://arxiv.org/pdf/1903.01945)
- BIT y FACT señalan que no submuestrean con ratio fijo porque eso elimina las features de segmentos cortos. En la ablación de FACT en EgoProceL, el down/upsampling temporal sube F1@50 un 1.6 % — [BIT arXiv 2308.14900](https://arxiv.org/pdf/2308.14900); [FACT supp.](https://www.khoury.northeastern.edu/home/eelhami/publications/cvpr24_suppmat_FACT_sup_action_seg.pdf)
- LTContext: submuestrear la entrada degrada el rendimiento frente a no submuestrear (apéndice, sin cifras en el texto leído) — [arXiv 2308.11358](https://arxiv.org/html/2308.11358)
- PathCRF trabaja a 5 FPS (submuestreado de 25) y sobremuestrea a 25 para el análisis, la misma configuración que la nuestra — [arXiv 2602.12080](https://arxiv.org/html/2602.12080)
- EAST hace segmentación por detección en frames muestreados de forma gruesa y mapea las propuestas a la tasa completa — [arXiv 2503.06316](https://arxiv.org/pdf/2503.06316)

### Inferences
- A 5 Hz un segmento de 1 s tiene 5 pasos y uno de 0.4 s tiene 2. Un TCN dilatado con campo receptivo grande tiende a promediarlos. Receta F: una rama fina a 25 fps (o 12.5) solo para la cabeza de frontera de la receta C, y la clase a 5 Hz. Es multi-tasa, en la línea del razonamiento de BIT y FACT, pero sin medición en tracking.
- Receta G: decodificar a 5 Hz con costes calibrados a esa tasa, o a 25 fps con coste/5, y comparar. Ahora mismo la evidencia interpolada se cuenta unas 5 veces.

### Gaps
- No hay curvas de recall por duración frente a fps en ninguna fuente.

## 5. Desbalance de segmentos cortos raros (focal loss, muestreo de crops)

### Takeaway
No encontré estudios de TAS que midan focal loss o sobremuestreo de crops con segmentos cortos y reporten su efecto. La ponderación 1/longitud ya se probó sin éxito. El cuello de botella probable está en el decodificador (sección 1), no en la pérdida.

### Cited Findings
- No hay ninguna fuente con mediciones encontrada en esta sesión. Lo más próximo es la observación de que los modelos sesgados a segmentos largos pierden F1 y recall en los cortos aunque el MoF apenas cambie — [arXiv 2412.17640](https://arxiv.org/pdf/2412.17640)

### Inferences
- Antes de tocar la pérdida, conviene comprobar si los cortos tienen señal en las probabilidades crudas: medir, por segmento GT corto, el log-ratio medio de la clase GT frente a las vecinas antes del Viterbi. Si supera el umbral de la sección 1 (≈1.6 nats/frame a 25 fps para 1 s) y aun así se pierden, el problema es el decodificador. Si no lo supera, focal loss o crops centrados en transiciones tienen sentido.

### Gaps
- No hay medición publicada de focal loss ni de muestreo de crops en TAS para segmentos cortos.

## 6. Específico de deporte: posesión / estado del balón

### Takeaway
PathCRF (fútbol, tracking a 5 FPS, sin balón) mide el valor de entrenar la estructura: el CRF con máscara dinámica entrenado por NLL supera en +2.55 de F1 de eventos al Viterbi aplicado sobre un modelo no-CRF, y en +10.35 al no-CRF sin decodificar.

### Cited Findings
- **PathCRF: Ball-Free Soccer Event Detection**, Hyunsung Kim, Kunhee Lee, Sangwoo Seo et al., 2026. Grafo dinámico de jugadores, una arista de posesión por paso (self-loop para conducción, emisor→receptor para pase). El CRF calcula emisiones y transiciones desde los embeddings de arista y enmascara con −∞ las transiciones imposibles. Decodificación Viterbi. Código: https://github.com/hyunsungkim-ds/pathcrf.git (licencia del código no indicada; paper CC BY 4.0). Dataset: Sportec Open DFL — [arXiv 2602.12080](https://arxiv.org/html/2602.12080)
  - F1 de eventos: no-CRF 65.34 %, no-CRF + Viterbi 73.14 %, Dynamic Masked CRF 75.69 %. Accuracy de arista correcta 69.64 % entre 676 candidatas. Un evento cuenta como acierto con mismo tipo, mismo jugador y tolerancia de 1 s — [arXiv 2602.12080](https://arxiv.org/html/2602.12080)
  - Pérdida: NLL del CRF + CE intermedia + CE de emisión (pesos λ1, λ2) — [arXiv 2602.12080](https://arxiv.org/html/2602.12080)

### Inferences
- Receta H (CRF entrenado de extremo a extremo): sustituir el coste fijo de 20 nats por una matriz de transición 3×3 aprendida, más un término de transición dependiente del input (features en t y t+1, como las transiciones calculadas desde embeddings en PathCRF), entrenada con NLL de secuencia sobre los logits del TCN a 5 Hz. Así el modelo aprende cuándo un cambio rápido es plausible (p. ej., pérdida y recuperación inmediata), en vez de pagar siempre lo mismo. Es la versión entrenada de nuestro intento "−cost + gain·logit", que se ajustó fuera del grafo de entrenamiento.
- La métrica de PathCRF es por evento con tolerancia de 1 s, no F1 por IoU de intervalos, así que el +2.55 no se traslada directamente a spellF1.

### Gaps
- PathCRF no publica un análisis por duración de posesión ni de pases rápidos. La parte final del HTML (unos 30k caracteres) no se leyó.
- No encontré otros trabajos de segmentación de estado del balón (vivo/muerto) con métricas por segmento.
