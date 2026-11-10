# Modelos y decodificación para segmentación de posesión desde tracking (2022-2026), y su encaje en v6

## 1. PathCRF (Kim et al., arXiv 2602.12080, KDD 2026): arquitectura, ablaciones, motor de la ganancia, requisitos, licencia y cómo acoplar un CRF de transición dinámica a nuestro encoder

### Takeaway
PathCRF elige en cada paso de 5 Hz una arista entre 26 nodos (22 jugadores + 4 nodos "fuera"), es decir quién lleva el balón o hacia quién va. Usa set attention dentro y entre equipos, un módulo temporal de atención y un CRF entrenado de extremo a extremo con transiciones enmascaradas y dinámicas. La mayor parte de su ganancia en F1 de eventos viene de la máscara de transiciones imposibles y de la supervisión auxiliar a nivel de jugador (emisor/receptor). La transición dinámica añade unos +1.9 F1 sobre una estática enmascarada. Para nuestro caso H/A/D, lo transferible es sobre todo la cabeza auxiliar por jugador (evidencia de toque); la máscara aporta poco con 3 estados.

### Cited Findings
- Autores: Hyunsung Kim, Kunhee Lee, Sangwoo Seo, Sang-Ki Ko, Jinsung Yoon, Chanyoung Park. Enviado el 12-feb-2026, v2 el 1-jun-2026, KDD 2026 — [arXiv abs](https://arxiv.org/abs/2602.12080)
- Arquitectura del backbone:
  - Dos encoders apilados (coarse→fine), inspirados en ROLAND. Cada uno tiene un módulo social y otro temporal.
  - Módulo social:
    - Red PPE (parcialmente permutation-equivariant): un ISAB por grupo (local, visitante, nodos exteriores).
    - Red FPE: un ISAB sobre todos los nodos.
    - Las salidas de PPE y FPE se concatenan.
  - Módulo temporal: positional encoding + SABs apilados, al estilo TranSPORTmer.
  - Cabezas del CRF:
    - La arista (u,v) se embebe con un MLP sobre la concatenación de los embeddings de u y v.
    - Emisión: f_t(e) = MLP(z_e,t).
    - Transición dinámica: ψ_t(e′,e) = MLP_trans([z_e′,t−1 ; z_e,t]).
    - Máscara dura: continuar, kick, recibir/one-touch y fuera de juego están permitidos; el resto vale −10⁴.
  - El texto visible no da el número de capas ni las dimensiones — [arXiv HTML v2](https://arxiv.org/html/2602.12080v2)
- Entradas y datos:
  - El balón no es entrada del modelo. Solo se usa para construir las etiquetas.
  - Jugadores no ordenados dentro del equipo. No se describe flag de portero ni identidades.
  - 25 FPS submuestreado a 5 FPS, ventanas de 10 s (50 pasos).
  - Dataset Sportec Open DFL (Bassek et al. 2025; 7 partidos de Bundesliga 1 y 2), split 5/1/1 partidos.
  - Etiquetas: eventos sincronizados con ELASTIC, toques añadidos con RDP, alineación con Needleman–Wunsch — [arXiv HTML v2](https://arxiv.org/html/2602.12080v2)
- Tabla 2 (partido de test):

  | Variante | Edge acc | Event P | Event R | Event F1 |
  |---|---|---|---|---|
  | Non-CRF | 69.24 | 58.77 | 73.57 | 65.34 |
  | Non-CRF + VCD (decodificación con restricción, sin entrenar CRF) | 69.68 | 69.50 | 77.17 | 73.14 |
  | Static Dense CRF | 69.41 | — | — | 68.78 |
  | Static Masked CRF | 69.54 | — | — | 73.83 |
  | Dynamic Dense CRF | 69.08 | — | — | 67.82 |
  | **Dynamic Masked CRF** | **69.64** | **73.18** | 78.37 | **75.69** |

  La precisión por frame apenas cambia (~69%). La ganancia está en la coherencia de los segmentos, y el principal salto viene de la máscara (dense→masked: +5 a +8 F1) — [arXiv HTML v2](https://arxiv.org/html/2602.12080v2)
- Posesión por equipo: accuracy por frame del 92.40%. Cuota de posesión local de 62.13% real frente a 62.64% predicha — [arXiv HTML v2](https://arxiv.org/html/2602.12080v2)
- Ablación del backbone (Tabla 3, Dynamic MCRF):

  | Backbone | Edge acc | Event F1 |
  |---|---|---|
  | TranSPORTmer (SAB + PE-SAB) | 62.97 | 68.19 |
  | Ball Radar (PPE-FPE + Bi-LSTM) | 70.39 | 75.24 |
  | Principal (PPE-FPE + PE-SAB) | 69.64 | 75.69 |

  Separar dentro de cada equipo (PPE) es lo que pesa. Cambiar el módulo temporal (Bi-LSTM o atención) da un efecto marginal — [arXiv HTML v2](https://arxiv.org/html/2602.12080v2)
- Ablación de pérdidas (Apéndice C, Dynamic MCRF, edge acc / F1):

  | Pérdidas | Edge acc | Event F1 |
  |---|---|---|
  | Todas | 69.64 | 75.69 |
  | Sin L_coarse | 67.52 | 72.77 |
  | Sin L_emit | 65.77 | 70.74 |
  | Sin ninguna auxiliar | 65.64 | 69.65 |

  L = L_CRF + λ1·L_coarse (CE sobre emisor/receptor por nodo) + λ2·L_emit. No se publican los valores de λ — [arXiv HTML v2, offset 100k](https://arxiv.org/html/2602.12080v2)
- Generalización (Apéndice B, F1 de eventos):

  | Entrenamiento → Test | Event F1 |
  |---|---|
  | Sportec LOMOCV | 73.88 ± 1.90 |
  | K League → K League (15 partidos, GPS 10 Hz) | 79.83 ± 2.60 |
  | K League → Sportec | 66.25 ± 7.65 |
  | Sportec → K League | 49.16 ± 3.37 |

  Cambiar de proveedor cuesta entre 8 y 30 puntos — [arXiv HTML v2, offset 100k](https://arxiv.org/html/2602.12080v2)
- Coste (RTX 4090):
  - Entrenamiento por época: 184 s sin CRF, 242 s con MCRF estático, 565 s con MCRF dinámico.
  - Inferencia: 20-35 s por partido a 5 FPS — [arXiv HTML v2](https://arxiv.org/html/2602.12080v2)
- Repositorio [hyunsungkim-ds/pathcrf](https://github.com/hyunsungkim-ds/pathcrf):
  - Licencia MPL-2.0. Usa kloppy y trae los eventos ya sincronizados en `data/sportec_synced`.
  - Incluye checkpoints en `saved/` (100 Non-CRF … 140 Dynamic Masked CRF), scripts en `scripts/*.sh` y `tutorial.ipynb`.

### Inferences
- **Qué impulsa la ganancia en PathCRF:** la máscara de transiciones imposibles y las pérdidas auxiliares por jugador. Con 3 estados H/A/D casi no hay transiciones imposibles que enmascarar, así que su ganancia principal no se traslada a nuestro decodificador.
- **Relación con lo ya probado:** "Non-CRF + VCD" (decodificación con restricciones sobre salidas sin CRF, 73.14) queda a 2.5 F1 del CRF dinámico entrenado. Nuestro Viterbi post-hoc ya está en esa zona. También probamos transiciones aprendidas y logit de cambio sin ganancia, lo que es coherente con que la parte dinámica valga ~+1.9 F1 y solo si se entrena de extremo a extremo con log-verosimilitud del CRF.
- **Acoplar una transición dinámica a nuestro encoder:**
  - Añadir sobre la salida del TCN una cabeza ψ_t(i→j) = MLP([h_t−1; h_t]) de 3×3 logits por paso.
  - Entrenar con la pérdida CRF (algoritmo forward, ~30 líneas en PyTorch), conservando la CE por frame como auxiliar, igual que L_emit en PathCRF.
  - Decodificar con Viterbi y ψ_t en lugar del coste constante 14.
  - Es determinista en CPU si se exporta a numpy.
  - Coste: ~1.5-2 días. Ganancia esperada: pequeña, de 0 a +0.01 spellF1, porque es casi lo ya probado con no-ganancia; solo cambia el entrenamiento de extremo a extremo.
  - Riesgo: medio. La parte entrenada de ψ puede volver a sesgar hacia pocos cambios.
- **Idea más prometedora: cabeza auxiliar por jugador ("quién toca").**
  - En el encoder DeepSets por equipo, añadir un logit por jugador (posesor actual) entrenado con CE sobre etiquetas de posesor derivadas de los datos externos (IDSSE/Metrica: eventos + balón).
  - Agregar ese logit (max/softmax por equipo) como entrada a la cabeza H/A/D.
  - Ataca justo nuestra debilidad: las posesiones < 2 s, donde el modelo da ~0.45 al equipo correcto por falta de evidencia de toque.
  - Evidencia indirecta: quitar L_coarse cuesta −2.9 F1 de eventos.
  - Coste: ~2 días. No necesita identidades (PathCRF trata a los jugadores como no ordenados).
  - Ganancia esperada: de +0.01 a +0.03 spellF1, concentrada en posesiones cortas.
  - Riesgo: la calidad de la etiqueta de posesor con balón borroso. Ya vimos que las features explícitas de toque con balón borroso no aportaron; aquí la diferencia es supervisar una tarea, no calcular un feature.
- **Encaje del backbone:**
  - Nuestro DeepSets por equipo (mean/max/atención por cercanía al balón) ya es un PPE simplificado.
  - Lo que falta es interacción entre equipos a nivel de jugador (FPE/ISAB) antes de agregar.
  - Un ISAB ligero (2 bloques, d=64) entre los 22 jugadores + balón cuesta ~1-1.5 días. La evidencia (SAB plano frente a PPE-FPE: −6.7 edge acc) favorece mantener la separación por equipo y añadir un FPE, no reemplazarla.

### Gaps
- No pude leer las dimensiones ni los valores de λ1/λ2 en el texto visible. Habría que mirar `scripts/*.sh` del repo.
- No se publica spellF1 ni F1 por segmento de posesión de equipo, solo accuracy por frame (92.4%) y F1 de eventos. No hay recall de posesiones cortas.
- El paper no dice "IDSSE"; usa el Sportec Open DFL (Bassek et al. 2025). Que sean los mismos partidos que nuestro IDSSE es probable pero no está verificado.

## 2. TranSPORTmer, Ball Radar, Borghesi et al. (ESWA 2023) y otros: precisión de posesión/estado reportada y transformers frente a TCN/GRU

### Takeaway
Ninguno de estos trabajos reporta F1 por spells. Todos dan accuracy por frame:
- Borghesi (TCN): 86.2% de posesión y 89.2% de vivo/muerto.
- TranSPORTmer: 87-90% en 4 estados con balón y 81-86% sin balón.
- Ball Radar: 64.7% de posesor exacto sin balón.

La comparación directa transformer frente a recurrente (PathCRF, Tabla 3) da diferencias marginales en el módulo temporal. El cambio de arquitectura temporal (TCN→transformer) no tiene evidencia de ganancia grande. La estructura social sí la tiene.

### Cited Findings
- **TranSPORTmer** (Capellera et al., ACCV 2024; Kognia + IRI CSIC-UPC):
  - Clasifica por frame 4 estados (pass, possession, uncontrolled, out-of-play) con un agente CLS extra.
  - Datos: LaLiga 2022-23, 283 partidos, 6.25 Hz, secuencias de 60 frames (9.6 s).
  - Arquitectura: dos encoders coarse-to-fine con 2 SAB temporales + 1 SAB social cada uno, d=128, 16 cabezas, 512 hidden.
  - Accuracy por frame: 87.35-90.47% con balón visible y 85.59% con el balón totalmente oculto (inferido).
  - La clase "uncontrolled" es la peor, por anotación subjetiva y desbalance.
  - No da comparación numérica con baselines para la clasificación de estados ni enlaza código.
  - El paper está bajo CC BY-NC-SA 4.0 — [arXiv 2410.17785v2](https://arxiv.org/html/2410.17785v2); [ACCV 2024](https://openaccess.thecvf.com/content/ACCV2024/html/Capellera_TranSPORTmer_A_Holistic_Approach_to_Trajectory_Understanding_in_Multi-Agent_Sports_ACCV_2024_paper.html)
- **Ball Radar** (Kim et al., KDD 2023):
  - Set Transformer + Bi-LSTM jerárquico. Un clasificador de posesor alimenta un regresor de trayectoria del balón.
  - Error medio del balón < 3.7 m y accuracy de posesor del 64.7%.
  - Datos: 15 partidos GPS de Fitogether + 3 de Metrica. Código en [pientist/ballradar](https://github.com/pientist/ballradar) — [arXiv 2306.08206](https://arxiv.org/abs/2306.08206)
  - PathCRF señala que estos enfoques solo con jugadores producen cambios de posesión irrealmente frecuentes — [arXiv 2602.12080](https://www.arxiv.org/pdf/2602.12080)
- **Borghesi, Costa, Morra, Lamberti** (ESWA vol. 223, 2023, DOI 10.1016/j.eswa.2023.119780):
  - TCN sobre posiciones de jugadores y balón, con embeddings por objeto y agregación permutation-invariant (suma, self-attention, conv 2D).
  - 86.2% de accuracy de posesión (+7.3 sobre el mejor método basado en reglas) y 89.2% vivo/muerto (+33.2).
  - Incluye ablaciones de las entradas — [búsqueda/abstract](https://smartdata.polito.it/publications/using-temporal-convolutional-networks-to-estimate-ball-possession-in-soccer-games); [IRIS accepted version](https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf) (403 al descargar)
- **Rudolph y Brefeld** (MLSA 2023): el preentrenamiento con masked autoencoder sobre trayectorias multiagente mejora la clasificación supervisada de eventos en tracking de élite. El abstract no trae cifras — [sponet record](https://www.sponet.de/sponet/Record/4088646?lng=en)
- **PathCRF Tabla 3:** Bi-LSTM frente a PE-SAB como módulo temporal da 75.24 frente a 75.69 F1 (marginal). SAB social plano frente a PPE-FPE da 68.19 frente a 75.69 — [arXiv HTML v2](https://arxiv.org/html/2602.12080v2)

### Inferences
- Nuestro TCN dilatado de ±50 s ya cubre un contexto mucho mayor que las ventanas de 10 s de PathCRF y TranSPORTmer. No hay evidencia de que un transformer temporal mejore a un TCN en esta tarea.
- Cambiar el TCN por un transformer: coste de ~2-3 días, ganancia esperada ≈0 y riesgo alto en 7 días. No recomendado.
- Las cifras por frame de la literatura (86-92%) no son comparables con nuestro spellF1. La accuracy por frame de equipo de PathCRF (92.4%) es una referencia útil si medimos la nuestra en IDSSE.
- El preentrenamiento auto-supervisado (masked autoencoder) sobre tracking externo sin etiquetas es una vía distinta del self-training ya probado. Coste: ~3 días. Hay evidencia cualitativa pero no cuantitativa: ganancia incierta y riesgo alto con el plazo.

### Gaps
- No pude descargar el PDF de Borghesi (403), así que no tengo las cifras por agregador ni la ablación de entradas.
- No encontré trabajos 2024-2026 que reporten F1 por segmento de posesión de equipo comparando transformer con TCN/GRU sobre tracking.

## 3. Decodificación: semi-Markov / duración explícita / restricciones, y qué ayuda contra la sub-segmentación

### Takeaway
La evidencia más reciente y cuantificada es Constraint-Aware Decoding (CAD, ICPR 2026):
- Un Viterbi con duración por clase [d_min, d_max], confianza de transición empírica y conjuntos inicio/fin.
- Sube F1@10 entre +1.0 y +4.7 puntos sin reentrenar, y el término de duración es el que más aporta (−3.8 F1@10 al quitarlo).
- Las restricciones duras van mejor que las blandas.

Contra la sub-segmentación, el mecanismo relevante es el límite superior de duración d_max, que obliga a cortar segmentos, o en general un coste de cambio dependiente de la duración. Un mínimo de duración, en cambio, actúa contra la sobre-segmentación (ya probado para muerto, sin ganancia). La literatura de TAS casi no estudia la sub-segmentación explícitamente.

### Cited Findings
- **CAD** (Ee, Roy, Li, Zhang, Fernando; arXiv 2605.10149, ICPR 2026):
  - Algoritmo:
    - El Viterbi mantiene las matrices V, D (duración del segmento actual) y B.
    - La autotransición solo se permite si la duración < d_max. Un cambio de clase solo se permite si el segmento previo ≥ d_min y la transición es válida.
    - El cambio suma log Conf(c′→c) = log Count(c′→c)/Count(c′).
    - Complejidad declarada O(CT).
  - Código en github.com/LUNAProject22/CAD; la licencia no aparece en la página — [arXiv abs](https://arxiv.org/abs/2605.10149); [HTML](https://arxiv.org/html/2605.10149)
- **CAD, resultados** (F1@10/25/50 y Edit):

  | Configuración | Sin restricciones | Con CAD | Edit |
  |---|---|---|---|
  | ASFormer, Breakfast | 74.1 / 68.7 / 55.5 | 78.8 / 73.1 / 58.0 | 72.8 → 77.9 |
  | ASFormer, 50Salads | 83.4 / 80.8 / 74.6 | 84.8 / 83.1 / 76.0 | — (Acc 85.0 → 82.9) |
  | Semisupervisado C2F-TCN + ICC, 50Salads | 51.8 / 47.7 / 37.0 | 55.8 / 51.4 / 38.5 | — |
  | FACT, Breakfast (backbone ya fuerte) | 81.3 / 76.4 / 66.2 | 81.8 / 76.5 / 64.8 | — |

  - En el caso semisupervisado de 50Salads, el Viterbi clásico obtiene 45.5 / 40.2 / 30.1, peor que no decodificar.
  - Con FACT, F1@50 baja — [arXiv HTML](https://arxiv.org/html/2605.10149)
- **CAD, ablación** (ASFormer, Breakfast, F1@10/25/50):

  | Variante | F1@10/25/50 |
  |---|---|
  | Completo | 78.8 / 73.1 / 58.0 |
  | Sin duración | 75.0 / 69.4 / 54.9 |
  | Sin transición | 75.1 / 70.0 / 56.3 |
  | Sin inicio-fin | 77.9 / 72.4 / 57.8 |
  | Restricciones blandas | 75.6 / 70.5 / 56.8 |

  - Ponderaciones transición/duración de 0.3/0.7 y 0.7/0.3: diferencias despreciables.
  - El paper no cuantifica la sobre- o sub-segmentación ni el número de segmentos — [arXiv HTML](https://arxiv.org/html/2605.10149)
- **Behrmann et al.** (arXiv 2209.00638, 2022): en seq2seq, la predicción explícita de duración localiza peor los segmentos que la implícita — [arXiv 2209.00638](https://arxiv.org/pdf/2209.00638)
- **Lea et al.** (ECCV 2016, arXiv 1602.02995): modelo semi-Markov sobre salidas de una CNN temporal, con un algoritmo de inferencia segmental restringida órdenes de magnitud más rápido. La v4 corrige un error matemático en la inferencia segmental — [arXiv 1602.02995](https://arxiv.org/abs/1602.02995)
- **ASRF** (Ishikawa et al., WACV 2021): una rama de regresión de fronteras refina las etiquetas por frame (+8.2% de F1 segmental). F1@50 de 77.3 en 50Salads, 56.1 en Breakfast y 79.8 en GTEA. Su objetivo declarado es la sobre-segmentación — [WACV 2021](https://openaccess.thecvf.com/content/WACV2021/papers/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.pdf); [paperswithcode](https://paperswithcode.com/paper/alleviating-over-segmentation-errors-by)
- **MSBATN** (2025, vídeo quirúrgico): las MS-TCN con campos receptivos grandes pueden causar tanto sobre- como sub-segmentación. Propone una pérdida unificada de clasificación + frontera, con mejoras en F1@25/50 — [arXiv 2504.18756](https://arxiv.org/pdf/2504.18756v2)

### Inferences
- **Diagnóstico de la sub-segmentación:** un coste de cambio constante (14) penaliza igual un cambio que abre un spell de 1 s que uno de 20 s. Para recuperar posesiones < 2 s hace falta que el coste dependa de la evidencia local, no solo de la duración.
- **Opción A, semi-Markov con prior de duración aprendido de externos.**
  - Coste de un segmento = −log p_dur(d | clase) + suma de log-probabilidades. p_dur es el histograma empírico de duraciones de spells en IDSSE/Metrica de entrenamiento: lognormal o histograma suavizado, con límite D_max ≈ 60-90 s a 5 Hz.
  - Es distinto de lo ya probado (mínimo de muerto y coste constante): da un prior explícito que favorece duraciones cortas frecuentes y desfavorece spells larguísimos (sub-segmentación).
  - Coste: ~1.5 días con numpy, O(T·D·3), unos 10⁷ operaciones por partido a 5 Hz con D=450, viable y determinista.
  - Ganancia esperada: de +0.005 a +0.02 spellF1, sobre todo por partir spells sobre-fusionados.
  - Riesgo: medio-bajo. Ajustar la temperatura entre emisiones y prior en LOMO externo.
- **Opción B, d_max duro al estilo CAD.** Variante mínima de A: medio día. Riesgo: cortes forzados en posesiones largas reales; hay que fijarlo con percentil alto (p99) de duraciones externas.
- **Opción C, cabeza de frontera (estilo ASRF/MSBATN).**
  - Una salida p(boundary_t) entrenada con etiquetas de cambio, que modula el coste de cambio: coste_t = 14·(1 − α·p_b,t).
  - Ya probamos el "change logit en el coste" sin ganancia; esta opción solo es distinta si la cabeza se entrena con una pérdida de frontera suavizada (gaussiana ±1 s) y con la evidencia por jugador de la sección 1.
  - Prioridad baja si no se hace antes la cabeza por jugador.
- **Fusión con el F1 por segmento:** que el Viterbi clásico pueda empeorar F1 cuando las emisiones son débiles (resultado semisupervisado de CAD) sugiere calibrar la mezcla HGB/encoder antes de ajustar los costes del decodificador. Ver sección 4.

### Gaps
- No encontré trabajos que midan explícitamente la sub-segmentación ni el número de segmentos predichos frente a los reales con un decodificador; CAD tampoco lo reporta.
- No verifiqué la licencia del código de CAD.
- No encontré cifras recientes (2022-2026) de HSMM o semi-CRF sobre tracking deportivo.
- No revisé DiffAct ni ASFormer en detalle: son modelos de vídeo con ganancias reportadas en benchmarks de cocina, sin evidencia en tracking.

## 4. Ensembling y calibración para segmentación (temperatura, HGB + red, stacking)

### Takeaway
No encontré evidencia específica 2022-2026 que cuantifique el efecto de la calibración por temperatura o del stacking sobre métricas F1 por segmento en tracking deportivo. Lo único directamente relacionado es indirecto: los decodificadores con costes fijos son sensibles a la escala de las log-probabilidades (CAD: blando < duro; el Viterbi clásico puede empeorar con emisiones débiles).

### Cited Findings
- En CAD, las restricciones blandas (penalización) rinden menos que las duras (75.6 frente a 78.8 F1@10 en Breakfast). Así, el equilibrio entre la escala de las emisiones y los costes del decodificador determina el resultado — [arXiv HTML 2605.10149](https://arxiv.org/html/2605.10149)
- En PathCRF, la pérdida auxiliar de emisión (CE por frame junto al CRF) aporta +5 F1 de eventos. Mantener emisiones bien calibradas por frame junto a la estructura de secuencia es importante — [arXiv HTML 2602.12080v2](https://arxiv.org/html/2602.12080v2)

### Inferences
- Escalado por temperatura separado para HGB y encoder antes de la mezcla 0.25/0.75, ajustado por log-loss en LOMO externo. Equivale a reparametrizar el peso de la mezcla y el coste de cambio.
  - Coste: 0.5 días. Ganancia esperada: pequeña (≤ +0.005), porque el barrido de costes ya absorbe la mayor parte.
  - Útil sobre todo para que un prior de duración (sección 3A) tenga una escala coherente.
- La mezcla en espacio logit (media geométrica) en lugar de lineal es un cambio barato de probar (horas). Sin evidencia publicada para este dominio.
- No re-proponer: más semillas, stacking HGB en dos etapas ni LightGBM (ya probados sin ganancia).

### Gaps
- No encontré estudios con cifras sobre ensembling o calibración evaluados con F1@IoU en segmentación temporal ni en tracking deportivo. Esta sección se apoya en inferencias.

## Resumen de prioridades (inferencias, para 7 días)

| # | Técnica | Evidencia | Coste | Ganancia esperada (spellF1) | Riesgo |
|---|---|---|---|---|---|
| 1 | Cabeza auxiliar por jugador (posesor) en el encoder | PathCRF: −2.9 a −6 F1 sin auxiliares | ~2 d | +0.01 a +0.03, en posesiones < 2 s | Medio (calidad de etiqueta de posesor) |
| 2 | Viterbi semi-Markov con prior de duración externo / d_max | CAD: duración = mayor aporte (+3.8 F1@10) | 0.5-1.5 d | +0.005 a +0.02 contra la sub-segmentación | Medio-bajo |
| 3 | ISAB entre equipos (FPE) además del DeepSets por equipo | PathCRF Tabla 3: PPE-FPE ≫ SAB plano | 1-1.5 d | Incierta, pequeña | Medio |
| 4 | CRF H/A/D entrenado de extremo a extremo con transición dinámica | +1.9 F1 dinámico frente a estático en PathCRF | 1.5-2 d | 0 a +0.01 | Medio (solapa con lo ya probado) |
| 5 | Temperatura por modelo / mezcla en logit | Indirecta | 0.5 d | ≤ +0.005 | Bajo |
| — | Transformer temporal en lugar de TCN | Bi-LSTM ≈ PE-SAB (diferencia marginal) | 2-3 d | ≈0 | Alto |
