# Inferencia / imputación del balón a partir de jugadores y su efecto en posesión/eventos

Notas de investigación (2026-11-02). "Medido" = número publicado por los autores en su propio benchmark. "Especulativo" = inferencia mía.

## 1. Ball Radar (Kim et al., KDD 2023): error, poseedor, huecos, código

### Takeaway
Con solo jugadores (0% balón) Ball Radar da ~3.66 m de error medio y 64.7% de acierto de poseedor (85.9% de equipo); si ve el 20% del balón real baja a 1.31 m y 93% de poseedor. El repo existe pero no tiene licencia declarada.

### Cited Findings
- Título: "Ball Trajectory Inference from Multi-Agent Sports Contexts Using Set Transformer and Hierarchical Bi-LSTM". Autores: Hyunsung Kim, Han-Jun Choi, Chang Jo Kim, Jinsung Yoon, Sang-Ki Ko. KDD 2023. Arquitectura: Set Transformer (invariante/equivariante a permutación) + Bi-LSTM jerárquico que primero predice el poseedor y después la posición del balón; "reality loss" y postproceso para que la trayectoria sea físicamente plausible — [arXiv 2306.08206](https://arxiv.org/abs/2306.08206)
- Datos: Fitogether GPS (10/2/3 partidos train/val/test) y Metrica (2 train, 0.5 val, 0.5 test). 10 Hz (Metrica bajado de 25 Hz). Ventanas de 10 s (100 pasos). Entrada: 22 jugadores × 6 features (x, y, vx, vy, rapidez, aceleración) — [arXiv HTML](https://arxiv.org/html/2306.08206)
- Medido, Tabla 2 (sin balón): H-LSTM-RL PE 3.6561 m, poseedor por jugador (PPA) 64.70%, poseedor por equipo (TPA) 85.85%. Con postproceso (H-LSTM-RL-PP) PE 4.07 m, PPA 64.32%. Baselines sin jerarquía: LSTM 5.47 m, Transformer 7.47 m, VRNN 12.37 m — [arXiv HTML](https://arxiv.org/html/2306.08206)
- Medido, ablación de entradas (Tabla 7): solo posición 4.65 m / PPA 53.85% / TPA 82.04%; posición+velocidad 3.92 m / 62.68% / 84.90%; todo 3.66 m / 64.70% / 85.85% — [arXiv HTML](https://arxiv.org/html/2306.08206)
- Medido, embeddings (Tabla 8): ordenar jugadores por dorsal ("naive ordering") da PPA 19.72%; las codificaciones equivariantes por jugador/equipo (PPE+FPE+FPI) dan 64.70%. Usa pertenencia a equipo; no usa roles — [arXiv HTML](https://arxiv.org/html/2306.08206)
- Medido, imputación (Tabla 3; entrenado con 80% enmascarado en la mitad de los batches). Antes de postproceso: 100% enmascarado PE 5.35 m / PPA 61.91%; 95% → 3.00 m / 79.51%; 90% → 2.09 m / 87.11%; 80% → 1.31 m / 93.13% (TPA 97.41%). El modelo entrenado mixto rinde peor a 100% (5.35 m) que el entrenado solo sin balón (3.66 m) — [arXiv HTML](https://arxiv.org/html/2306.08206)
- Medido, anotación de pases (Tabla 4): F1 de pase 0.388 (100% enmascarado), 0.597 (95%), 0.743 (90%), 0.869 (80%); F1 de pasador 0.604→0.915; F1 de receptor 0.571→0.899 — [arXiv HTML](https://arxiv.org/html/2306.08206)
- No hay análisis por longitud de hueco; el proxy es el porcentaje de enmascarado. No se reporta tiempo de inferencia — [arXiv HTML](https://arxiv.org/html/2306.08206)
- Repo: github.com/pientist/ballradar redirige a hyunsungkim-ds/ballradar (35 estrellas, último push 2026-02-03). Python 3.10, PyTorch 1.13. No trae pesos preentrenados. Incluye eventos de Metrica preprocesados; los datos Fitogether no se comparten. La API de GitHub no devuelve campo de licencia — [GitHub](https://github.com/pientist/ballradar)

### Inferences
- Especulativo: nuestro partido tiene ~59% de frames con balón, más cerca del régimen "imputación" (80-90% enmascarado → 1.3-2.1 m, poseedor 87-93%) que del régimen "sin balón". Pero el enmascarado de Ball Radar no está descrito como huecos largos y contiguos. Nuestros huecos (mediana ~10.7 s) equivalen a una ventana entera de 10 s sin balón, así que dentro del hueco el error esperable estaría entre ambos regímenes, más cerca de 3-5 m.
- Especulativo: sin licencia declarada, el código queda con todos los derechos reservados por defecto. Para usarlo en el hackathon convendría reimplementarlo o pedir permiso a los autores.

### Gaps
- Error en función de la longitud de un hueco contiguo: no reportado.
- No se encontró la tabla con el resultado "solo Metrica" (el README remite a la Tabla 5, pero la Tabla 5 del HTML es zoom-in ROI). No se verificó.
- Coste en CPU: no reportado.

## 2. TranSPORTmer, Peral et al. (multimodal), UniTraj, U2Diff, Graph Imputer

### Takeaway
TranSPORTmer mejora a Ball Radar sin necesitar velocidades (2.57-2.71 m frente a 3.89 m en LaLiga, a 6.25 Hz). Peral et al. 2025 bajan a 1.17 m añadiendo recortes de imagen, que nosotros no tenemos. UniTraj modela el balón como agente pero no da error específico del balón. Graph Imputer imputa jugadores, no balón.

### Cited Findings
- TranSPORTmer: Guillem Capellera et al., ACCV 2024 (LNCS pp. 333-352, DOI 10.1007/978-981-96-0901-7_20). Transformer de Set Attention Blocks; una máscara decide la tarea (forecasting, imputación, inferencia del balón). Reporta un 25% de mejora sobre el estado del arte en inferencia del balón — [IRI-UPC](https://www.iri.upc.edu/publications/show/2921); [ACCV PDF](https://openaccess.thecvf.com/content/ACCV2024/papers/Capellera_TranSPORTmer_A_Holistic_Approach_to_Trajectory_Understanding_in_Multi-Agent_Sports_ACCV_2024_paper.pdf)
- Medido, TranSPORTmer, inferencia sin balón en LaLiga 2022-23: 283 partidos, secuencias de 9.6 s (60 frames a 6.25 Hz), 82,954/7,500/6,258 train/val/test. Ball Radar 3.89 m ADE / 8.79 m MaxErr; Ball Radar sin poseedor 5.43 / 10.98; TranSPORTmer 2.71 / 7.39 con 80.84% de acierto en el estado (pase / posesión / descontrolado / fuera de juego); con los datos originales que tienen huecos, 2.57 m ADE, 7.22 MaxErr, 81.64% — [arXiv 2410.17785](https://arxiv.org/html/2410.17785v2)
- Medido, imputación del balón: con 80% enmascarado, Ball Radar 0.88 m ADE y TranSPORTmer 0.80-0.88; con 90%, Ball Radar 1.18 y TranSPORTmer 1.09-1.23. La asignación exacta de cada variante en la tabla extraída es ambigua — [arXiv 2410.17785](https://arxiv.org/html/2410.17785v2)
- TranSPORTmer "does not require player velocities"; para compararlo, a Ball Radar se le dio poseedor real, velocidades y posición de porteros. Entrenamiento: RTX A6000, 100 epochs; embedding 128, 16 cabezas. No se menciona código publicado. El paper es CC BY-NC-SA 4.0 — [arXiv 2410.17785](https://arxiv.org/html/2410.17785v2)
- Peral, Capellera, Ferraz, Rubio, Agudo (IRI y Kognia), "Multi-Modal Soccer Scene Analysis with Masked Pre-Training", arXiv 2512.19528, dic. 2025. Entradas: trayectorias de jugadores, tipo de jugador (equipo) y recortes de imagen por jugador de la cámara táctica; no usa el balón como entrada. Medido: poseedor 89.56% (CropDrop) frente a TranSPORTmer 83.79% y Ball Radar 78.12%; ADE/MaxErr 1.17/4.77 m frente a 1.72/6.56 (TranSPORTmer) y 1.86/7.02 (Ball Radar); estado 85.85% frente a 79.77%. Datos propietarios; no se menciona código. Critica que Ball Radar depende de velocidades precisas, difíciles con tracking ruidoso — [arXiv 2512.19528](https://arxiv.org/html/2512.19528v1)
- UniTraj / Sports-Traj (Xu & Fu, ICLR 2025): el balón es un agente más en Basketball-U, Football-U y Soccer-U. Máscaras: hueco central de 25-40 de 50 frames, agentes aleatorios enmascarados, etc. Métricas agregadas sobre todos los agentes, en pies/yardas/píxeles; no da error del balón. 1.77M parámetros, entrenado en una A100. Código, pesos y datos en colorfulfuture/UniTraj-pytorch, Apache-2.0 según la API de GitHub — [arXiv 2405.17680](https://arxiv.org/html/2405.17680v2); [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/03a9a9c1e15850439653bb971a4ad4b3-Paper-Conference.pdf); [GitHub](https://github.com/colorfulfuture/UniTraj-pytorch)
- U2Diff: difusión unificada con incertidumbre para completar trayectorias multiagente; compara con UniTraj. Solo vi un resumen secundario, sin números — [resumen](https://liner.com/review/unified-uncertaintyaware-diffusion-for-multiagent-trajectory-modeling)
- Graph Imputer (Omidshafiei et al., DeepMind): arXiv 2106.04219 (2021), en Scientific Reports 2022 como "Multiagent off-screen behavior prediction in football", con 105 partidos. Imputa jugadores fuera de plano en broadcast y lo aplica a pitch control; el abstract no menciona el balón ni números — [arXiv](https://arxiv.org/abs/2106.04219); [Sci Rep](https://www.nature.com/articles/s41598-022-12547-0)
- Antecedentes: Amirli et al. estiman el balón desde la formación con errores de 7.56 m (x) y 5.01 m (y) — [citado en Ball Radar](https://arxiv.org/html/2306.08206). Wang, Ablavsky, Ben Shitrit, Fua (CVIU, 2014), "Take your Eyes off the Ball", plantean el tracking como decidir quién tiene el balón, en baloncesto y fútbol — [EPFL](https://infoscience.epfl.ch/record/185107)
- Preprint de julio de 2026 (arXiv 2607.11548) sobre jugadores ocultos en broadcast: ignorarlos infla el error de pitch control en la zona oculta a 25.1-26.9 puntos; propone imputación sin entrenamiento. Es sobre jugadores, no balón — [arXiv 2607.11548](https://arxiv.org/abs/2607.11548)

### Inferences
- Especulativo: las cifras de distintos papers no se comparan directamente. Usan datasets y frecuencias distintos (10 Hz GPS frente a 6.25 Hz óptico), y Ball Radar da 3.66 m en su dataset pero 3.89 m en LaLiga.

### Gaps
- Inferencia del balón en baloncesto (Zhan et al., NBA): no encontré un trabajo específico de inferencia del balón; no lo verifiqué.
- "BallNet": no encontré ninguna fuente.
- Código de TranSPORTmer: no encontré repo público.

## 3. Efecto en la detección de posesión/eventos (balón inferido, real o ausente)

### Takeaway
La evidencia más directa es PathCRF (Kim et al., KDD 2026). Usa exactamente nuestros 7 partidos Sportec/DFL y detecta eventos sin balón: F1 de eventos 75.69% frente a 48.39% de la cadena "regresar el balón + heurísticas tipo Ball Radar". Ningún trabajo encontrado compara balón inferido con balón real en la misma tarea de posesión.

### Cited Findings
- PathCRF: "PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories", Hyunsung Kim, Kunhee Lee, Sangwoo Seo, Sang-Ki Ko, Jinsung Yoon, Chanyoung Park. arXiv 2602.12080 (feb. 2026, v2 jun. 2026), KDD 2026. Modela a los jugadores como un grafo dinámico; un CRF con Viterbi elige en cada instante la arista de posesión, y cada cambio de arista es un evento (control, pase) — [arXiv](https://arxiv.org/abs/2602.12080)
- Datos de PathCRF: Sportec Open DFL Dataset, 7 partidos de Bundesliga 1 y 2, split 5/1/1, bajado de 25 a 5 FPS — [arXiv HTML](https://arxiv.org/html/2602.12080)
- Medido, PathCRF (test): precisión 73.18%, recall 78.37%, F1 75.69%. Acierto del emisor 80.44%, del receptor 82.28%, de la arista (entre 676 candidatas) 69.64%, de posesión por equipo frame a frame 92.40%. Recall de eventos ~80% con tolerancia de 1 s y 4 m, y 86.1% con 2 s y 10 m — [arXiv HTML](https://arxiv.org/html/2602.12080)
- Medido, baselines: "Ball TP" (regresión del balón más heurísticas tipo Ball Radar) F1 48.39%. Ablación del backbone: TranSPORTmer 68.19%, Ball Radar 75.24%, híbrido 75.69% — [arXiv HTML](https://arxiv.org/html/2602.12080)
- Medido, validación cruzada leave-one-match-out en Sportec: arista 66.75 ± 2.76%, F1 73.88 ± 1.90%. K League (GPS, 15 partidos): 68.20% / 79.83%. Entre proveedores cae: K League→Sportec 52.71% / 66.25%; Sportec→K League 41.65% / 49.16% — [arXiv HTML](https://arxiv.org/html/2602.12080)
- No hay comparación con el balón real ni análisis por duración de la posesión; no hay desglose por tipo de evento — [arXiv HTML](https://arxiv.org/html/2602.12080)
- Ball Radar, indirectamente: el F1 de pase sube de 0.388 (sin balón) a 0.869 (20% del balón visible), lo que mide el valor de los fragmentos de balón observado — [arXiv HTML](https://arxiv.org/html/2306.08206)
- Código de PathCRF: github.com/hyunsungkim-ds/pathcrf, licencia MPL-2.0 según la API de GitHub (26 estrellas, push 2026-06-08) — [GitHub](https://github.com/hyunsungkim-ds/pathcrf)

### Inferences
- Especulativo, pero con una cifra de referencia: como PathCRF usa nuestro mismo conjunto de 7 partidos DFL, su F1 de eventos sin balón (73.9 ± 1.9% en LOMO) es un punto de comparación directo para nuestro modelo en los tramos sin balón. Siguiente paso: ejecutar PathCRF (MPL-2.0, se puede usar) en LOMO con nuestro degradado y medir el recall en posesiones de 2-16 s.
- Especulativo: PathCRF supera claramente a "regresar el balón y aplicar reglas" (75.7% frente a 48.4%). Eso sugiere que, para detectar posesión, inferir directamente al poseedor rinde más que imputar coordenadas del balón y reutilizar las reglas.
- Especulativo: con nuestro ~59% de balón visible, una variante que combine la arista de PathCRF con el balón observado donde exista (estilo imputación de Ball Radar, Tabla 3) es lo que más se acerca a nuestra situación. Ningún paper encontrado la evalúa.

### Gaps
- No encontré ningún trabajo que reporte la exactitud de posesión/eventos con balón inferido, balón real y sin balón en el mismo setup.
- No hay resultados por longitud de posesión ni de hueco.

## 4. Robustez a tracking automático ruidoso y sin identidades; datos y coste

### Takeaway
Todos los métodos son equivariantes a permutación y solo necesitan pertenencia a equipo, no identidades ni roles; ninguno trae un benchmark de ruido. La transferencia entre proveedores cuesta 7-25 puntos de F1 en PathCRF. Nadie reporta coste en CPU; PathCRF procesa un partido en 22.9 s en GPU a 5 FPS.

### Cited Findings
- Ball Radar usa equipo y codificaciones equivariantes; ordenar por dorsal hunde el resultado (PPA 19.72%). Necesita velocidad y aceleración: con solo posición pasa de 3.66 a 4.65 m. Pierde también algo de poseedor, aunque esa comparación se hizo en un benchmark distinto — [arXiv HTML](https://arxiv.org/html/2306.08206)
- TranSPORTmer no necesita velocidades. Su modelo con los datos originales con huecos rinde igual o algo mejor (2.57 m) que con datos limpios — [arXiv 2410.17785](https://arxiv.org/html/2410.17785v2)
- Peral et al.: las velocidades precisas son difíciles de estimar con tracking ruidoso o escaso; su robustez a la oclusión es cualitativa y no tiene benchmark — [arXiv 2512.19528](https://arxiv.org/html/2512.19528v1)
- PathCRF: no hay experimentos de ruido. Sí mide transferencia entre GPS (K League) y óptico (Sportec), con caídas de hasta 30.7 puntos de F1 (Sportec→K League). Latencia en una RTX 4090: 0.24 s por episodio, 22.88 s por partido; los autores dicen que un partido de 90 min tarda menos de un minuto — [arXiv HTML](https://arxiv.org/html/2602.12080)
- Necesidad de datos: Ball Radar entrena con 2 partidos Metrica más 10 de Fitogether, y dice que el setup solo-Metrica rinde "only slightly worse" — [GitHub](https://github.com/pientist/ballradar); [arXiv HTML](https://arxiv.org/html/2306.08206). PathCRF entrena con 5 partidos — [arXiv HTML](https://arxiv.org/html/2602.12080). TranSPORTmer usa 283 partidos — [arXiv 2410.17785](https://arxiv.org/html/2410.17785v2)
- Tamaño: UniTraj tiene 1.77M parámetros — [arXiv 2405.17680](https://arxiv.org/html/2405.17680v2)

### Inferences
- Especulativo: con 1.8M parámetros (UniTraj) y menos de 25 s por partido en GPU (PathCRF), la inferencia en CPU de un solo partido debería costar minutos, no horas. No está medido.
- Especulativo: nuestro tracking automático sin identidades estables encaja con los modelos equivariantes, que tratan a los jugadores como un conjunto por equipo. El punto débil esperable son las velocidades derivadas de posiciones ruidosas (Ball Radar), lo que favorece TranSPORTmer/PathCRF o suavizar antes. Lo medible: entrenar con nuestro degradado (ruido más huecos) y evaluar por longitud de hueco.

### Gaps
- No existe ningún benchmark publicado de inferencia del balón con ruido sintético en jugadores, swaps de ID o jugadores perdidos.
- Coste en CPU: ninguno lo reporta.
- No verifiqué qué features usa PathCRF (velocidad, identidad); no se especifica en el texto revisado, habría que leer el código.
