# Posesión de balón por frame desde tracking (H/A/D) — repos, papers y segmentación temporal

Fecha de la investigación: 2026-10-28. Presupuesto: ~20 llamadas de herramientas. Algunos PDFs devolvieron 403 (ESWA/Polito), así que varios detalles no se han podido verificar y aparecen en Gaps.

## 1. Librerías open-source que asignan posesión o dueño del balón desde tracking (y cómo la definen)

### Takeaway
Ninguna librería madura del ecosistema PySport resuelve la posesión de equipo robusta a la falta de balón. **databallpy** implementa la regla de Vidal-Codina (zona de 1.5 m más un cambio de velocidad o dirección del balón como prueba de control) para la posesión individual, y una posesión de equipo que se deriva de **eventos** (la posesión cambia solo con un evento con éxito del rival), una regla muy parecida a nuestra definición de "control". kloppy, floodlight y unravelsports se limitan a leer el `ball_owning_team` o `ball_state` que trae el proveedor, o lo infieren con umbrales de distancia simples.

### Cited Findings
- **databallpy** (alek050/databallpy, PyPI): sirve para cargar, preprocesar, visualizar y sincronizar eventos con tracking. Sincroniza con Needleman-Wunsch y costes basados en la distancia del evento al balón y al jugador — [docs sync](https://databallpy.readthedocs.io/en/latest/getting_started/synchronisation_page.html), [GitHub](https://github.com/alek050/databallpy), [PyPI](https://pypi.org/project/databallpy/)
- La posesión de equipo de databallpy, `add_team_possession(event_data, home_team_id)`, no usa umbrales de distancia ni de duración. El primer evento on-ball (pass, dribble o shot) fija el equipo inicial, y la posesión solo cambia cuando el equipo rival completa con éxito un evento on-ball (`outcome == 1`). El resultado es una columna por frame con "home"/"away". La documentación enseña un ejemplo donde esta posesión, derivada de eventos, no coincide con la del proveedor de tracking — [databallpy Team and Player Possession](https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html)
- La posesión individual de databallpy (Vidal-Codina et al. 2022) es `add_individual_player_possession(pz_radius=1.5, bv_threshold=5.0, ba_threshold=10.0, min_frames_pz=0)`. Funciona en tres pasos:
  1. **Zona de posesión (PZ):** el balón está a menos de 1.5 m de un jugador. Si dos compañeros califican, gana el más cercano. Si califican rivales, se trata como duelo y queda fuera del alcance de la regla.
  2. **Control:** la posesión se valida si el balón cambia de dirección más de 10° o de velocidad más de 5 m/s mientras está en la zona. Así se descartan los balones que solo pasan por encima de un jugador.
  3. **Pérdida:** el jugador conserva la posesión hasta el último frame con el balón en su zona antes de la siguiente ganancia válida.

  Los ejemplos de la página usan otros valores (2.5 m y 1.5 m/s) y la propia página avisa de que el notebook puede no coincidir con el paquete — [databallpy docs](https://databallpy.readthedocs.io/en/main/features/team_and_player_possession.html)
- **floodlight**: guarda la posesión como objetos `Code` por segmento (home/away) y el `ballstatus` igual. El parser de TRACAB extrae "position data, possession and ballstatus codes". No he encontrado un algoritmo propio de inferencia de posesión: lee la del proveedor — [floodlight getting started](https://floodlight.readthedocs.io/en/latest/guides/getting_started.html), [floodlight TRACAB io](https://floodlight.readthedocs.io/en/latest/modules/io/tracab.html), [GitHub](https://github.com/floodlight-sports/floodlight)
- **kloppy**: estandariza tracking y eventos de muchos proveedores (Metrica y Sportec, entre otros). Tiene campos `ball_state` ("alive"/"dead") y `ball_owning_team`. Un ejemplo de Sportec usa `load_open_tracking_data(only_alive=True)` — [kloppy docs](https://kloppy.readthedocs.io), [PyPI](https://pypi.org/project/kloppy/)
- **unravelsports** (envuelve kloppy y usa polars): según su documentación, si el proveedor no trae `ball_owning_team` "it will be inferred using distance thresholds, which may be inaccurate during contested ball situations" — [unravelsports KloppyPolarsDataset](https://unravelsports.readthedocs.io/en/latest/api/generated/unravel.soccer.KloppyPolarsDataset.html), [source](https://unravelsports.readthedocs.io/en/stable/_modules/unravel/soccer/dataset/kloppy_polars.html)
- **Vidal-Codina, Evans, El Fakir, Billingham (MIT Sports Lab + FIFA), arXiv 2202.00804 (v3, agosto 2022)**: modelo de posesión determinista basado en un árbol de decisión y, después, detección de eventos (pases, tiros, centros, paradas, recepciones, intercepciones y balón parado) a partir de los cambios de posesión. Supera el 90% de detección en la mayoría de categorías, con varios torneos y proveedores — [arXiv](https://arxiv.org/abs/2202.00804v3), [MIT DSpace](https://dspace.mit.edu/handle/1721.1/145347)
- **Link & Hoernig 2017, "Individual ball possession in soccer" (PLOS ONE 12(7): e0179953)**: usa 60 partidos de la Bundesliga 2013/14 con 69,667 intervalos IBA. Estiman con ML el tiempo que el balón pasa en la esfera de influencia del jugador y clasifican el grado de control con una red bayesiana. Resultados: F = .88 para IBA (individual ball action), F = .83 para IBP (individual ball possession) y kappa = .67 para IBC (grado de control). Describe también la posesión de equipo. Datos en Figshare — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225), [figshare](https://figshare.com/articles/dataset/Individual_ball_possession_in_soccer/5190445)

### Inferences
- La regla de databallpy de "cambiar solo con un evento con éxito del rival" coincide con nuestra definición: hace falta control, y un toque, una desviación o un duelo aéreo no bastan. Una máquina de estados con **histéresis** (cambiar de equipo solo cuando el rival "controla" de forma sostenida, es decir, varios frames en la zona de posesión o una alteración de la velocidad o dirección del balón cerca de un jugador rival) debería subir mucho la spellF1 frente al baseline del jugador más cercano con voto ±2 s (0.2433), porque elimina los cambios espurios que fragmentan los spells.
- Los parámetros de Vidal-Codina/databallpy (1.5–2.5 m, 5 m/s o 10°) son un buen punto de partida para la evidencia de control, pero se apoyan en el balón. Con un 41% de frames sin balón hacen falta interpolación o inferencia del balón (sección 2).
- Se pueden usar Smart Tagging y los eventos automáticos de Metrica como "eventos" para replicar `add_team_possession` de databallpy. No es etiquetado manual, así que está permitido.

### Gaps
- No he verificado los stars, la licencia ni la fecha del último release de databallpy, floodlight, kloppy y unravelsports (no hice fetch de GitHub). Recuerdo, sin verificar, que kloppy es BSD-3 y floodlight MIT: confirmarlo.
- No he encontrado documentado el umbral exacto que usa unravelsports para inferir `ball_owning_team`.
- No he comprobado si socceraction, mplsoccer o el código de Friends-of-Tracking/LaurieOnTracking tienen asignación de posesión desde tracking. Por lo que conozco, LaurieOnTracking trabaja con los datos de muestra de Metrica y con pitch control y no tiene un módulo de posesión robusto, pero no lo he verificado en esta sesión.

## 2. Papers con código sobre inferencia de posesión o portador del balón desde tracking, especialmente robustos a la falta de balón

### Takeaway
La línea más relevante es la de Hyunsung Kim (KAIST/Fitogether). **Ball Radar** (KDD 2023) infiere el poseedor y la trayectoria del balón solo desde los jugadores, y entrena y evalúa con los **datos de muestra públicos de Metrica**. **PathCRF** (KDD 2026) infiere la "ruta de posesión" (quién tiene el balón o entre quién viaja) sin balón, con un CRF y decodificación Viterbi. Su posesión de equipo agregada queda casi exacta (62.64% predicha frente a 62.13% real). En precisión de posesión con tracking que sí incluye balón, la mejor cifra reportada es la del TCN de Borghesi et al. (ESWA 2023): 86.2%.

### Cited Findings
- **PathCRF: Ball-Free Soccer Event Detection** (Kim, Lee, Seo, Ko, Yoon, Park; KAIST, U. Seoul, Fitogether). KDD '26; arXiv 2602.12080v2, 1 de junio de 2026 — [arXiv html](https://arxiv.org/html/2602.12080)
  - Cada frame es un grafo con 22 jugadores más 4 nodos "fuera" (las líneas del campo): 26 nodos y 676 aristas. En cada instante se elige exactamente una arista. Un self-loop significa que un jugador controla el balón, y una arista dirigida significa un pase o chut en vuelo. Los eventos se detectan cuando cambia la arista elegida.
  - Backbone: set attention (ISAB/SAB) social y temporal, al estilo TranSPORTmer. El CRF tiene emisiones generadas por un MLP y transiciones **dinámicas** calculadas a partir de pares de embeddings consecutivos. Las transiciones ilegales se enmascaran con −10⁴. Se entrena con NLL vía el algoritmo forward y se decodifica con **Viterbi**.
  - Datos: Sportec Open DFL (7 partidos; 5 de train, 1 de validación y 1 de test), submuestreados de 25 a 5 fps con ventanas de 10 s.
  - Resultados en el partido de test (accuracy de arista / F1 de eventos):

    | Método | Acc. arista | F1 eventos |
    |---|---|---|
    | Ball TP (basado en Ball Radar) | 47.01% | 48.39% |
    | Sin CRF | 69.24% (2.80% de violaciones) | 65.34% |
    | Sin CRF + Viterbi con restricciones | — | 73.14% |
    | CRF estático enmascarado | — | 73.83% |
    | **PathCRF (CRF dinámico)** | **69.64%** | **75.69%** (P 73.18, R 78.37) |

  - A 25 fps el F1 sube a 77.21%, pero cada epoch tarda más de 10 veces más.
  - Con una tolerancia de 2 s y 10 m, el recall llega al 86.1%.
  - Posesión del local: 62.64% predicha frente a 62.13% real.
  - No reporta métricas por segmento ni de IoU.
- **Repo PathCRF**: [github.com/hyunsungkim-ds/pathcrf](https://github.com/hyunsungkim-ds/pathcrf). Licencia MPL-2.0, 26 stars, 15 commits. Incluye checkpoints preentrenados en `saved/100–140` (Non-CRF, Static/Dynamic, Dense/Masked CRF) y `tutorial.ipynb`. Usa kloppy y eventos ya sincronizados con ELASTIC. El preprocesado genera tracking con el estado de posesión real por frame — [GitHub](https://github.com/hyunsungkim-ds/pathcrf)
- **Ball Radar: Ball Trajectory Inference from Multi-Agent Sports Contexts Using Set Transformer and Hierarchical Bi-LSTM** (Kim et al., KDD 2023, arXiv 2306.08206) — [arXiv](https://arxiv.org/abs/2306.08206)
  - Es jerárquico: un Ball Possessor Classifier (qué jugador tiene el balón en cada instante) y un regresor de trayectoria condicionado a esas probabilidades. Usa Set Transformer y Bi-LSTM, una "reality loss" y post-procesado basado en reglas.
  - Error medio de posición del balón inferior a **3.7 m**. Accuracy de **poseedor a nivel de jugador: 64.7%**, como producto intermedio. Estas dos cifras salen del texto del PDF.
  - Datos: GPS de Fitogether (15 partidos de la K League 2020, privados) más los **3 partidos de muestra de Metrica**. Train: Metrica Sample Game 1 y 2. Validación: primera parte del Game 3. Test: segunda parte del Game 3.
  - En el Game 3 los IDs de jugador no coinciden entre tracking y eventos, así que los autores los reasignaron por distancia al balón — [arXiv PDF](https://arxiv.org/pdf/2306.08206)
- **Repo Ball Radar**: [github.com/pientist/ballradar](https://github.com/pientist/ballradar). 35 stars, 6 forks, 37 commits. La página no muestra licencia. Python 3.10 y PyTorch 1.13. Tiene un pipeline solo con Metrica que según el README sufre "little performance degradation" y scripts `player_ball.sh` y `team_ball.sh` (este último sería la variante a nivel de equipo) — [GitHub](https://github.com/pientist/ballradar)
- **Borghesi et al., "Using Temporal Convolutional Networks to estimate ball possession in soccer games"** (Expert Systems with Applications, 2023; Politecnico di Torino): TCN sobre tracking sin reglas hechas a mano. **86.2% de accuracy en posesión, +7.3 puntos sobre el estado del arte basado en reglas**, cuyo ejemplo citado son las reglas de Link & Hoernig 2017 — [IRIS Polito](https://iris.polito.it/handle/11583/2976470), [PDF](https://iris.polito.it/bitstream/11583/2976470/7/1-s2.0-S0957417423002816-main.pdf)
- **ELASTIC** (Kim et al. 2025, KAIST/Ajax/Fitogether; arXiv 2508.09238) sincroniza eventos y tracking usando solo features derivadas del tracking. Ground truth de 2,134 eventos en 3 partidos de Eredivisie. Repo: [github.com/hyunsungkim-ds/elastic](https://github.com/hyunsungkim-ds/elastic), con tutorial sobre Sportec Open DFL. Hay una versión extendida (arXiv 2608.30227, aceptada en CIKM 2026 según un post de LinkedIn, sin verificar) — [arXiv 2508.09238](https://arxiv.org/abs/2508.09238v1), [arXiv 2608.30227](https://arxiv.org/pdf/2608.30227)
- **Datos etiquetados públicos para entrenar**: Bassek, Rein, Weber y Memmert, "An integrated dataset of spatiotemporal and event data in elite soccer", Scientific Data 12:195 (feb. 2025), CC-BY 4.0. Siete partidos de la 1ª y 2ª Bundesliga con posiciones de todos los jugadores y del balón más eventos oficiales DFL. Mirror en Hugging Face (pysport/idsse-data), que indica temporada 2022/23 y TRACAB (dato sin verificar en el paper) — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11787359/), [HF](https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md)

### Inferences
- Para nuestro caso, con un **41% de frames sin balón** y sin identidades de jugador, PathCRF y Ball Radar son los únicos métodos públicos pensados explícitamente para funcionar sin balón. Ambos son **permutation-equivariant** (Set Transformer), así que no necesitan IDs: encaja con los datos de Metrica.
- Estrategia razonable:
  1. Entrenar o ajustar con Sportec Open DFL (posesión real por frame vía el preprocesado de PathCRF) y con los datos de muestra de Metrica.
  2. Inferir la probabilidad P(equipo) por frame.
  3. Combinarla con la evidencia directa del balón cuando exista.
  4. Decodificar con Viterbi y una penalización de cambio alta.

  Hay que validar con partidos de otros juegos (la ventana de test de PathCRF), nunca con el partido objetivo.
- PathCRF reporta métricas de eventos, no de spells. Su arista "self-loop de jugador" agregada por equipo da directamente H/A, y "balón en vuelo entre compañeros" se mantiene en el mismo equipo, lo que crea spells largos coherentes con nuestra métrica.
- Ball Radar entrena en 2 juegos de Metrica, el mismo proveedor y formato que el hackathon, aunque con tracking corregido. Su accuracy de poseedor de 64.7% es a nivel de **jugador**. A nivel de equipo debería ser bastante mayor, pero no tengo la cifra.

### Gaps
- No pude leer el PDF de Borghesi et al. (403): no sé el dataset, los fps, si las clases incluyen "nadie" o balón muerto, ni si hay código público. Buscar el repo del grupo de Polito (Cagliero, Vaccaro?).
- No he extraído la accuracy de Ball Radar a nivel de equipo (variante `team_ball`) ni la tabla 5 (solo Metrica).
- No he encontrado papers de "posesión de equipo desde la forma del equipo, sin balón" con cifras aparte de PathCRF y Ball Radar. No he verificado TranSPORTmer (Capellera et al. 2024), que también infiere el estado del balón; buscar su repo y sus cifras.
- No sé la licencia de Ball Radar.

## 3. Segmentación temporal de la posesión (HMM, Viterbi, CRF, change-point, MS-TCN/ASFormer) y métricas a nivel de segmento o IoU

### Takeaway
El ejemplo publicado más directo es PathCRF: un CRF lineal con transiciones dinámicas enmascaradas y Viterbi. Incluso **"no-CRF + Viterbi con restricciones"** sube el F1 de eventos de 65.3 a 73.1, lo que demuestra que la decodificación estructurada por sí sola aporta mucho. No he encontrado trabajos que apliquen HMM, MS-TCN o ASFormer a la posesión de equipo por frame con tracking, ni que reporten F1@IoU de segmentos de posesión. La métrica F1@IoU de segmentos (Lea et al. 2017) es, sin embargo, la misma familia que nuestra spellF1.

### Cited Findings
- PathCRF hace una ablación de la estructura temporal (F1 de eventos): sin CRF 65.34 (2.80% de transiciones ilegales); sin CRF + Viterbi con restricciones 73.14; CRF estático enmascarado 73.83; CRF dinámico enmascarado 75.69. Ninguno comete violaciones salvo el modelo sin CRF — [arXiv 2602.12080](https://arxiv.org/html/2602.12080)
- El TCN para segmentación de acciones (Lea et al., arXiv 1611.05267) introdujo el F1 segmental: un segmento predicho es TP si su IoU con el real supera τ. La métrica penaliza la sobre-segmentación y no castiga los desplazamientos pequeños — [arXiv 1611.05267](https://arxiv.org/pdf/1611.05267)
- MS-TCN (arXiv 1903.01945) apila etapas con convoluciones dilatadas sin pooling que refinan la predicción anterior. En 50Salads: Acc 80.7, Edit 67.9, F1@{10,25,50} = 76.3/74.0/64.5 — [paperswithcode](https://paperswithcode.com/paper/ms-tcn-multi-stage-temporal-convolutional), [MS-TCN++ arXiv 2006.09220](https://arxiv.org/abs/2006.09220v1)
- HMM en fútbol:
  - Ötting, Langrock y Maruotti usan HMM sobre estadísticas por minuto de la Bundesliga, con estados interpretados como "momentum" o control. Es a nivel de partido, no posesión por frame — [Springer AStA](https://link.springer.com/10.1007/s10182-021-00395-8)
  - Un estudio japonés modela con HMM las secuencias de balón y posiciones en recuperaciones de balón, para clasificar el resultado ofensivo — [J-STAGE](https://www.jstage.jst.go.jp/article/jcoaching/31/1/31_11/_article/-char/en)
  - Hay un modelo de mezcla de cadenas de Markov por posesión, con un estado absorbente para la pérdida, sobre eventos de StatsBomb de la Ligue 1 — [arXiv 2511.14297](https://arxiv.org/abs/2511.14297v1)
  - Ninguno segmenta la posesión de equipo por frame desde tracking.

### Inferences
- Para optimizar la spellF1 (IoU de spells entre 0.3 y 0.7), lo crítico es **no fragmentar** los spells ni fusionar dos reales. Un HMM o semi-Markov de 3 estados (H, A, D) cuyas emisiones sean probabilidades por frame (de un clasificador como un TCN o un set transformer, o de reglas: proximidad, control del balón, Smart Tagging de fase, balón fuera) y con una matriz de transición de diagonal muy alta (coste de cambio afinado en partidos de validación) es la aplicación directa del resultado "Viterbi con restricciones" de PathCRF.
- Las restricciones duras útiles son: H↔A solo con un evento de control. D solo cuando el balón sale o hay una pausa según Smart Tagging. Duración mínima de un spell, con un semi-Markov o HSMM y un prior de duración aprendido de Sportec o Metrica.
- MS-TCN y ASFormer encajan como clasificadores por frame con suavizado aprendido (añaden una pérdida de suavizado T-MSE). Se pueden entrenar con features de tracking de otros partidos (Sportec, Metrica sample) y evaluarse con F1@IoU, que es casi idéntica a nuestra métrica. No hay precedente publicado en fútbol: es una apuesta.

### Gaps
- No he encontrado papers de change-point detection (PELT, BOCPD) aplicados a la posesión con tracking.
- No he encontrado ningún trabajo que reporte F1 de segmentos o IoU de spells de posesión. Nuestra métrica no tiene benchmark externo.

## 4. ¿Qué ha funcionado mejor, cuantitativamente, y en qué datasets?

### Takeaway
Las cifras no son comparables entre sí: distintos datasets, niveles (jugador frente a equipo) y métricas.

| Enfoque | Datos | Resultado | Fuente |
|---|---|---|---|
| ML + red bayesiana (Link & Hoernig) | Bundesliga 13/14, 60 partidos | IBP F = .83 | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225) |
| Árbol de decisión determinista (Vidal-Codina) | Varios torneos y proveedores | Más del 90% de detección de eventos en la mayoría de categorías | [arXiv](https://arxiv.org/abs/2202.00804v3) |
| TCN (Borghesi) | Tracking con balón | **86.2%** de accuracy de posesión, +7.3 puntos sobre reglas | [IRIS](https://iris.polito.it/handle/11583/2976470) |
| Ball Radar, sin balón | Fitogether GPS + Metrica sample | 64.7% de poseedor a nivel de jugador; error de balón < 3.7 m | [arXiv](https://arxiv.org/pdf/2306.08206) |
| PathCRF, sin balón | Sportec Open DFL | Arista 69.6%; F1 de eventos 75.7%; posesión del local agregada con error de 0.5 puntos | [arXiv](https://arxiv.org/html/2602.12080) |

### Cited Findings
- Ver las cifras de la tabla y de las secciones 1 a 3, con sus fuentes.
- La posesión del proveedor y la derivada de eventos pueden discrepar en tramos, por errores de tracking o por definiciones distintas — [databallpy](https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html)

### Inferences
- Ranking priorizado para el hackathon:
  1. **Máquina de estados o HMM con histéresis y Viterbi** sobre la evidencia de control (al estilo Vidal-Codina/databallpy) más el Smart Tagging para D. Coste bajo, sin entrenamiento y con una ganancia esperada grande sobre el baseline.
  2. **PathCRF o Ball Radar** (pesos preentrenados, MPL-2.0 en el caso de PathCRF) como emisión ball-free para el 41% de frames sin balón, ajustado con Sportec y Metrica sample.
  3. **TCN o MS-TCN supervisado** con features de tracking entrenado en Sportec Open DFL (posesión real por frame) y Metrica sample.
- Licencias: PathCRF (MPL-2.0) y Sportec (CC-BY 4.0) son aptos. La licencia de Ball Radar está por confirmar.

### Gaps
- No hay un benchmark común. Ninguna fuente evalúa con tracking automático sin corregir de Metrica ni con un 41% de balón ausente.
- Falta la precisión de equipo por frame de Ball Radar y PathCRF (PathCRF solo da el agregado), y la licencia y el código de Borghesi.
