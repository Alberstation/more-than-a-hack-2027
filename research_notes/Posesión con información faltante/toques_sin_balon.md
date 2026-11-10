# Toques, portador y cambios de posesión sin balón (o con balón débil): señales de jugadores y contexto de equipo

Alcance: literatura 2018-2026 que infiere poseedor/toques/cambios de posesión a partir de la cinemática de jugadores y del contexto colectivo. Revisé el texto completo de tres trabajos clave: Ball Radar (KDD'23), PathCRF (2026) y Multi-Modal Soccer Scene Analysis (2025). El resto lo vi solo por abstract o snippet, y en cada caso lo indico.

## 1. ¿Qué trabajos detectan cambios de posesión, toques o poseedor solo con tracking de jugadores? Precisión con y sin balón

### Takeaway
Hay tres líneas maduras que usan solo jugadores: Ball Radar (Set Transformer + Bi-LSTM jerárquico), TranSPORTmer / Multi-Modal MAE (Set Attention coarse-to-fine) y PathCRF (grafo + CRF + Viterbi). Sin balón, la posesión por equipo frame a frame ronda el 85 %, el poseedor individual el 64-82 % y la detección de eventos de posesión un F1 de ~0.76. La cifra más útil para nosotros: con solo un 5-20 % de frames de balón visibles, la posesión por equipo sube del 85 % al 91-97 %. El balón parcial más los jugadores es el régimen correcto, y nuestro 59 % de frames visibles está muy por encima de ese umbral.

### Cited Findings
**Ball Radar (Kim et al., KDD'23, arXiv 2306.08206)**
- La entrada por jugador son 6 features: (x, y), (vx, vy), speed y aceleración. No usa identidades. Los jugadores se codifican con Set Transformer en variantes permutation-equivariant e invariant (PPE/FPE/FPI) por equipo. Una jerarquía primero predice qué jugador posee el balón (Player Possession Classifier) y luego regresa la trayectoria condicionada a esa posesión — [arXiv 2306.08206](https://arxiv.org/abs/2306.08206)
- Datos: Metrica sample data más GPS de Fitogether (mixed). El código está en [github.com/pientist/ballradar](https://github.com/pientist/ballradar). No he verificado la licencia del repositorio — [arXiv PDF](https://arxiv.org/pdf/2306.08206)
- Resultados sin balón (Tabla 2). H-LSTM-RL: error de posición 3.66 m, posesión por jugador (PPA) 64.70 %, posesión por equipo (TPA) 85.85 %. La variante no jerárquica LSTM: PE 5.47 m y sin salida de posesión. H-Transformer: PPA ~49-52 % y TPA ~78 %. Las LSTM superan a los Transformers porque "las últimas posiciones de jugadores son el contexto más importante" — [arXiv PDF](https://arxiv.org/pdf/2306.08206)
- Hacer la posesión por jugador objetivo intermedio (tarea auxiliar) mejora la trayectoria: los jerárquicos superan a los no jerárquicos — [arXiv PDF](https://arxiv.org/pdf/2306.08206)
- **Con balón parcial** (Tabla 3, H-LSTM-RL antes del postproceso; enmascarando 100/95/90/80 % del balón):

| Balón enmascarado | PE | PPA | TPA |
|---|---|---|---|
| 100 % | 5.35 m | 61.9 % | 84.0 % |
| 95 % | 3.00 m | 79.5 % | 91.5 % |
| 90 % | 2.09 m | 87.1 % | 95.0 % |
| 80 % | 1.31 m | 93.1 % | 97.4 % |

  El entrenamiento enmascara el 80 % del balón en la mitad de los batches y el 100 % en la otra mitad — [arXiv PDF](https://arxiv.org/pdf/2306.08206)
- Detección de pases (Tabla 4, pase correcto si cae dentro de ±2 s). Con el 100 % enmascarado: F1 pase 0.39, pasador 0.60, receptor 0.57. Con el 80 % enmascarado: pase 0.87, pasador 0.91, receptor 0.90 — [arXiv PDF](https://arxiv.org/pdf/2306.08206)
- Un "possession score" (probabilidad dividida por la distancia al balón estimado) da picos más nítidos que la probabilidad sola y permite separar "intervalos de transición", en los que el balón aún no ha llegado al nuevo poseedor — [arXiv PDF](https://arxiv.org/pdf/2306.08206)

**PathCRF (Kim et al., 2026, arXiv 2602.12080, DOI 10.1145/3770855.3818152)**
- Usa un grafo dinámico de 26 nodos: 22 jugadores más 4 nodos de "fuera". En cada instante se elige una arista: self-loop = control, emisor→receptor = balón en vuelo. Un CRF lineal con transiciones enmascaradas (prohibidas = −∞) y decodificación Viterbi produce la secuencia, y hay evento cuando cambia la arista. Los módulos son permutation-equivariant dentro de cada equipo. El balón se usa solo para crear las etiquetas — [arXiv HTML](https://arxiv.org/html/2602.12080v2)
- Datos: Sportec Open DFL (IDSSE), 7 partidos (5 train / 1 val / 1 test), ventanas de 10 s a 5 FPS. Test: 91 episodios y 1 831 eventos — [arXiv HTML](https://arxiv.org/html/2602.12080v2)
- Resultados en edge acc / event F1. Ball TP (regresar el balón y aplicar heurísticas) 47.0 % / 48.4 % con error de balón de 2.74 m. Non-CRF 69.2 % / 65.3 %. Non-CRF + Viterbi 69.7 % / 73.1 %. PathCRF (dynamic masked CRF) 69.6 % / **75.7 %**, con P = 73.2 % y R = 78.4 %. A 25 FPS el F1 sube a 77.2 %, pero el entrenamiento es unas 12 veces más lento — [arXiv HTML](https://arxiv.org/html/2602.12080v2)
- Recall con tolerancia: ~80 % a 1 s / 4 m y 86.1 % a 2 s / 10 m. El reparto de posesión del equipo local sale en 62.64 % predicho frente a 62.13 % real — [arXiv HTML](https://arxiv.org/html/2602.12080v2)
- Ablación de pérdidas auxiliares (Tabla 7, dynamic MCRF, edge acc / F1). Completo 69.6 / 75.7. Sin L_coarse (auxiliar de la primera etapa) 67.5 / 72.8. Sin L_emit 65.8 / 70.7. Sin ambas 65.6 / 69.7. Es decir, las cabezas auxiliares aportan unos 6 puntos de F1 — [arXiv HTML](https://arxiv.org/html/2602.12080v2)
- Código: [github.com/hyunsungkim-ds/pathcrf](https://github.com/hyunsungkim-ds/pathcrf). El paper es CC BY 4.0 y la licencia del código no consta — [arXiv HTML](https://arxiv.org/html/2602.12080v2)

**TranSPORTmer (Capellera et al., 2024, arXiv 2410.17785) y Multi-Modal Soccer Scene Analysis (Peral et al., arXiv 2512.19528)**
- TranSPORTmer reporta una mejora del ~25 % frente al SOTA en inferencia de balón y añade clasificación del estado del balón: pase, posesión, incontrolado, fuera. Solo vi el abstract — [arXiv 2410.17785](https://arxiv.org/html/2410.17785v2)
- Peral et al. comparan sin balón (secuencias de 9.6 s a 6.25 Hz; 43 397 / 3 968 / 3 347 secuencias). Ball Radar: ADE 1.86 m, poseedor 78.1 %. TranSPORTmer: ADE 1.72 m, estado 79.8 %, poseedor 83.8 %. Su modelo (trayectorias + tipo de jugador + recortes de vídeo): ADE 1.45 m, estado 84.4 %, poseedor 87.7 %. Con CropDrop llega a ADE 1.17 m y poseedor 89.6 % — [arXiv 2512.19528](https://arxiv.org/pdf/2512.19528)
- Ablación por modalidad. Solo trayectorias: ADE 1.97 m, estado 77.8 %, poseedor 82.1 %. Trayectorias + tipo de agente (equipo/rol): 1.78 m, 79.6 %, 83.7 %. Solo tipos o solo recortes: ADE de 20-24 m, aunque mantienen un poseedor del 82 % gracias a pistas visuales — [arXiv 2512.19528](https://arxiv.org/pdf/2512.19528)
- Enmascarar a jugadores completos durante toda la secuencia funciona mejor que enmascarar frames sueltos aleatorios, y se parece más a los fallos reales (oclusión o salida de cámara durante segundos) — [arXiv 2512.19528](https://arxiv.org/pdf/2512.19528)
- Los propios autores reconocen que el modelo falla cuando la secuencia empieza o acaba sin poseedor, por falta de contexto temporal — [arXiv 2512.19528](https://arxiv.org/pdf/2512.19528)

**Otros**
- La EPFL ("Take your Eyes off the Ball", Maksai et al., anterior a 2018) mejora el tracking del balón razonando primero qué jugador lo tiene — [EPFL](https://infoscience.epfl.ch/bitstreams/e13784ff-53cd-4688-aad3-4c92abbe0f83/download)

### Inferences
- Nuestro régimen (59 % de frames con balón suavizado) cae entre las filas "90 % enmascarado" y "80 % enmascarado" de Ball Radar, o mejor. Su TPA del 95-97 % frente a nuestro problema en spells cortos sugiere que lo que falta no es cobertura de balón, sino resolución temporal en los cambios rápidos. Las métricas de esos papers son por frame y no por spell, así que no son comparables directamente con spellF1.
- La lección transferible de PathCRF es estructural: una variable latente por jugador (quién controla o quién recibe) restringida por transiciones legales (control → pase → control) y decodificada con Viterbi. Nuestro Viterbi de 3 estados (H/A/D) no puede expresar "en vuelo entre A y B". Un estado latente de "balón en tránsito/disputa" separaría los spells cortos de los duelos. Una opción ligera es un Viterbi de 4-5 estados (H-control, A-control, en-vuelo, disputa, muerto) con transiciones prohibidas.
- Las cabezas auxiliares por jugador (quién posee) aparecen en los tres trabajos como un beneficio medido: unos 6 puntos de F1 en PathCRF, y mejor trayectoria en Ball Radar. Nuestro DeepSets ya tiene cercanía al balón; la variante análoga sería una cabeza por tracklet "es el poseedor" entrenada con IDSSE/Metrica, cuyo pooling alimente la cabeza de equipo.
- Lo de LSTM > Transformer en Ball Radar es coherente con que el contexto local en el tiempo domine. Nuestra ventana de ±50 s puede diluir la señal de los spells < 2 s, y un componente de ventana corta (±2-5 s) con más peso merece una prueba.
- Todos los modelos usan posición, velocidad y aceleración por jugador. Ninguno publica la importancia de la aceleración aislada.

### Gaps
- No encontré cifras de PathCRF ni de Ball Radar desglosadas por duración de posesión (< 2 s) ni para duelos. PathCRF no discute los casos de fallo en el texto que leí.
- No he verificado las licencias de ballradar y pathcrf (hay que mirar el LICENSE del repositorio).
- Ninguno evalúa con balón degradado o suavizado (solo enmascarado limpio) ni con tracklets fragmentados: todos asumen 22 jugadores con trayectoria continua.

## 2. Señales colectivas (centroide, compacidad, altura de línea, jugadores girando hacia su portería) que indican pérdida o robo, y con qué retardo

### Takeaway
La literatura táctica fija las ventanas de reacción a la transición en 2-5 s tras la pérdida: contrapresión en ≤ 5-6 s, features clave a los 2-3 s, y repliegue de la línea medido a 10 s. Con fuentes primarias no encontré ningún estudio que mida explícitamente el retardo entre el cambio de posesión y la inversión de velocidad del centroide como detector.

### Cited Findings
- Bauer & Anzer (Data Min. Knowl. Disc. 2021) plantean la detección de contrapresión como aprendizaje supervisado con features de experto sobre tracking + eventos, aplicado a 6 temporadas completas de la Bundesliga. Enmarcan la contrapresión como la recuperación en los primeros ~5 s tras la pérdida. No pude acceder al texto completo (paywall), así que no tengo AUC ni lista de features — [Springer](https://link.springer.com/article/10.1007/s10618-021-00763-7); [Crossref](https://api.crossref.org/works/10.1007%2FS10618-021-00763-7)
- Según un resumen secundario del mismo trabajo, la ventana principal son los primeros 2 s tras la pérdida, con algunas features hasta 3 s, y un stretch index rival más alto (menos compacto) a los 2 s aumenta la probabilidad de contrapresión. Es fuente secundaria (snippet de búsqueda) y no la he verificado en el original — [Springer](https://link.springer.com/article/10.1007/s10618-021-00763-7)
- Otro trabajo que cita a Bauer & Anzer describe la ventana de detección como 6 s tras la pérdida. Soccerment define la contrapresión como una acción defensiva en ≤ 6 s en el 40 % ofensivo del campo. Son definiciones heterogéneas — [Soccerment GPI](https://blog.soccerment.com/?p=27977)
- Zhang et al. (2025, SAGE) encuentran que más de una cuarta parte de las fases de transición duran < 3 s y que las ventanas fijas pueden perderlas. Citan la ventana de 3 s de Frencken et al. — [Zhang et al. 2025](https://journals.sagepub.com/doi/full/10.1177/17543371251392456)
- En 153 partidos de la Bundesliga, la compacidad del equipo completo apenas difiere entre jugadas defensivas exitosas y fallidas. La compacidad local de los defensores más cercanos influye más. Lo vi solo en snippet; el artículo es "Prediction of defensive success in elite soccer using ML" — [ResearchGate](https://www.researchgate.net/publication/372513984_Prediction_of_defensive_success_in_elite_soccer_using_machine_learning_-_Tactical_analysis_of_defensive_play_using_tracking_data_and_explainable_AI)
- En un caso reproducible con Metrica, la distancia centroide-balón pasa de 4.9 m a 22.0 m y luego a 6.6 m en una transición de 11 s. Es solo un ejemplo, no un estudio — [GitHub HyzenTech](https://github.com/HyzenTech/football-tracking-compactness-analysis)
- Un texto docente define métricas de transición como el retroceso de la línea defensiva en los 10 s posteriores a la pérdida. No es revisado por pares — [datafield.dev](https://datafield.dev/professional-soccer-analytics/part-02/chapter-11/)

### Inferences
- Con una reacción colectiva de 2-5 s y posesiones < 2 s, el equipo a menudo no llega a "reaccionar" antes de que la posesión cambie otra vez. Esto explica por qué Voronoi o forma no aportaron nada en los spells cortos: la señal colectiva es lenta frente a la duración del spell. Las señales colectivas sirven para fijar el lado de los spells largos, no para los micro-spells.
- La señal que encaja con un spell corto es la individual y local: el poseedor desacelera o cambia de dirección al recibir, y el rival más cercano acelera hacia él. Lo que dice la literatura (Bauer & Anzer, compacidad local) es que lo local pesa más que lo global.
- Como el Viterbi ve ±50 s, un spell corto de A entre dos de H sin reacción colectiva de H (sin repliegue, centroide de H aún avanzando) probablemente sea un toque de A sin control. Esto puede ayudar a decidir si se etiqueta el spell, pero según la definición del reto un toque breve puede contar como posesión.

### Gaps
- No encontré ninguna medición publicada del retardo entre la pérdida y la inversión del signo de la velocidad del centroide, ni del número de jugadores que se mueven hacia su propia portería, como detector de cambios de posesión. Lo más cercano son las ventanas fijas de 2-10 s anteriores.
- No hay AUC ni importancia de features de Bauer & Anzer (paywall).

## 3. GNN / modelos de interacción sin identidades (TacticAI, receptor de pase)

### Takeaway
Los modelos de grafo o de conjuntos para predecir el receptor funcionan sin identidad (invarianza a permutaciones y simetrías del campo), pero las cifras dependen mucho del protocolo: TacticAI declara un top-3 del 78 % en córners y una reimplementación independiente se queda en el 47 %. Para la posesión continua, el diseño social PPE/FPE de Ball Radar es el mejor backbone según la ablación de PathCRF.

### Cited Findings
- TacticAI (Wang et al., Nature Communications 2024): GAT con convoluciones de grupo D2 (simetrías de reflexión), 7 176 córners de la Premier League. Predicción de receptor top-3 = 0.782 ± 0.039 — [Nature Comms](https://www.nature.com/articles/s41467-024-45965-x); [ar5iv](https://ar5iv.labs.arxiv.org/html/2310.10553)
- La reimplementación (arXiv 2026) obtiene un top-3 del 47.43 %, con expertos humanos en 23.75 % ± 4.52 %. La discrepancia no está explicada — [arXiv 2601.00748](https://arxiv.org/pdf/2601.00748) (snippet de búsqueda; está por confirmar que sea el mismo artículo)
- Ablación del backbone en PathCRF (event F1 / edge acc). Diseño de TranSPORTmer 68.2 / 63.0. Diseño de Ball Radar 75.2 / 70.4. Híbrido (social PPE-FPE + temporal PE-SAB) 75.7 / 69.6 — [arXiv HTML](https://arxiv.org/html/2602.12080v2)
- Rahimian, Davis & Toka (Machine Learning 2026) usan Temporal Graph Networks para estimar la probabilidad de recepción por compañero o de intercepción por rival, eligiéndolas por su invarianza a permutaciones. Solo vi el abstract, sin cifras — [mlanthology](https://mlanthology.org/mlj/2026/rahimian2026mlj-temporal)
- Hay un MPNN de grafo en estrella (pasador en el centro, compañeros como satélites, agregación invariante) para la selección de receptor. No tengo las cifras — [arXiv 2605.25696](https://arxiv.org/html/2605.25696v1)

### Inferences
- TacticAI usa snapshots estáticos y requiere jugadores completos. Lo aprovechable para nosotros es el aumento por simetría (reflejos en x/y) como data augmentation en IDSSE/Metrica, que es barato y libre de identidad.
- La anticipación del receptor (un jugador que acelera hacia un punto libre mientras el balón aún "está" en otro) es exactamente lo que modela la arista emisor→receptor de PathCRF. Una feature HGB aproximada sería, por equipo, el máximo de la velocidad proyectada hacia la posición del balón estimada y desplazada t + 0.5-1 s. No la he encontrado evaluada como feature aislada.

### Gaps
- No encontré trabajos de GNN que operen con tracklets fragmentados (número variable de nodos y sin continuidad). Todos asumen 22 nodos estables. Los DeepSets por frame lo toleran mejor.
- No hay cifras de Rahimian et al. ni del MPNN de 2026.

## 4. Aprendizaje multitarea y con etiquetas débiles para detectar toques o poseedor (cabezas auxiliares, objetivos por jugador)

### Takeaway
La multitarea con un objetivo por jugador está bien respaldada. PathCRF pierde unos 6 puntos de F1 sin sus pérdidas auxiliares, en Ball Radar el poseedor intermedio mejora la trayectoria, y en Peral et al. las tareas conjuntas (balón + estado + poseedor) más el enmascarado estructurado mejoran todas las métricas. No encontré trabajos específicos sobre etiquetas ruidosas de toque.

### Cited Findings
- PathCRF: L_CRF + λ1·L_coarse + λ2·L_emit. Quitar ambas auxiliares baja el F1 de 75.7 a 69.7 y la edge acc de 69.6 a 65.6 — [arXiv HTML](https://arxiv.org/html/2602.12080v2)
- Ball Radar: los modelos jerárquicos con poseedor intermedio superan a los no jerárquicos (PE 3.69 frente a 5.47 m en LSTM). La "reality loss" física reduce el RL y mejora ligeramente el PE y el PPA (64.0 → 64.7 %) — [arXiv PDF](https://arxiv.org/pdf/2306.08206)
- Ball Radar entrena con el balón enmascarado al 80 % en la mitad de los batches y al 100 % en el resto, de modo que un solo modelo sirve para cualquier grado de visibilidad — [arXiv PDF](https://arxiv.org/pdf/2306.08206)
- Peral et al.: pérdida conjunta ADE + CE de estado + CE de poseedor. El estado del balón tiene 4 clases (pase / posesión / incontrolado / fuera). En su dataset hay 82 961 frames de pase, 97 841 de posesión y 20 018 de incontrolado. Enmascarar por jugador y secuencia completa supera al enmascarado aleatorio por frame — [arXiv 2512.19528](https://arxiv.org/pdf/2512.19528)

### Inferences
- Para nuestro caso: entrenar el set encoder con el balón de IDSSE/Metrica enmascarado aleatoriamente por bloques largos (huecos con la mediana de 10.7 s del objetivo) y con la mitad de los batches sin balón alguno, más una cabeza auxiliar de poseedor por tracklet y una de estado (control / vuelo / disputa / muerto). Es la receta que mide ganancias en los tres papers. Lo hemos probado con features de toque explícitas, pero no con objetivos auxiliares por jugador.
- La clase "incontrolado" (~10 % de los frames en Peral et al.) se corresponde con nuestro problema: spells cortos en los que ambos equipos están igual de cerca. Modelarla como estado propio, en lugar de forzar H/A, puede hacer que la probabilidad del 0.45 se separe.

### Gaps
- No encontré trabajos sobre aprendizaje con etiquetas débiles o ruidosas específicamente para toques de balón en tracking. Tampoco ablaciones con balón suavizado o desenfocado (solo enmascarado).

## 5. Retardo entre pérdida/robo y reacción observable del equipo (relevante para ±50 s y Viterbi)

### Takeaway
La reacción colectiva documentada se concentra en 2-5 s (contrapresión en ≤ 5-6 s, features a 2-3 s) y el repliegue se mide hasta 10 s. Más de un 25 % de las transiciones dura < 3 s. Los modelos sin balón con mejor rendimiento usan ventanas de 9.6-10 s, y Ball Radar indica que el contexto más reciente es el que más pesa.

### Cited Findings
- Ventana de contrapresión: ~5 s (Bauer & Anzer), 6 s (otras definiciones) — [Springer](https://link.springer.com/article/10.1007/s10618-021-00763-7); [Soccerment](https://blog.soccerment.com/?p=27977)
- Más del 25 % de las fases de transición duran < 3 s, y Frencken propone ventanas de 3 s — [Zhang et al. 2025](https://journals.sagepub.com/doi/full/10.1177/17543371251392456)
- PathCRF entrena con ventanas de 10 s y Peral et al. con 9.6 s. Los fallos de Peral et al. se concentran en los bordes de la secuencia — [arXiv HTML](https://arxiv.org/html/2602.12080v2); [arXiv 2512.19528](https://arxiv.org/pdf/2512.19528)
- Ball Radar: las LSTM superan a los Transformers porque las últimas posiciones son el contexto más importante — [arXiv PDF](https://arxiv.org/pdf/2306.08206)
- La tolerancia de PathCRF de 1 s / 4 m da un recall de ~80 %, frente al 86 % con 2 s / 10 m. La imprecisión temporal de los eventos sin balón es del orden de 1-2 s — [arXiv HTML](https://arxiv.org/html/2602.12080v2)

### Inferences
- Una reacción colectiva a 2-5 s implica que una ventana de ±50 s sobra para fijar el lado de los spells largos, pero en los spells < 2 s la evidencia colectiva llega después de que el spell termina. Para esos spells, la señal tendría que leerse a +2-5 s del candidato (¿reaccionó el otro equipo?). Una feature "reacción diferida" (cambio del centroide, de la altura de línea o del número de jugadores retrocediendo entre t y t+3 s) es plausible y no está medida en la literatura.
- La resolución temporal de ~1-2 s de PathCRF sugiere que el techo para spells < 2 s sin balón limpio es bajo en todos los métodos publicados. No he encontrado ningún paper que reporte recall por duración de spell.

### Gaps
- No encontré una medición directa del retardo de la reacción colectiva (centroide, línea, orientación) tras la pérdida.
- No encontré estudios de orientación corporal estimada desde tracking (dirección de velocidad como proxy) aplicados a detectar toques. La orientación corporal real procede de vídeo, y nosotros no tenemos vídeo.
