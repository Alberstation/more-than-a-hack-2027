# Aprendizaje con observación parcial y cambio de dominio (etiquetado secuencial de posesión)

Contexto: etiquetador H/A/D por frame, balón ausente en el 41 % de frames evaluados (huecos largos), jugadores anónimos, entrenamiento sólo con datos externos degradados por simulación (7 IDSSE + 3 Metrica), HGB + DeepSets/TCN mezclados 0.25/0.75, Viterbi de 3 estados con coste de cambio, métrica spellF1 por IoU temporal. Caída entre proveedores ~0.1 spellF1; recall de posesiones < 2 s ~20 %.

Nota de método: unas 16 búsquedas y lecturas. La mayoría de las cifras vienen de abstracts o de resúmenes de búsqueda; las de Constraint-Aware Decoding (arXiv 2605.10149) se leyeron en el HTML del paper. Las cifras que no pude verificar en la fuente primaria van marcadas.

## 1. ¿Qué mejora de forma más fiable la robustez a entradas faltantes en clasificación o segmentación temporal?

### Takeaway
La evidencia más consistente apoya tres cosas baratas: dar al modelo la falta de datos como señal explícita (indicadores de máscara, tiempo desde la última observación) en lugar de imputar; usar árboles con manejo nativo de NaN; y entrenar con corrupciones del tipo de las de test (descartar frames, ocultar sensores) añadiendo una pérdida de consistencia. El preentrenamiento enmascarado ayuda sobre todo cuando hay pocas etiquetas. No encontré pruebas directas de que aporte robustez a huecos estructurados largos.

### Cited Findings
- **Indicadores de falta de datos en RNN.** Lipton, Kale y Wetzel (2016) dan a una RNN indicadores binarios de falta de datos como entrada. Según los autores, la RNN los aprovecha bien y mejora el AUC y el F1 de forma significativa frente a imputar. En varias enfermedades, el patrón de qué se midió predice más que los propios valores. No obtuve las tablas numéricas. — [arXiv 1606.04130](https://arxiv.org/abs/1606.04130v4)
- **GRU-D.** Che et al. (Sci. Rep. 2018) añaden a una GRU vectores de máscara y dos decaimientos aprendidos: de la entrada hacia la media empírica y del estado oculto, ambos en función del tiempo desde la última observación. Reportan el mejor AUC entre los métodos comparados en MIMIC-III y en PhysioNet 2012. Las cifras de terceros no coinciden (84.24 % frente a 86.3 / 0.818), así que trátense como no verificadas. — [arXiv 1606.01865](https://arxiv.org/pdf/1606.01865v1); [paperswithcode](https://paperswithcode.com/paper/recurrent-neural-networks-for-multivaria2)
- **Implementación de la máscara.** Un estudio de 2024 en PhysioNet 2012 encuentra que la forma de implementar el enmascaramiento cambia tanto la imputación como la predicción posterior. Las arquitecturas recurrentes son más estables entre estrategias. — [arXiv 2405.17508](https://arxiv.org/html/2405.17508v3)
- **Árboles con NaN nativo.** Perez-Lebel et al. (GigaScience 2022) compararon métodos en 4 bases de historia clínica electrónica, imagen cerebral, una encuesta y 2 bases de UCI. El soporte nativo de NaN en gradient boosting ("missing incorporated attribute") predice mejor que la imputación de estado del arte y cuesta mucho menos cómputo. Si se imputa, recomiendan añadir columnas indicadoras, porque los datos suelen ser no-MAR. — [arXiv 2202.10580](https://arxiv.org/abs/2202.10580)
- **Corrupción temporal en detección de acciones.** Zeng et al. (CVPR 2024) crearon THUMOS14-C y ActivityNet-C con 5 corrupciones temporales a 3 severidades. Los detectores caen mucho aunque se corrompa un solo frame, y los modelos end-to-end son más frágiles que los que usan un extractor preentrenado. La caída viene sobre todo de errores de localización, no de clasificación. Una corrupción a mitad de una acción es la que más daña. La defensa que proponen es un aumento FrameDrop más una pérdida de consistencia temporal entre la vista limpia y la corrupta, y mejora también en datos limpios. — [CVPR 2024 / arXiv 2403.20254](https://arxiv.org/abs/2403.20254)
- **Sensores llevables (wearables).** La encuesta de modalidades faltantes (2024) señala que los IMU y las señales fisiológicas pierden segmentos enteros. Esa falta estructurada no la captura el enmascaramiento aleatorio, y los benchmarks rara vez modelan fallos sistemáticos o asíncronos. — [arXiv 2409.07825](https://arxiv.org/pdf/2409.07825)
- **LSM-2 / AIM.** Su enmascaramiento adaptativo degrada menos al quitar el acelerómetro (F1 de 0.47 a 0.20) que el baseline LSM-1, que cae más. El dato viene del resumen de búsqueda. — [arXiv 2506.05321](https://arxiv.org/pdf/2506.05321)
- **Diversidad de sensores en entrenamiento.** En ActivityNarrated, entrenar con las 15 posiciones de sensor y evaluar en subconjuntos aguanta mejor que entrenar sólo con el subconjunto de test. — [arXiv 2604.00767](https://arxiv.org/pdf/2604.00767)
- **PrimeNet (AAAI 2023).** Preentrena con contraste temporal y reconstrucción, enmascarando una duración constante en lugar de un número fijo de puntos. Los márgenes más grandes aparecen en few-shot: AUC 1-shot en MIMIC-III de 0.595, frente a 0.534 de mTAND y 0.558 de TNC (cifra de un resumen secundario). No mide robustez ante falta de datos añadida. — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/25876)

### Inferences
- Nuestro HGB ya cubre la recomendación de "NaN nativo". Lo que más se repite en la literatura y quizá falte en el encoder son canales de "segundos desde la última observación del balón" y "duración del hueco actual", al estilo GRU-D o Lipton. Son baratos en numpy.
- Zeng et al. encaja con nuestro caso, porque el balón falta a mitad de las posesiones y el error dominante es de localización. Una pérdida de consistencia entre la versión limpia y la degradada del mismo clip es aplicable a la TCN en una RTX 4060.
- El preentrenamiento enmascarado tiene más sentido como reconstrucción de la posición del balón dentro del hueco (con la geometría de jugadores como contexto) que como SSL genérico. La evidencia publicada lo apoya en régimen de pocas etiquetas, no en huecos estructurados.

### Gaps
- No encontré un benchmark de segmentación temporal de acciones (no de detección) con oclusión o frames faltantes, ni cifras en PhysioNet de "máscara frente a imputación" del mismo paper con tablas.
- No hallé estudios de preentrenamiento enmascarado evaluados bajo falta de datos añadida y estructurada (huecos largos).

## 2. Aleatorizar la degradación en lugar de imitar un perfil: ¿hay evidencia de mejora sim-to-real?

### Takeaway
En visión y robótica, aleatorizar la degradación ayuda de forma clara, pero no todo factor ayuda, y una variación que no representa el despliegue puede empeorar. Mezclar severidades y cadenas de aumento sin ver las corrupciones de test (AugMix) reduce el error ante corrupciones nuevas a menos de la mitad. No encontré una comparación directa de ruido aleatorizado frente a ruido igualado al sensor real.

### Cited Findings
- **Ablación factor a factor.** En localización de cubos, la aleatorización de textura baja el error de 7.55 a 2.52 cm. Añadir aleatorización de iluminación no mejora. — [arXiv 2307.15320](https://arxiv.org/pdf/2307.15320)
- **Variación fuera del despliegue.** Una revisión advierte que la variación excesiva y no representativa del despliegue (sobre todo de pose de cámara) puede ser perjudicial, así que hay que vigilar el parecido con las condiciones reales. — [emergentmind: domain randomization](https://www.emergentmind.com/topics/domain-randomization)
- **Control.** En un coche que derrapa, quitar la perturbación dinámica mejora un poco en simulación (80 % frente a 75 %), pero sólo la configuración con aleatorización completa transfiere al mundo real. — [arXiv 2507.23339](https://arxiv.org/pdf/2507.23339)
- **Ruido correlacionado.** Un trabajo con UAV modela el ruido de percepción como temporalmente correlacionado (proceso con reversión a la media), porque los errores físicos rara vez son i.i.d. No hay ablación frente a ruido i.i.d. — [arXiv 2512.17349](https://arxiv.org/pdf/2512.17349)
- **Trabajo fundacional.** Tobin et al. (2017) incluyen una ablación de las opciones de aleatorización. — [arXiv 1703.06907](https://arxiv.org/pdf/1703.06907v1.pdf)
- **AugMix.** Mezcla convexa de varias cadenas de aumento, con operaciones disjuntas de las corrupciones de test. En CIFAR-10-C el error cae del 28.4 % al 12.4 %, y la mejora se mantiene en las 15 corrupciones y las 5 severidades (según resumen). — [arXiv 1912.02781](https://arxiv.org/pdf/1912.02781); [resumen liner](https://liner.com/review/augmix-a-simple-data-processing-method-to-improve-robustness-and)
- **Aumento y corrupción.** Otro trabajo estudia la interacción entre aumentos y corrupciones en robustez natural y encuentra que la similitud aumento-corrupción influye en la mejora. Esto viene del título y la búsqueda; no leí las cifras. — [arXiv 2102.11273](https://arxiv.org/pdf/2102.11273)

### Inferences
- Para nosotros, la versión AugMix sería muestrear por clip la tasa de huecos, la distribución de longitudes, el blur y el ruido desde rangos amplios que contengan el perfil estimado del partido (calculable sin etiquetas, sólo a partir de la máscara de presencia del balón). Así no se fija un único perfil. La advertencia de la revisión sugiere centrar el rango en el perfil observado y no irse a extremos irreales.
- Hacer ablaciones factor a factor (huecos, blur, ruido, balón muerto móvil) contra ctrl separa lo que aporta de lo que sobra, como en la ablación de textura frente a iluminación.
- Huecos y ruido conviene simularlos temporalmente correlacionados (en ráfagas o con reversión a la media), no i.i.d.

### Gaps
- No encontré estudios en series temporales o tracking que comparen "perfil igualado" frente a "rango aleatorizado" con cifras.

## 3. Generalización de dominio con sólo 2-3 fuentes: qué funciona y cuánto

### Takeaway
Con un protocolo de selección justo, casi ningún método de DG supera a un ERM bien ajustado por más de 1 punto (DomainBed). Lo que sí da mejoras modestas y repetibles es promediar pesos o ensamblar (SWAD +3.6 pp de media sobre ERM). En HAR entre datasets los métodos de mezcla de estilo (MixStyle, EFDM) y la adaptación multi-fuente existen, pero con cifras dispersas. Con 2-3 fuentes lo más defendible es: ERM con más fuentes, normalización por partido, aumento y ensemble o promedio de pesos.

### Cited Findings
- **DomainBed.** Gulrajani y Lopez-Paz (ICLR 2021) encuentran que, con la misma selección de modelo, ningún algoritmo de DomainBed supera la media de ERM por más de 1 punto. También sostienen que un método de DG sin estrategia de selección de modelo está incompleto. La media de ERM según versión es PACS ~85.5, VLCS ~77.5, OfficeHome ~66.5. — [arXiv 2007.01434](https://arxiv.org/abs/2007.01434)
- **SWAD.** Muestrea pesos en cada iteración y elige el inicio y el fin del promedio por pérdida de validación. Mejora la media de DG en +3.6 pp sobre ERM (63.3) y +1.6 pp sobre los mejores métodos previos en PACS, VLCS, OfficeHome, TerraInc y DomainNet. Se puede combinar con otros métodos de DG. — [arXiv 2102.08604](https://arxiv.org/abs/2102.08604)
- **CoDATS y CALDA.** Wilson et al. (KDD 2020) proponen adaptación multi-fuente para series temporales de sensores (CoDATS, adversarial) y CALDA (contrastiva adversarial). Requieren datos del dominio objetivo sin etiquetar, lo que es adaptación, no DG pura. — [alphaxiv 2005.10996](https://alphaxiv.org/abs/2005.10996); [tesis WSU](https://rex.libraries.wsu.edu/esploro/outputs/doctoral/Multi-Source-Unsupervised-Domain-Adaptation-for-Time-Series/99900651794201842)
- **Mezcla de estilo en HAR.** Un paper de ESANN 2025 estudia MixStyle y EFDM (mezclar estadísticas de características entre dominios) para HAR con cambios de colocación de sensor y de dispositivo. — [ESANN 2025](https://esann.org/sites/default/files/proceedings/2025/ES2025-135.pdf)
- **Ensembles en HAR multi-fuente.** El paper MUCT sostiene que la adaptación sin un paradigma de ensemble adecuado generaliza mal en multi-fuente. — [Springer 2024](https://link.springer.com/10.1007/s40747-023-01218-w)

### Inferences
- La recuperación que medimos al añadir Metrica (+0.01-0.02) está en la escala de "más de un punto" que DomainBed asigna a casi cualquier método de DG. Coherente con eso, añadir fuentes y diversidad al entrenamiento es lo más rentable.
- Opciones baratas y deterministas en inferencia:
  - Promediar pesos (SWA/SWAD) del encoder: un único modelo, numpy, coste cero en inferencia.
  - Ensemble de semillas o de modelos con fuente excluida (leave-one-provider-out), promediando probabilidades antes de Viterbi.
  - Normalizar las características por partido: estadísticos del propio input de test (medias y escalas de posiciones o velocidades). Es una transformación fija sin parámetros aprendidos. Hay que confirmar si las reglas lo aceptan; ver la pregunta 5.
  - MixStyle sobre embeddings de jugador entre partidos de distinto proveedor.
- Con sólo 2 proveedores, validar con leave-one-provider-out es la única forma honesta de elegir hiperparámetros de DG, en línea con el aviso de DomainBed sobre la selección.

### Gaps
- No encontré cifras de AdaBN ni de normalización por fuente en series de sensores con pocas fuentes, ni resultados de Ensemble of Averages (EoA).
- No hay datos de DG en tracking deportivo entre proveedores.

## 4. Decodificación para F1 a nivel de segmento: ¿se recuperan segmentos cortos?

### Takeaway
Las priors estructurales en Viterbi (duración por clase, transiciones, fronteras) suben el F1 segmental entre 1 y 10 puntos sin reentrenar. Constraint-Aware Decoding da +4.7 F1@10 con ASFormer en Breakfast. Están pensadas contra la sobresegmentación y a veces bajan la exactitud por frame o el F1@50. No encontré evidencia de que recuperen segmentos cortos cuando falta evidencia. Para eso, la vía teórica es decodificar maximizando el F1 esperado o el IoU esperado, y no el camino MAP.

### Cited Findings
- **Constraint-Aware Decoding** (Ee, Roy, Li, Zhang, Fernando; arXiv 2605.10149). Es un Viterbi modificado con confianza de transición, conjuntos de inicio y fin, y duración por clase normalizada a la longitud del vídeo, obtenidos de datos anotados. Sólo se permite la autotransición mientras la duración esté dentro de los límites. No requiere reentrenar. — [arXiv HTML 2605.10149](https://arxiv.org/html/2605.10149)
  - ASFormer, Breakfast: F1@10/25/50 de 74.1/68.7/55.5 a 78.8/73.1/58.0, y Edit de 72.8 a 77.9.
  - ASFormer, 50Salads: F1@10 de 83.4 a 84.8 y F1@50 de 74.6 a 76.0, pero la exactitud por frame baja de 85.0 a 82.9.
  - Semi-supervisado 5 %, 50Salads: F1@10 de 51.8 a 55.8. Frente a un Viterbi clásico, el constreñido sube F1@10 de 45.5 a 55.8.
  - Ablación en Breakfast: quitar la duración es lo que más baja F1@50 (58.0 a 54.9), y quitar transiciones es lo que más baja Edit (77.9 a 73.8).
  - Restricciones suaves rinden peor que duras: 75.6 frente a 78.8 en F1@10.
  - Con el backbone FACT apenas aporta: F1@10 de 81.3 a 81.8, y F1@50 baja de 66.2 a 64.8.
  - Tiempo: 2.81 s frente a 148.9 s del método KARI.
  - No reporta efectos sobre segmentos cortos ni casos de fusión excesiva.
- **Pérdidas de suavizado.** ASRF (Ishikawa et al., WACV 2021) combina una pérdida de suavizado de probabilidades con clasificación ponderada por clase y regresión de fronteras. Reporta +8.2 % de F1 segmental y +5.9 % de Edit, y hasta +16.1 % de F1 frente al estado del arte previo. Las pérdidas de suavizado tipo MS-TCN (MSE truncada sobre log-probabilidades consecutivas) penalizan el parpadeo; la cifra de ~5 puntos es de un resumen secundario. — [WACV 2021](https://openaccess.thecvf.com/content/WACV2021/papers/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.pdf); [arXiv 1903.01945](https://alphaxiv.org/abs/1903.01945)
- **Definición del F1@k** (Lea et al., TCN 2017). Un segmento predicho es verdadero positivo si su IoU con el segmento real supera τ = k/100. El F1@k penaliza la sobresegmentación y no los desplazamientos temporales pequeños. — [arXiv 1611.05267](https://arxiv.org/pdf/1611.05267)
- **HSMM y semi-CRF.** El Viterbi segmental (Yu 2015) puntúa segmentos candidatos y el término de duración penaliza segmentos implausiblemente cortos, lo que suaviza. El coste crece con la longitud máxima L y se mitiga podando. Shi et al. (IJCV 2011) argumentan que las cadenas de Markov no capturan la duración del segmento. — [arXiv 2606.17939](https://arxiv.org/pdf/2606.17939); [arXiv 2209.09149](https://arxiv.org/pdf/2209.09149); [Shi et al. IJCV 2011](https://www.ece.ualberta.ca/~lcheng5/papers/ShiEtAl_IJCV11.pdf)
- **Decisiones óptimas para F1 e IoU.** Si las probabilidades están calibradas, el umbral que maximiza F1 es la mitad del F1 óptimo alcanzable (p ≥ F/2), por debajo de 0.5 (Lipton et al.). Nowozin (2014) deriva decisiones que maximizan el IoU esperado bajo un posterior en lugar del MAP. Jansche (ACL 2007) decodifica secuencias maximizando el F esperado, calculado de forma exacta. — [arXiv 1402.1892](https://arxiv.org/pdf/1402.1892); [Nowozin 2014](https://microsoft.com:443/en-us/research/wp-content/uploads/2016/02/nowozin2014intersectionoverunion.pdf); [Jansche ACL 2007](https://preview.aclanthology.org/moar-dois/P07-1093.pdf)

### Inferences
- Nuestro problema es el contrario del que resuelven los trabajos de TAS: subsegmentamos por falta de evidencia, no sobresegmentamos. Una prior de duración mínima (HSMM o duraciones acotadas) tiende a eliminar aún más posesiones cortas. El lado útil de la prior es la duración máxima: en un hueco largo, una cola de duración sugiere que probablemente hubo un cambio dentro.
- Como spellF1 cuenta cada posesión por igual, el argumento de Lipton (umbral F/2 < 0.5) y el de Jansche/Nowozin (maximizar el F o el IoU esperado) indican que emitir más posesiones cortas cuando el posterior las hace plausibles puede subir el F1 aunque baje la exactitud por frame.
- Implementación posible en numpy: forward-backward sobre el HMM actual para obtener posteriores de cambio, y después insertar segmentos donde la ganancia esperada de un verdadero positivo supere el coste esperado de un falso positivo. Equivale a bajar el coste de cambio de forma localizada dentro de los huecos. Es determinista y CPU.
- Que las restricciones duras rindan mejor que las suaves en 2605.10149 sugiere probar límites de duración por estado como restricción (p. ej. D máximo) además del coste de cambio.
- Lo que hay que medir contra ctrl es el número de posesiones emitidas frente a las reales (sesgo de recuento) y el spellF1 por rango de duración.

### Gaps
- No encontré trabajos que muestren recuperación cuantificada de segmentos cortos bajo evidencia faltante, ni aplicaciones de decodificación de F1 esperado a segmentación temporal con IoU.
- 2605.10149 no explica cómo calcula los límites de duración (no dice si usa percentiles) y no reporta GTEA.

## 5. ¿Cuenta la adaptación no supervisada en test (TTA) como "entrenar con el test"?

### Takeaway
Depende de las reglas de cada competición. Al menos un reto de adaptación de dominio (VisDA 2020) prohibió explícitamente el uso transductivo del test sin etiquetas, y la literatura define la TTA como la contraparte transductiva de la adaptación sin fuente, es decir, que usa muestras del objetivo. Con reglas que exigen entrenar y ajustar sólo con datos externos, lo prudente es tratar cualquier actualización de parámetros con el input de test como prohibida.

### Cited Findings
- **VisDA 2020.** Ante la pregunta de si se podían usar los datos de test sin etiquetas de forma transductiva, la organización respondió que no: sólo el conjunto de entrenamiento objetivo, "DO NOT use the validation/test set for training". También anunció que revisaría el código y pediría un informe para replicar resultados, de modo que el ganador podía no ser el primero del leaderboard. — [foro VisDA 2020](https://competitions.codalab.org/forums/21360/4142)
- **Definición de TTA.** La encuesta de TTA la describe como la contraparte transductiva de la adaptación sin fuente: usa muestras de la distribución objetivo y evalúa sobre los mismos datos de test. — [arXiv 2303.15361](https://arxiv.org/html/2303.15361v2)
- **Fragilidad de la TTA.** Hay trabajos dedicados a sus trampas: inestabilidad, dependencia de hiperparámetros elegidos con etiquetas del objetivo, y flujos ruidosos. — [On Pitfalls of TTA, arXiv 2306.03536](https://ar5iv.labs.arxiv.org/html/2306.03536); [SoTTA arXiv 2310.10074](https://ar5iv.labs.arxiv.org/html/2310.10074)

### Inferences
- Hay una frontera razonable entre dos cosas. Una transformación fija, sin parámetros aprendidos y aplicada por igual a cualquier partido (p. ej. normalizar por estadísticos del propio input o estimar la tasa de huecos para elegir un modelo de un banco precalculado) es inferencia. Cualquier actualización de pesos o elección de hiperparámetros mirando el partido de test es TTA. Aun así, elegir con el perfil del partido roza el ajuste, así que conviene documentarlo o evitarlo.
- La TTA además rompe la exigencia de inferencia determinista y la reproducibilidad que piden los organizadores, como en la verificación de código de VisDA.

### Gaps
- No encontré reglas publicadas de otras competiciones de TTA, ni de retos deportivos de tracking, sobre normalización por estadísticos del test.
