# Fuentes complementarias para rellenar huecos de tracking (eventos, vídeo, audio) en posesión por frame

Contexto: etiquetas H/A/D por frame, ATD con balón ausente en el 41% de frames evaluados, Smart Tagging disperso (balón parado, saques), vídeo táctico, inferencia CPU determinista, sin entrenar con este partido. Ya medido internamente: ventanas Smart Tagging en el decodificador = sin ganancia; balón recuperado por vídeo simulado (WASB) = +0.026 spellF1.

## (a) Sincronización evento-tracking: métodos y si unas pocas etiquetas pueden anclar la posesión

### Takeaway
Los sincronizadores modernos (ELASTIC 2025/2026, Biermann 2023, databallpy) son muy precisos, pero todos presuponen un feed de eventos denso (cada pase con jugador y tipo) y balón en el tracking; con Smart Tagging limitado a balón parado sirven para anclar reinicios (inicio/fin de D y equipo del saque), no para decidir posesiones cortas en juego. Lo más aprovechable para nuestro hueco es PathCRF (2026, mismo grupo que ELASTIC): posesión sin balón a partir solo de jugadores, 92.4% de acierto de equipo por frame.

### Cited Findings
- ELASTIC (arXiv 2508.09238, MLSA@ECML PKDD 2025, KAIST + datos Ajax) sincroniza solo con rasgos del tracking, sin ubicaciones de evento anotadas; 4 etapas: alineación de kick-off, eventos mayores (pases, tiros, intercepciones, balón parado), detección de recepción, eventos menores — [arXiv HTML](https://arxiv.org/html/2508.09238v1)
- Entradas que ELASTIC exige: eventos con timestamp, jugador, tipo y resultado; tracking 10/25 fps con jugadores (x,y), balón (x,y,z) y flag de juego vivo/muerto. No describe cómo trata balón ausente — [arXiv HTML](https://arxiv.org/html/2508.09238v1)
- Resultados ELASTIC (2.134 eventos, 3 partidos Eredivisie): inicio de evento 88.4% exacto y 95.0% a ≤25 frames (ETSY: 21.1% / 74.7%); pases 95.5% exacto, error medio 0.567 frames (ETSY 14.05); fin de evento 92.6% exacto; error medio global 2.35 frames (~0.09 s) frente a 15.15 de ETSY. Sobre Biermann et al. solo dice que "<10% de los pases se alinean exactamente" — [arXiv HTML](https://arxiv.org/html/2508.09238v1)
- Código ELASTIC: github.com/hyunsungkim-ds/elastic con tutorial sobre Sportec Open Dataset; la licencia del código no figura en el paper — [arXiv HTML](https://arxiv.org/html/2508.09238v1)
- Versión extendida (arXiv 2608.30227, aceptada en CIKM 2026 según un autor): extrae frames candidatos donde un toque es plausible y alinea con Needleman-Wunsch extendido (orden preservado); publica benchmark con timestamps reales sobre Sportec Open DFL — [arXiv PDF](https://arxiv.org/pdf/2608.30227) (cifras de esta versión no verificadas)
- Biermann et al. 2023 (Scientific Reports): SwiftEvent + refinamiento estadístico de pases; pases a ≤0.5 s del ground truth pasan de 14% a 70% en 4 partidos 2014/15; comparable a un detector deep learning con mucha menos data — [PMC10518005](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10518005/)
- Estudio de detección de pases/tiros por reglas: F1 de pases 0.71 cuando evento y posición son independientes vs hasta 0.92 cuando son dependientes; sostiene que las cifras de la literatura están sobrestimadas — [PMC11025972](https://pmc.ncbi.nlm.nih.gov/articles/PMC11025972)
- databallpy: `match.synchronise_tracking_and_event_data()` con Needleman-Wunsch (preserva orden); modo "smart" por defecto; proveedores de eventos Opta/Metrica/Instat y tracking Tracab/Metrica/Inmotio; versión 0.7.3 en PyPI — [PyPI](https://pypi.org/project/databallpy), [GitHub](https://github.com/Alek050/databallpy)
- PathCRF (arXiv 2602.12080): grafo dinámico 22 jugadores + 4 nodos "fuera", CRF enmascarado + Viterbi que elige un "edge" de posesión por paso; solo tracking de jugadores, sin balón. Test (Sportec Open DFL, 5/1/1 partidos): edge 69.64%, emisor 80.44%, receptor 82.28%, **posesión de equipo por frame 92.40%**, % posesión local 62.64 vs 62.13 real; F1 de eventos 75.69% a 5 fps y 77.21% a 25 fps (tolerancia 1 s, mismo tipo y jugador). Inferencia de un partido de 90' ≈23 s (5 fps) / 49 s (25 fps) en RTX 4090. Código github.com/hyunsungkim-ds/pathcrf (licencia no indicada) — [arXiv HTML](https://arxiv.org/html/2602.12080)
- Detección de eventos por cambios de posesión en tracking (arXiv 2202.00804): ">90% de tasa de detección" en la mayoría de categorías (pases, tiros, centros, paradas, intercepciones) — [fugumt resumen de arXiv 2202.00804](https://fugumt.com/fugumt/paper_check/2202.00804v1_enmode)

### Inferences
- Smart Tagging no es un feed de eventos al estilo Opta: sin pases ni jugador ejecutor, ELASTIC/databallpy no tienen qué alinear en juego abierto. Lo coherente con nuestro resultado nulo es usar las etiquetas solo como anclas duras de reinicio (frame de inicio del saque y equipo que saca) y dejar el juego abierto al decodificador.
- PathCRF ataca exactamente nuestro problema (balón ausente) y su backbone se entrena con datos externos licenciables (Sportec Open DFL); usarlo como emisor de probabilidades H/A combinables con nuestro decodificador cuando falta balón es la vía de mayor valor esperado de esta sección. Su 92.4% es en datos con tracking de calidad DFL y jugadores identificados; nuestro ATD es anónimo, así que habría que adaptar la entrada a equipo + posición sin identidad (el edge jugador-jugador se reduce a equipo).
- Lógica de "frames candidatos de toque" de ELASTIC-extendido (aceleración/cambio de dirección del balón cerca de un jugador) es reutilizable sin eventos para proponer toques en los tramos con balón presente.

### Gaps
- Licencia del código de ELASTIC y PathCRF no encontrada (comprobar LICENSE del repo). Licencia del Sportec Open DFL Dataset no confirmada en estas fuentes.
- Ningún trabajo encontrado que sincronice etiquetas tan dispersas como Smart Tagging ni que mida ganancia de posesión por frame al anclar solo balón parado.
- No se encontró cómo maneja ELASTIC tramos sin balón.

## (b) Modelos de vídeo preentrenados y abiertos: balón, toques/acciones y equipo por color

### Takeaway
Hay modelos abiertos con licencia usable: WASB (MIT, balón), ganador BAS 2023 de lRomul (MIT, pase/conducción, mAP@1 ≈86.5) y T-DEED (GPL-3.0, 12 clases, ganador BAS 2024 con 73.39 mAP@1); en 2025 la tarea pasó a ser con equipo (Team-mAP@1 60.03). Todos están entrenados en vídeo broadcast de SoccerNet, no en cámara táctica gran angular, y la tolerancia es 1 s, más gruesa que nuestras posesiones <2 s. Para ~199k frames, lo realista es GPU 4060 como preproceso cacheado, no CPU.

### Cited Findings
**Balón**
- WASB-SBDT (BMVC 2023, NTT): licencia MIT; repo con evaluación de modelos preentrenados (MODEL_ZOO.md); probado con Python 3.8 / CUDA 11.3 — [GitHub](https://github.com/nttcom/WASB-SBDT)
- WASB en fútbol (dataset D'Orazio et al. 2009, 1920×1080 a 25 fps, cámaras fijas multi-vista; reanotado por los autores): F1 88.2 / AP 86.2 a τ=4 px, 23.6 FPS (step=1) o 55.7 FPS (step=3); TrackNetV2 F1 86.6, 66 FPS; DeepBall F1 44.5. Entrada redimensionada a 288×512; servidor con V100 — [arXiv HTML](https://arxiv.org/html/2311.05237v2)

**Acciones / toques (SoccerNet Ball Action Spotting)**
- BAS 2023 (2 clases: pase y conducción): ganador Ruslan Baikulov, repo lRomul/ball-action-spotting, **licencia MIT**, mAP@1 ≈86.47 (challenge) / 87.03 (test), tolerancia 1 s; entrada gris 1280×736 a 12.5 fps (salta 1 de cada 2 frames de 25 fps), ventana 33 frames; EfficientNetV2-B0 2D + 4 bloques 3D; pesos en Google Drive; requiere CUDA ≥11.8; caché de features "acelera varias veces" la predicción — [GitHub](https://github.com/lRomul/ball-action-spotting)
- BAS 2024: 12 clases (Pass, Drive, Header, High Pass, Out, Cross, Throw In, Shot, Ball Player Block, Player Successful Tackle, Free Kick, Goal), 77 partidos de ligas inglesas en 720p + 22 de challenge; ganador T-DEED 73.39 mAP@1 (ensemble de 2 modelos, entrenado con SoccerNet Ball + SoccerNet original); 2º 71.35; baseline 56.15 — [SoccerNet 2024 results](https://arxiv.org/html/2409.10587v1)
- T-DEED: código **GPL-3.0** (más un fichero "LICENSE E2E-Spot"), checkpoints en Google Drive, ejemplo de inferencia a 796×448 y la resolución debe coincidir con la de entrenamiento; sin datos de velocidad — [GitHub](https://github.com/arturxe2/T-DEED); paper bajo CC BY-NC-SA 4.0 — [arXiv](https://arxiv.org/html/2404.05392v2)
- Team BAS 2025: mismas 12 clases con equipo (izquierda/derecha de cámara), métrica Team-mAP@1 con δ=1 s; ganador 60.03 (re-implementación de T-DEED en framework dude.k, cabeza única equipo×acción, pre-entreno en 500+ partidos), 2º 56.78, baseline T-DEED 51.72; un equipo aplica post-proceso que alinea etiquetas de equipo en ventanas de 4 s — [SoccerNet 2025 results](https://arxiv.org/html/2508.19182v1)
- FOOTPASS (arXiv 2511.16183): 54 partidos 2023/24, 81 h de broadcast, 102.992 eventos on-ball con jugador por dorsal; baselines TAAD+GNN y DST (recall ~33% cuando el ejecutor está fuera de plano vs ≤4%); anotaciones y código en GitHub/HuggingFace, frames bajo NDA de SoccerNet — [arXiv](https://arxiv.org/pdf/2511.16183)
- Peral et al. (VISAPP 2025): pases/recepciones desde vídeo siguiendo quién tiene el balón, mayoría de eventos con precisión de 0.6 s — [Scitepress](https://scitepress.org/publishedPapers/2025/133177/pdf/index.html)

**Equipo por color sin entrenar con el partido**
- roboflow/sports: recortes de jugador → embeddings SigLIP → UMAP → K-means a 2 clusters (no supervisado, por partido); porteros por proximidad; basado en Supervision (MIT) — [Roboflow blog](https://blog.roboflow.com/identify-basketball-players/), [roboflow/sports soccer](https://github.meowingcats01.workers.dev/roboflow/sports/tree/main/examples/soccer)
- SoccerNet GSR 2025 (posición, rol, dorsal y lado de equipo desde broadcast): mejor GS-HOTA 63.90 vs baseline 29.01 — [SoccerNet 2025 results](https://arxiv.org/html/2508.19182v1)

### Inferences
- Combinar spotter + tracking para decidir equipo: no encontré paper que lo haga con tracking táctico; el mecanismo natural es tomar el timestamp de "pass/drive" (±1 s) y asignar el toque al jugador del ATD más cercano al balón (o a la proyección del balón WASB vía homografía) en ese instante. El equipo ya viene en el ATD, así que no hace falta clasificar camisetas salvo para validar. Team-BAS 2025 da equipo directamente, pero como "izquierda/derecha de cámara" y a 60 Team-mAP en broadcast.
- Tolerancia 1 s vs posesiones <2 s: un spotter con error ~0.5–1 s puede confundir el toque que abre y el que cierra una posesión corta; su aporte más plausible es la *existencia* de toques (número de cambios), no su frame exacto. Peral et al. con 0.6 s es el dato más fino encontrado.
- Dominio: los pesos BAS/T-DEED se entrenaron en broadcast 720p de ligas inglesas; en cámara táctica el balón ocupa pocos píxeles. WASB se entrenó en cámaras fijas de campo, más cercano a nuestro caso. Es la razón de que la simulación +0.026 spellF1 se haya hecho con WASB.
- Licencias: MIT (WASB, lRomul) es limpio. GPL-3.0 (T-DEED) permite uso; obliga solo si se distribuye el derivado. Los pesos BAS se entrenaron con vídeo SoccerNet bajo NDA: la licencia de esos pesos es una zona gris a documentar. YOLO de Ultralytics es AGPL-3.0 (conocido, no verificado en esta búsqueda).
- Coste estimado (inferencia propia, sin fuente de benchmark CPU): WASB 288×512 a ~24–56 FPS en V100 ⇒ 199k frames ≈ 1–2.5 h en GPU; en una 4060 orden similar. En CPU un orden de magnitud más lento (decenas de horas), inviable dentro de la regla de "inferencia CPU" salvo que el preproceso GPU se acepte como caché. El spotter lRomul a 12.5 fps procesa ~100k frames; con caché de features, horas en GPU.
- Determinismo: cuDNN no es determinista por defecto; fijar `torch.use_deterministic_algorithms(True)`, `cudnn.benchmark=False`, semillas, decodificación de vídeo con la misma build de ffmpeg y guardar las salidas (CSV/NPZ con hash) para que la inferencia oficial lea solo la caché en CPU.
- Riesgo 7 días / RTX 4060 (8 GB): WASB (instalar, homografía, proyectar, fusionar) es 1–2 días de trabajo y ya tiene ganancia simulada; spotter de toques 2–3 días con incertidumbre de dominio alta; equipo por color innecesario (el ATD ya da equipo). PathCRF (sección a) es CPU-ligero y no requiere vídeo.

### Gaps
- Sin benchmarks de velocidad en CPU para WASB, lRomul o T-DEED.
- Sin evaluación publicada de spotters BAS en cámara táctica gran angular.
- Licencia exacta de los pesos (Google Drive) de lRomul y T-DEED, y del código roboflow/sports, no confirmada.
- Ningún trabajo encontrado que mida ganancia en posesión por frame al fusionar spotter de vídeo con tracking.

## (c) Audio: golpeo de balón y silbato para toques y balón muerto

### Takeaway
La evidencia útil es escasa: el audio mejora el spotting de acciones broadcast en ~4 puntos de mAP, pero no hay modelos abiertos ni datasets de silbato de árbitro en fútbol profesional; los recursos existentes son de RoboCup. Depende además de que el vídeo táctico traiga pista de audio con sonido de campo.

### Cited Findings
- Vanderplaetse & Dupont (CVPRW 2020): añadir audio a arquitecturas solo-vídeo en SoccerNet (500 partidos) da +4.19 puntos absolutos de mAP en spotting y +7.43 en clasificación — [arXiv 2011.04258](https://arxiv.org/abs/2011.04258), [CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w53/Vanderplaetse_Improved_Soccer_Action_Spotting_Using_Both_Audio_and_Video_Streams_CVPRW_2020_paper.html)
- Silbato: dataset RoboCup con 60.000 muestras sin silbato y 6.000 con silbato (10:1); 2024 usa CNN continuas para detectar silbato en NAO — [arXiv 2411.17347](https://arxiv.org/pdf/2411.17347); bit-bots/whistle_detection usa ResNet sobre un dataset SPL de 10 grabaciones, "en desarrollo intenso" — [GitHub](https://github.com/bit-bots/whistle_detection)
- En broadcast, el silbato es muy informativo, pero se confunde con vuvuzelas, bocinas y pitos de la grada; los anotadores necesitaron el vídeo para distinguirlo — [arXiv 1411.6496](https://arxiv.org/pdf/1411.6496) (anterior a 2020, contexto)
- Detección de golpeo como onset impulsivo (SuperFlux de librosa) descrita en patente para béisbol; otra patente corrige el retardo del sonido con el tracking (el golpeo crea el sonido en el punto del jugador) — [US7324984](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7324984), [US11303997](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11303997)
- Dataset multimodal de deportes de invasión (vídeo, audio y posiciones sincronizados, fútbol y balonmano) — [arXiv 2108.11149](https://ar5iv.labs.arxiv.org/html/2108.11149)
- Pádel (CVsports 2024): detección de golpeos multimodal audio+vídeo — [CVF](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Decorte_Multi-Modal_Hit_Detection_and_Positional_Analysis_in_Padel_Competitions_CVPRW_2024_paper.pdf)

### Inferences
- Un detector de onset sin entrenamiento (librosa, determinista en CPU, minutos para 2.2 h de audio) no viola las reglas y podría marcar candidatos de silbato (banda estrecha 2–4 kHz sostenida) y de golpeo. A distancia de cámara táctica el golpeo suele quedar por debajo del ruido de grada; el silbato es la señal más prometedora para el inicio/fin de D.
- El retardo acústico (~0.15 s a 50 m) es menor que la tolerancia útil para D, pero hay que corregirlo si se usa para toques.

### Gaps
- No hay modelo abierto preentrenado de silbato de árbitro para fútbol profesional; ni datos de precisión de golpeo de balón en estadio.
- Desconozco si el vídeo táctico del reto trae pista de audio (comprobar con ffprobe).

## (d) Ganancias cuantificadas en posesión/eventos

### Takeaway
Las cifras publicadas son por tarea, no de fusión con nuestro decodificador: sincronizar con tracking sube pases a ≤0.5 s de 14% a 70% (Biermann) o a 95.5% exactos (ELASTIC); posesión sin balón alcanza 92.4% de equipo por frame (PathCRF); audio aporta +4.19 mAP en spotting. No encontré ninguna cifra de ganancia de posesión por frame al añadir vídeo o audio sobre tracking.

### Cited Findings
- Biermann 2023: 14% → 70% de pases a ≤0.5 s — [PMC10518005](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10518005/)
- ELASTIC: 95.5% de pases exactos vs 21.9% de ETSY — [arXiv HTML](https://arxiv.org/html/2508.09238v1)
- PathCRF: 92.40% posesión de equipo por frame, F1 de eventos 75.69–77.21% a 1 s, sin balón — [arXiv HTML](https://arxiv.org/html/2602.12080)
- Audio sobre vídeo: +4.19 mAP spotting, +7.43 clasificación — [arXiv 2011.04258](https://arxiv.org/abs/2011.04258)
- Spotting de toques desde vídeo: 86.5 mAP@1 (2 clases, 2023), 73.39 (12 clases, 2024), 60.03 Team-mAP@1 (con equipo, 2025), todo a 1 s en broadcast — [lRomul](https://github.com/lRomul/ball-action-spotting), [2024](https://arxiv.org/html/2409.10587v1), [2025](https://arxiv.org/html/2508.19182v1)
- Pases desde tracking independiente: F1 0.71 (vs 0.92 con fuentes dependientes) — [PMC11025972](https://pmc.ncbi.nlm.nih.gov/articles/PMC11025972)

### Inferences
- Referencia propia: balón recuperado por vídeo (WASB simulado) +0.026 spellF1. Orden de prioridad para 7 días por coste/beneficio esperado: (1) WASB real en GPU con caché determinista; (2) prior de posesión sin balón tipo PathCRF entrenado con Sportec/Metrica para los huecos; (3) silbato por onset como pista de D; (4) spotter de toques solo si sobra tiempo, por el desajuste de dominio y la tolerancia de 1 s.

### Gaps
- Sin estudios que midan la mejora del recall de posesiones <2 s al añadir cualquiera de estas fuentes.
