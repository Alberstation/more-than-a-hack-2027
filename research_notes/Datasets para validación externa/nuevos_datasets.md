# Nuevos datasets públicos de tracking de fútbol (2025-2026) con señal de posesión, para validar el modelo congelado (local / visitante / balón muerto)

Fecha de consulta: 2026-11-13. [medido] = comprobado con la API de GitHub, Hugging Face, Zenodo, Figshare o arXiv en esta sesión.

## 1. ¿Publican datos (o código con datos) los papers de posesión / eventos / inferencia del balón desde tracking?

### Conclusión
Ningún paper de 2025-2026 publica partidos nuevos con tracking real. Los que sí comparten datos se basan en los 7 partidos IDSSE/DFL que ya teníamos, o en Metrica. Lo único nuevo son productos derivados de esos mismos partidos: la posesión por frame de PathCRF y el ground truth de eventos de ELASTIC. Pueden servir como etiqueta de referencia independiente sobre IDSSE, pero no amplían la validación a otros partidos.

### Hallazgos citados
- **PathCRF** (Kim et al., KDD 2026, arXiv 2602.12080). El repo `hyunsungkim-ds/pathcrf` [medido]: licencia MPL-2.0, 187 MB, último push 2026-06-08. Usa solo "Sportec Open DFL Dataset (Bassek et al., 2025)", es decir IDSSE, y no trae datos nuevos. Incluye los eventos sincronizados con ELASTIC en `data/sportec_synced` y un script (`datatools/preprocess.py`) que genera "the tracking data combined with per-frame ground-truth possession states". También trae modelos ya entrenados (`saved/100`–`saved/140`). — [README PathCRF](https://github.com/hyunsungkim-ds/pathcrf); [arXiv](https://arxiv.org/abs/2602.12080)
  - Uso para nosotros: una segunda etiqueta de posesión por frame sobre IDSSE, independiente de la nuestra, para contrastar. Además, el modelo de PathCRF sirve como baseline "sin balón". Acceso abierto e inmediato.
- **ELASTIC** (Kim et al., arXiv 2508.09238; versión CIKM 2026 en arXiv 2608.30227). El repo `hyunsungkim-ds/elastic` [medido]: licencia MPL-2.0, 22,6 MB, último push 2026-09-07. Usa 3 partidos de Sportec/IDSSE (J03WMX, J03WN1, J03WPY). En `benchmark/gt/` incluye eventos con `frame_id` y `receive_frame_id` anotados a mano: la mediana de 3 anotadores. Esos datos de `benchmark/` tienen licencia CC BY 4.0 y se describen como "derived from" Sportec. — [repo ELASTIC](https://github.com/hyunsungkim-ds/elastic); [arXiv 2608.30227](https://arxiv.org/abs/2608.30227)
  - El abstract de 2508.09238 cita además "2,134 events from three Eredivisie matches" con timestamps anotados. Esos datos de Eredivisie no se publican: el repo actual solo menciona Sportec. — [arXiv 2508.09238](https://arxiv.org/abs/2508.09238)
- **Ball Radar** (Kim et al., KDD 2023). El repo se movió de `pientist/ballradar` a `hyunsungkim-ds/ballradar` [medido: redirección 301]. No tiene licencia declarada [medido] y ocupa 167 MB. Sus datos son 15 partidos GPS de Fitogether ("we cannot share the Fitogether data") y los 3 partidos de Metrica, preprocesados y alojados en Google Drive. — [README Ball Radar](https://github.com/hyunsungkim-ds/ballradar)
- **TranSPORTmer** (Capellera et al., ACCV 2024, arXiv 2410.17785). Usa "real soccer match data from LaLiga's 2022-2023 season, including 283 matches" a 6,25 Hz, con "one state label per frame" (pase, posesión, incontrolado, fuera de juego). No publica ni código ni datos: la única URL del paper es un vídeo. — [arXiv html](https://arxiv.org/html/2410.17785v2)
- **Trabajos posteriores de Capellera** (Unified Uncertainty-Aware Diffusion 2503.18589, JointDiff 2509.22522 en ICLR 2026, Heteroscedastic Diffusion 2605.10717). Usan benchmarks llamados "Soccer-U" y "Football-U". JointDiff dice "We also share a new unified sports benchmark". El repo `guillem-cf/JointDiff` [medido] pesa 18 MB, no tiene licencia y su último push es 2026-04-13. — [arXiv API, autor Capellera](https://export.arxiv.org/api/query?search_query=au:Capellera); [repo JointDiff](https://github.com/guillem-cf/JointDiff)
- **Borghesi et al.** (ESWA 223, 2023, 119780). Clasifica cada frame en juego inactivo, balón del local o balón del visitante: exactamente nuestras etiquetas. La tesis asociada usa "tracking data collected during the 2019-20 season of the Italian Serie A". No encontré datos publicados. — [PoliTO IRIS](https://iris.polito.it/handle/11583/2976470); [tesis](https://webthesis.biblio.polito.it/18073/)
- **Seguimientos de Bassek 2025 (IDSSE)**:
  - arXiv 2606.09289, sobre fases en posesión con T-GAN, se evalúa "on seven Bundesliga matches", es decir los mismos de IDSSE. — [arXiv 2606.09289](https://arxiv.org/abs/2606.09289)
  - floodlight incluye un loader `IDSSEDataset` [medido en `floodlight/io/datasets.py`]. databallpy ofrece `get_open_game()`, que devuelve el partido abierto de la DFL. kloppy ofrece `sportec.load_open_tracking_data`. — [floodlight datasets.py](https://github.com/floodlight-sports/floodlight/blob/main/floodlight/io/datasets.py); [databallpy README](https://github.com/Alek050/databallpy); [kloppy docs](https://kloppy.pysport.org/user-guide/getting-started/)
  - Existe además un espejo en Hugging Face: `pysport/idsse-data`. — [HF idsse-data](https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md)
- **Otros papers de 2025-2026** con tracking y posesión revisados en la API de arXiv, ninguno con datos nuevos abiertos:
  - DEFCON (2512.10355): Eredivisie 2023-25, privado.
  - Wide Open Gazes (2602.18519): 32 partidos de la Copa América 2024 con tracking enriquecido con pose. Los autores abren las herramientas, pero no consta que abran los datos.
  - GenTac (2604.11786) y Vector field (2607.27838): una temporada entera, sin datos públicos.
  - — [arXiv búsqueda](https://export.arxiv.org/api/query?search_query=all:soccer+AND+all:tracking+AND+(all:possession+OR+all:event)&sortBy=submittedDate&sortOrder=descending&max_results=40); [Wide Open Gazes](https://arxiv.org/abs/2602.18519)
- **SambaGraph** (2609.25569). Construye 4.070 episodios a partir de PFF WC2022, que ya conocíamos. No aporta partidos nuevos. — [arXiv 2609.25569](https://arxiv.org/abs/2609.25569)

### Inferencias
- PathCRF puede generar etiquetas de posesión por frame, en formato "jugador en posesión" o "balón en tránsito entre A y B". Basta mapear el jugador a su equipo para tener local/visitante. Para balón muerto haría falta añadir el BallStatus de IDSSE. Así obtenemos un segundo ground truth sobre IDSSE sin pedir permiso a nadie.
- El GT de ELASTIC, con frames de inicio y recepción anotados a mano en 3 partidos, sirve para medir el error temporal en los cambios de posesión con más precisión que los eventos DFL originales.

### Lagunas
- No pude leer el README de JointDiff (la descarga raw volvió vacía). Queda sin confirmar si "Soccer-U" se publica, de qué proveedor viene, si incluye balón y si trae estados del balón.
- No encontré la declaración de disponibilidad de datos del paper de Borghesi en ScienceDirect.

## 2. ¿Hay publicaciones nuevas en 2025-2026 en GitHub, Hugging Face, Zenodo, Figshare, Kaggle u OSF?

### Conclusión
Solo hay dos candidatos realmente nuevos que no estaban en la lista de conocidos:
- **FOOTPASS / SoccerNet SN-PCBAS-2026**: 54 partidos completos con posiciones de jugadores y play-by-play con equipo. Acceso con NDA aprobado a mano, y no tiene balón.
- **GRF simulated tracking**: 3.000 partidos sintéticos de Google Research Football con un flag `ON_BALL` por jugador. Acceso abierto inmediato, pero sin licencia.

No encontré ninguna publicación abierta nueva de DFL/IDSSE, de muestras de MLS/Second Spectrum, Hawk-Eye, Stats Perform, Signality o Tracab, de PFF ampliado, ni de tracking femenino de partidos completos.

### Hallazgos citados

**A) FOOTPASS (Footovision), publicado como SoccerNet/SN-PCBAS-2026**
- Partidos: "54 full-length matches", repartidos en 48 de entrenamiento, 3 de validación y 3 de test (challenge). Frecuencia: 25 fps. — [arXiv 2511.16183](https://arxiv.org/html/2511.16183v1)
- Tracking: los datos de "game-state" dan posición y velocidad en el campo de todos los jugadores, con equipo, dorsal y rol. Las posiciones de los jugadores que no salen en la emisión están **imputadas**. El paper no menciona tracking del balón. — [arXiv 2511.16183](https://arxiv.org/html/2511.16183v1)
- Señal de posesión: eventos con el formato (frame, clase, equipo, dorsal), con equipo 0 = izquierda y 1 = derecha, y 8 clases de acción con balón. No hay campo de balón vivo/muerto. — [arXiv 2511.16183](https://arxiv.org/html/2511.16183v1)
- Licencia, citada literalmente del paper:
  - "The FOOTPASS annotation files supporting this study will be released on Hugging Face and the baselines on Github"
  - "where the final license and README will specify terms of use for research and evaluation."
  - "The associated broadcast videos are distributed via the SoccerNet dataset repository on Hugging Face … remain subject to SoccerNet's current non-disclosure agreement (NDA)."
  - "Researchers wishing to access the footage must request it directly from SoccerNet and agree to the NDA"
  - — [arXiv 2511.16183](https://arxiv.org/html/2511.16183v1)
- En Hugging Face [medido]: `SoccerNet/SN-PCBAS-2026`, con `gated: manual`, etiqueta `license:other` y última modificación el 2026-10-04. Los ficheros `tactical_data_TRAIN/VAL/CHALLENGE.zip` suman unos 2,77 GB (2.458 + 157 + 155 MB). El total, contando los vídeos, es de unos 235,7 GB. El README y el fichero de formato no se pueden leer sin acceso. — [HF SN-PCBAS-2026](https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026)
- Baselines: repo `JeremieOchin/FOOTPASS`, licencia Apache-2.0, último push 2026-06-17 [medido]. — [GitHub FOOTPASS](https://github.com/JeremieOchin/FOOTPASS)
- **Requiere aprobación manual (NDA de SoccerNet).** Con el plazo del 17/11 no hay garantía de acceso a tiempo.

**B) GRF simulated tracking (Radke & Tilbury, AAMAS 2025 Demo, arXiv 2503.19809)**
- Partidos: "3,000 pre-recorded simulated games", alojados en Google Drive. Subconjuntos disponibles:

  | Subconjunto | Partidos | Tamaño aprox. |
  |---|---|---|
  | tiny | 12 | 100 MB |
  | small | 150 | 1,25 GB |
  | medium | 600 | 5 GB |
  | large | 1.500 | 12,5 GB |
  | full | 3.000 | 25 GB |

  — [README grf_tracking_data](https://github.com/Dtradke/grf_tracking_data)
- Esquema [medido en `scripts/constants.py`]: `GAME_ID, TIMESTEP, TEAM_0_SCORE, TEAM_1_SCORE, PLAYER_ID, TEAM_ID, PLAYER_ROLE, LOCATION_X/Y/Z, VELOCITY_X/Y/Z, ON_BALL`. El balón aparece como `PLAYER_ID == -1` [medido en `scripts/event_maker.py`]. — [constants.py](https://github.com/Dtradke/grf_tracking_data/blob/main/scripts/constants.py)
- En el entorno GRF original, `ball_owned_team` toma valores {-1, 0, 1} y `game_mode` indica saque, córner, etc. — [GRF observation.md](https://github.com/google-research/football/blob/master/gfootball/doc/observation.md). No consta que `game_mode` se guarde en el esquema exportado.
- Licencia: el repo **no declara licencia** [medido: `license: null`]. Su último push es de 2025-03-24 [medido].
- Acceso abierto e inmediato. Es sintético, así que hay que tener en cuenta el cambio de dominio respecto a partidos reales.

**C) Comprobaciones sin hallazgos nuevos**
- **Zenodo** [medido], búsquedas "soccer AND tracking data", "football AND tracking data…" y "soccer tracking": no hay datasets de tracking de partidos completos de 2025-2026. Solo salen paquetes de código o artículos, por ejemplo [16929566](https://zenodo.org/records/16929566) y [21338808](https://zenodo.org/records/21338808). — [Zenodo API](https://zenodo.org/api/records?q=%22soccer%22%20AND%20%22tracking%20data%22)
- **Figshare** [medido]: lo único de tracking es "Frame-Level Women's Soccer Penalty-Kick Data Captured by Computer Vision Models" (2026-06-24). Solo cubre penaltis, así que no sirve para posesión. — [Figshare 31464526](https://figshare.com/articles/dataset/Women_s_Soccer_Penalty_Kick_Data_Captured_by_Computer_Vision_Models/31464526)
- **Hugging Face** [medido], búsquedas soccer, football y tracking-data: solo aparecen conjuntos ya conocidos (SoccerNet GSR/Tracking, `atomscott/soccertrack-v2` con gating automático y CC-BY-4.0, y el derivado `algo-dev99/soccertrack-v2-followcam-events`). — [HF API](https://huggingface.co/api/datasets?search=soccer)
- **withqwerty/open-football** [medido]: licencia CC0, último push 2026-10-03. Su sección "Tracking data" lista solo datasets que ya conocíamos (SkillCorner, Metrica, IDSSE, PFF WC2022, Last Row, Alfheim, SoccerMon). Advierte de que "genuinely open event or tracking data from Africa, Asia, and South America barely exists". — [open-football README](https://github.com/withqwerty/open-football)
- **Cambio en un dataset conocido**: SkillCorner open data usa "v3 format since September 2026", con 20 partidos de la A-League 2024/25 y licencia MIT. El repo tuvo su último push el 2026-09-14 [medido]. La descripción del repo sigue diciendo "10 matches", lo que es incoherente con los 20 que lista open-football. — [open-football](https://github.com/withqwerty/open-football); [SkillCorner/opendata](https://github.com/SkillCorner/opendata)
- **kloppy**: soporta DataFactory, HawkEye, Impect, Hudl Sportscode/StatsBomb, Metrica, Opta, PFF, SecondSpectrum, Signality, SkillCorner, Sportec, Stats Perform, TRACAB y Wyscout. El único loader abierto de tracking documentado es Sportec/IDSSE. — [kloppy docs](https://kloppy.pysport.org/user-guide/getting-started/)
- **DFL/Sportec**: desde la temporada 2025-26 la DFL captura tracking 3D de todos los partidos de Bundesliga, pero no encontré ninguna publicación abierta nueva. — [SVG Europe](https://svgplay.sportsvideo.org/videos/3d-player-tracking-in-the-bundesliga); [DSHS Köln](https://www.dshs-koeln.de/en/institut-fuer-trainingswissenschaft-und-sportinformatik/detail/meldung/official-game-data-in-the-journal-scientific-data)
- **PFF FC**: no encontré ninguna publicación abierta en 2026. Su tracking por emisión (con Sportlogiq) es un producto de pago. — [fc.pff.com](https://fc.pff.com/)

### Tabla resumen de candidatos nuevos

| Candidato | Partidos | Hz | Balón | Señal de posesión | Vivo/muerto | Licencia | Acceso | Tamaño | Uso |
|---|---|---|---|---|---|---|---|---|---|
| PathCRF: posesión por frame sobre IDSSE | 7 (IDSSE) | 25 | sí (IDSSE) | posesión por frame (jugador o pase), generada por script | sí, vía BallStatus de IDSSE | código MPL-2.0; datos base CC-BY 4.0 | abierto | repo 187 MB [medido] | validación (segunda etiqueta) |
| ELASTIC GT | 3 (IDSSE) | 25 | sí | eventos con frames de inicio y recepción anotados a mano | sí, vía IDSSE | CC BY 4.0 (benchmark) | abierto | repo 22,6 MB [medido] | validación de los instantes de cambio |
| FOOTPASS / SN-PCBAS-2026 | 54 | 25 | **no** (jugadores imputados fuera de cámara) | eventos (frame, equipo, dorsal) | no | "other"; "terms of use for research and evaluation" + NDA | **gated manual (NDA)** | tactical ≈ 2,77 GB [medido] | validación y entrenamiento, si se obtiene acceso |
| GRF simulated | 3.000 | no consta (GRF ≈ 10 pasos/s, sin verificar) | sí | `ON_BALL` por jugador (verdad del simulador) | no explícito | **sin licencia** | abierto (Google Drive) | 100 MB – 25 GB | solo prueba de robustez; dominio sintético |

### Inferencias
- Para validar antes del 17/11, lo único accesible de inmediato es un contraste adicional sobre IDSSE (PathCRF y ELASTIC) y una prueba sintética (GRF). Ninguno añade partidos reales nuevos con balón.
- FOOTPASS es el único conjunto real nuevo con muchos partidos, pero tiene dos limitaciones. Sin balón, no permite validar la clase "balón muerto". Y para usarlo como fuente de partidos habría que solicitar el NDA ya.
- La falta de licencia en GRF impide redistribuirlo. Usarlo de forma interna como prueba no sería redistribución, pero conviene confirmarlo con RULES.md.

### Lagunas
- No pude leer `tactical_data_format.txt` de SN-PCBAS-2026 sin acceso, así que no puedo descartar que incluya balón o estado del juego.
- No consta la frecuencia exacta ni si se exporta `game_mode` en GRF tracking.
- Solo hice búsquedas web sobre Kaggle, Mendeley Data y OSF, sin consultar sus APIs. No encontré datasets nuevos de tracking de partidos completos en ellos.
- No encontré muestras públicas nuevas de Second Spectrum, Hawk-Eye, Signality ni Tracab en 2025-2026.
