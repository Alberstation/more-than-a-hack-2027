# Datos externos: tracking jugadores+balón con posesión o eventos sincronizados (a nov. 2026)

Criterio del equipo: solo descarga directa (sin formularios, registro ni petición). Ya usados: IDSSE/DFL (7 partidos) y Metrica (3). Descartado: SkillCorner (broadcast, saturado).

## 1. ¿Hay más partidos abiertos de DFL/Sportec además de los 7 de Bassek et al. 2025?

### Takeaway
No he encontrado ninguna ampliación: la versión abierta sigue siendo de 7 partidos (2022/23, 2 de 1. Bundesliga y 5 de 2. Bundesliga). El "Bundesliga Data Shootout" es vídeo con etiquetas de evento, sin tracking. Lo único nuevo que se puede añadir sobre esos mismos 7 partidos son las etiquetas de ELASTIC: timestamps de eventos reanotados a mano en 3 partidos, con eventos de control añadidos.

### Cited Findings
- IDSSE: 7 partidos completos de la temporada 2022/23 (1. y 2. Bundesliga). Tracking óptico TRACAB de todos los jugadores y del balón a 25 fps, eventos sincronizados y metadatos. Licencia CC-BY 4.0, propietario de los datos DFL. IDs: J03WPY, J03WMX, J03WN1, J03WOH, J03WOY, J03WPF, J03WQF. Figshare DOI 10.6084/m9.figshare.28196177. Espejo en Hugging Face `pysport/idsse-data`, que solo describe 7 partidos — [HF pysport/idsse-data README](https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md)
- La organización `pysport` en Hugging Face solo aloja ese dataset (actualizado el 4 de marzo) — [HF pysport](https://huggingface.co/pysport)
- databallpy `get_open_game` acepta los proveedores "metrica", "dfl", "sportec" y "tracab", con valor por defecto sportec/J03WMX. Descarga y cachea automáticamente — [databallpy game API](https://databallpy.readthedocs.io/en/main/api_docs/game_api.html)
- Bundesliga Data Shootout (Kaggle, 30 jul.–13 oct. 2022, DFL + AWS + Sportec): vídeo con etiquetas challenge/play/throwin. Según fuentes secundarias, 12 vídeos de 60 min para entrenar. No incluye tracking — [bundesliga.com](https://bundesliga.com/en/aws-bundesliga-data-shootout); [resumen secundario](https://xxxxyyyy80008.github.io/3_machine_learning/1_data_processing/video%20data/dfl-exploratory-data-analysis-and-video-processing.html)
- ELASTIC (CIKM '26, arXiv 2608.30227): reanotó los timestamps de eventos de **3** de los 7 partidos DFL. El ground truth es la mediana de 3 anotadores, que además insertaron los eventos de **control**, out y gol que faltaban. El paper dice que el código y el benchmark son públicos en github.com/hyunsungkim-ds/elastic. El conjunto Eredivisie (200+50+157 partidos) es propietario y no se publica — [arXiv 2608.30227](https://arxiv.org/html/2608.30227)
- PathCRF (detección de eventos sin balón, posesión inferida desde trayectorias) usa solo el Sportec Open DFL (5 partidos de train, 1 de val y 1 de test, a 5 fps) "para reproducibilidad". El apéndice B usa un conjunto no público. No hay un K League abierto — [arXiv 2602.12080](https://arxiv.org/html/2602.12080)

### Inferences
- La única ampliación "DFL" disponible son las etiquetas de ELASTIC sobre 3 de los partidos que ya tenemos. Sirven para afinar la sincronización evento-tracking y como etiquetas de control (recepciones), que es justo lo que define la posesión controlada. El esfuerzo debería ser bajo: mismo ID de partido, solo hay que hacer join por event id/frame.
- El código de PathCRF (github.com/hyunsungkim-ds/pathcrf) seguramente incluye la etiquetación de posesión derivada del DFL. Conviene comparar su definición de posesión con la nuestra.

### Gaps
- No pude abrir la página de Figshare (HTTP 403), así que no verifiqué su historial de versiones. Conviene revisarla a mano por si hay una v2.
- No verifiqué el contenido ni la licencia del repo de ELASTIC (formato de las etiquetas, si incluye "control").

## 2. SkillCorner Open Data (actualizaciones 2024-2026): ¿dynamic events / phases of play sincronizados? Licencia

### Takeaway
Sí. El repo actual tiene 10 partidos de la A-League australiana 2024/25 a 10 fps, y cada frame trae un campo `possession` (player_id y grupo home/away). Por partido incluye además `dynamic_events.csv` (con filas `player_possession`) y `phases_of_play.csv` (frames de inicio y fin). La licencia es MIT. Es tracking broadcast, el mismo dominio que el equipo ya da por saturado. Pero si lo que se descargó fue la versión antigua de 9 partidos, sin eventos, conviene comprobarlo: las etiquetas ya vienen hechas y no hay que derivarlas.

### Cited Findings
- 10 partidos de broadcast tracking, muestra de la A-League 2024/2025. Tracking a 10 fps; el Body Pose va a 25 fps, con las posiciones 2D interpoladas para cuadrar. `ball_data` en cada frame. Dict `possession` con `player_id` y `group` (home/away en posesión) — [GitHub SkillCorner/opendata](https://github.com/SkillCorner/opendata)
- `{id}_dynamic_events.csv`: una fila por evento, con x/y sin escalar a un campo estándar, 12 columnas de EPV y 8 de presión/dificultad (rellenas solo en las filas `player_possession`). `{id}_phases_of_play.csv`: frames de inicio y fin de cada fase, definidas solo con el balón en juego. Formatos: `{id}_match.json`, `{id}_tracking_extrapolated.jsonl`, más CSV. El Body Pose está en Hugging Face, unos 3,3 GB de JSON por partido — [GitHub SkillCorner/opendata](https://github.com/SkillCorner/opendata)
- Licencia: "The MIT License. Copyright (c) 2020 SkillCorner" (permite uso comercial; pide atribución) — [LICENSE](https://raw.githubusercontent.com/SkillCorner/opendata/master/LICENSE)
- Cada fase en posesión tiene su fase fuera de posesión correspondiente. Desde febrero de 2026 los Dynamic Events de SkillCorner se generan con detección automática de eventos on-ball — [skillcorner.github.io/opendata](https://skillcorner.github.io/opendata/)
- kloppy: `skillcorner.load_open_data(match_id='1925299', ..., include_empty_frames=False, only_alive=True)` lee directamente del repo de GitHub — [kloppy docs preview](https://deploy-preview-546--unruffled-nobel-ed43d4.netlify.app/reference/providers/skillcorner)
- La versión antigua de "SkillCorner open data", tal como la enlaza el repo de Metrica: 9 partidos de broadcast tracking, sin eventos y sin vídeo — [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)

### Inferences
- Si los "20 partidos" descargados son 9 antiguos + 10 nuevos (más uno), el `possession` por frame y `player_possession` ya existen en los 10 de la A-League. Probarlos como etiquetas nativas, en vez de derivadas, cuesta poco: se pueden cargar directamente con kloppy.
- `possession.group` sirve como etiqueta de posesión de equipo por frame. Las filas `player_possession` de dynamic_events dan los intervalos de control individual (frames de inicio y fin).

### Gaps
- No sé qué versión descargó exactamente el equipo, ni si su evaluación de "saturación" usó las etiquetas nativas.
- No aparece el tamaño total del repo, ni si usa Git LFS (hay `.gitattributes`, pero su contenido no está visible).
- No confirmé si los ficheros del repo son anteriores o posteriores al cambio a eventos automáticos de febrero de 2026.

## 3. Otros candidatos (inventario)

### Takeaway
Con descarga directa y tracking que incluye el balón, solo aparecen dos candidatos. Last Row (19 goles, 20 fps, sin eventos ni posesión, precisión baja) y SoccerTrack v2 (10 partidos universitarios, CC BY 4.0, descarga directa, posiciones de jugadores en metros y 12 clases de eventos con timestamp, aunque el balón por frame no está documentado). Los conjuntos ricos (PFF/Gradient con los 64 partidos del Mundial 2022, SoccerNet-GAR, FOOTPASS) piden formulario o NDA y quedan fuera según el criterio del equipo. No encontré ningún dataset broadcast/automático abierto con etiquetas de posesión, aparte de SkillCorner.

### Cited Findings

**PFF FC / Gradient Sports, Mundial 2022 (excluido: formulario)**
- Los 64 partidos combinan tracking broadcast y eventos. Tracking `{game_id}.jsonl.bz2` por partido; `events.json`, que en la actualización pasa a ficheros por partido; metadatos en CSV (metadata, players, rosters, competitions); especificaciones en PDF. Acceso: "simply fill out the form below". No se publica una licencia explícita — [Gradient Sports blog](https://www.gradientsports.com/blog/pff-fc-release-2022-world-cup-data); [Enhanced dataset](https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset)
- Sigue requiriendo formulario ("fill out the form below to gain instant access") — [Gradient enhanced](https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset)
- Hay un loader en kloppy, `pff.load_tracking(raw_data=..., meta_data=..., roster_meta_data=...)`, para ficheros locales. No hay `load_open_data` — [UnravelSports keynote](https://rawcdn.githack.com/UnravelSports/keynotes/main/html/20250430-AIMS-Cameroon.html)
- No encontré espejos en Kaggle ni en HF de los `.jsonl.bz2` originales.

**SoccerNet-GAR (OpenSportsLab, HF), derivado de PFF WC2022 (excluido: formulario + NDA)**
- 64 partidos y 87.939 eventos en 10 clases (PASS 65,4 %, TACKLE 12,4 %, OUT, HEADER, THROW IN, CROSS, FREE KICK, SHOT, GOAL, HIGH PASS). Posiciones 2D de jugadores y balón 3D a unos 29,97 fps; el balón aparece en el 93,4 % de los frames. Son clips de unos 4,8 s por evento (rama `tracking-full`, 144 frames), no partidos completos. Formato Parquet + TAR WebDataset. Acceso con formulario y NDA de SoccerNet, solo para investigación no comercial — [HF soccernetpro-classification-GAR](https://huggingface.co/datasets/OpenSportsLab/soccernetpro-classification-GAR/blob/main/README.md); [HF SoccerNet-GAR](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR)

**SoccerTrack v2 (Nagoya / BePro), descarga directa**
- 10 partidos universitarios amateur (~900–932 min), vídeo panorámico 4K (BePro Cerberus en 2 partidos, sistemas de 3 cámaras en 8). GSR: coordenadas 2D en metros por frame, track/player ID, rol, dorsal y lado del equipo. BAS: 12 clases (Pass, Drive, Header, High Pass, Out, Cross, Throw In, Shot, Ball Player Block, Player Successful Tackle, Free Kick, Goal) con timestamp en ms, equipo y player ID. Licencia: dataset CC BY 4.0, código MIT. Descarga desde Hugging Face (principal) o Google Drive, con `./scripts/download.sh --dest ./data [--match]`. Formatos: MP4, JSON (GSR, BAS), MOTChallenge y raw (keypoints JSON, `.npy`, XML del tracker) — [proyecto SoccerTrack v2](https://atomscott.github.io/SoccerTrack-v2/)
- El paper no documenta que se siga el balón: el GSR cubre jugadores, porteros y árbitros, y el balón solo aparece como timestamps de BAS. No hay etiquetas de posesión. Los eventos BAS se convirtieron desde los logs de evento de BePro y se revisaron a mano — [arXiv 2508.01802](https://arxiv.org/html/2508.01802v1)
- La doc de OpenStarLab tiene un loader "BePro (SoccerTrackv2)" que "añade columnas de tracking específicas del balón". Los eventos son un CSV sincronizado en el tiempo con los frames, y la salida se submuestrea a 5 fps — [OpenStarLab BePro](https://openstarlab.readthedocs.io/en/latest/Pre_Processing/Sports/Phase_data/Data_Provider/Football/Bepro/contents.html)
- En HF el dataset aparece como `atomscott/soccertrack-v2` — [búsqueda HF atomscott](https://www.huggingface.co/atomscott)

**Last Row (Friends of Tracking), descarga directa**
- `liverpool_2019.csv`: 19 goles del Liverpool en 2019 a 20 fps, con balón (z solo para el balón). Solo aparecen los jugadores cercanos al balón, y los IDs no son consistentes entre jugadas. No hay eventos ni posesión. Los propios autores dicen que "lacks the accuracy needed for research". Licencia informal: "Feel free to use the dataset for whatever purpose, but please credit the source" — [GitHub Last-Row](https://github.com/Friends-of-Tracking-Data-FoTD/Last-Row)

**Signality / Hammarby**
- 3 partidos del Hammarby trackeados por Signality, del curso de David Sumpter. Tracking y vídeo sin sincronizar y sin eventos. Están enlazados desde el repo de Metrica, que apunta a un curso de Uppsala — [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)

**Metrica (ya usado)**: Sample Games 1 y 2 en CSV, con tracking y eventos sincronizados; Game 3 en EPTS FIFA + JSON de eventos. No se nombra una licencia: piden "be responsible" y citar la fuente — [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)

**SoccerNet GSR (sn-gamestate)**: 200 clips de 30 s. Roles anotados: player, goalkeeper, referee y other, sin clase balón. Descarga con `SoccerNetDownloader(task="gamestate-2024")`, sin NDA mencionado. El repo es GPL-3.0; el dataset no tiene licencia propia indicada — [GitHub sn-gamestate](https://github.com/SoccerNet/sn-gamestate); [arXiv 2404.11335](https://arxiv.org/abs/2404.11335v1)

**FOOTPASS (2025, arXiv 2511.16183)**: 54 partidos completos de la temporada 2023/24 (Ligue 1, Bundesliga, Serie A, LaLiga, UCL). Game-state broadcast a 25 fps, con posiciones y velocidades por frame y 13 roles; el paper lo califica de "FIFA EPTS certified". Unos 102.992 eventos on-ball (drive, pass, cross, shot, header, throw-in, tackle, block) en forma de (frame, clase, equipo, dorsal). No describe trayectoria del balón ni campo de posesión. Las anotaciones "se publicarán" en HF con una licencia "for research and evaluation" aún no cerrada; los vídeos requieren el NDA de SoccerNet — [arXiv 2511.16183](https://arxiv.org/html/2511.16183)

**TeamTrack (CVPRW 2024)**: MOT multideporte en Kaggle (`atomscott/teamtrack`). Para fútbol, 30 min de fisheye lateral y 30 min de dron cenital. No encontré que el balón esté anotado ni que haya eventos, y la licencia no está verificada — [arXiv 2404.13868](https://arxiv.org/abs/2404.13868)

**Respo.Vision Copa América 2024**: un paper usa tracking broadcast de Respo.Vision con eventos de Stats Perform/Opta sincronizados. No encontré ninguna publicación abierta de esos datos — [arXiv 2602.18519](https://arxiv.org/pdf/2602.18519); [búsqueda StatsBomb Copa América, solo eventos](https://blogarchive.statsbomb.com/news/statsbomb-release-free-copa-america-2024-data/)

**Impect open data**: `impect.load_open_data(match_id=122840)` existe en kloppy. Es data de eventos y no se describe tracking — [GitHub ImpectAPI/open-data](https://github.com/ImpectAPI/open-data)

**K League / BePro (PathCRF)**: BePro vende los datos de la K League por paquetes y API, con precio a consultar. No hay versión abierta — [BePro data API](https://www.bepro11.com/football-data-api)

### Inferences
- Entre las opciones de descarga directa, la que más se parece a nuestro dominio (automático, cámara fija, balón a menudo ausente) es SoccerTrack v2. Si los XML/JSON raw del tracker BePro incluyen el balón, como sugiere OpenStarLab, junto con los eventos BAS (Pass/Drive/Shot con player ID y timestamp en ms) se podría derivar posesión controlada como en DFL. Esfuerzo: descargar las anotaciones sin los vídeos 4K, si el script lo permite, y mapear BAS a frames del GSR. Siguiente paso: listar los ficheros de `atomscott/soccertrack-v2` y abrir un frame raw.
- Last Row sirve como mucho para comprobar el seguimiento del balón: no tiene eventos, solo cubre los jugadores cercanos y su precisión es baja.
- Si el equipo relaja la norma de "sin formularios", el siguiente gran salto sería PFF/Gradient: 64 partidos broadcast con balón y eventos, con loader en kloppy. FOOTPASS y SoccerNet-GAR requieren NDA.

### Gaps
- No confirmé si SoccerTrack v2 trae el balón por frame en coordenadas de campo, ni su tamaño (sin los vídeos) o su fps.
- No verifiqué la licencia de TeamTrack ni si anota el balón.
- No se pudieron verificar las tasas de fps ni la licencia exacta de PFF/Gradient (la página no publica licencia).
- No encontré datasets abiertos 2023-2026 de "Ball Radar" ni de "TranSPORTmer" con descarga propia. Por lo visto, estos papers suelen evaluar sobre DFL/Metrica/SoccerNet, pero no lo verifiqué paper a paper.

## 4. ¿Qué datasets permiten posesión de equipo por frame o derivarla de eventos sincronizados?

### Takeaway
Etiqueta nativa por frame: solo SkillCorner (`possession.group` + `player_possession`). Derivable de eventos sincronizados: DFL (más las etiquetas de control de ELASTIC en 3 partidos), Metrica (juegos 1-2), PFF/Gradient y SoccerNet-GAR (ambos con formulario) y, potencialmente, SoccerTrack v2 (BAS con timestamps, aunque falta confirmar el balón). Sin eventos: Last Row, Signality/Hammarby, SoccerNet GSR, TeamTrack.

### Cited Findings
- SkillCorner: dict `possession` (player_id, group) por frame y `phases_of_play` con frames de inicio y fin — [GitHub SkillCorner/opendata](https://github.com/SkillCorner/opendata)
- DFL: eventos sincronizados con tracking a 25 fps — [HF idsse-data](https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md). ELASTIC añade eventos de control en 3 partidos — [arXiv 2608.30227](https://arxiv.org/html/2608.30227)
- Metrica: en los juegos 1-2, tracking y eventos sincronizados — [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data). databallpy confirma que en Metrica "the datetimes are already perfectly aligned" y ofrece `game.is_synchronised` — [databallpy sync](https://databallpy.readthedocs.io/en/main/getting_started/synchronisation_page.html)
- SoccerTrack v2: BAS con timestamp en ms en la línea temporal global, más equipo y player ID — [proyecto](https://atomscott.github.io/SoccerTrack-v2/)

### Inferences
- Para cumplir la regla de descarga directa en 7 días, el orden por coste/beneficio es: (1) etiquetas de ELASTIC sobre DFL (ya cargado); (2) etiquetas nativas de SkillCorner A-League, si no se usaron; (3) SoccerTrack v2, explorando primero si trae el balón.

### Gaps
- Ninguna fuente abierta con descarga directa ofrece tracking automático o broadcast con balón y posesión nativa, aparte de SkillCorner. No encontré alternativas, y es posible que no existan públicamente a noviembre de 2026.
