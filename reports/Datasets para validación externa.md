# Datasets externos para validar el modelo congelado de posesión (H/A/D)

Fecha: 2026-11-13. Entrega: 17/11/2026. Objetivo: validar v6 congelado fuera de IDSSE (7), Metrica (3) y SkillCorner (20), sin reentrenar. **[measured]** = comprobado en esta sesión contra la API de GitHub, Hugging Face, Zenodo, Figshare o arXiv, o leyendo el fichero raw. No se descargó ningún dataset ni se aceptó ningún gate o formulario.

**Respuesta corta.** Hay tres fuentes de partidos reales con acceso inmediato o casi inmediato. **SoccerTrack v2** tiene 10 partidos nuevos, licencia CC BY 4.0, gate de aprobación automática y un `ballStatus` por frame en el XML del proveedor. Es la primera que conviene bajar. **PFF/Gradient WC2022** tiene 64 partidos de tracking broadcast con huecos de balón reales y lector en kloppy. Su formulario promete acceso instantáneo, pero no se puede redistribuir y sus términos siguen sin leerse. **PathCRF/ELASTIC** aporta una segunda etiqueta independiente sobre partidos IDSSE, no partidos nuevos. **FOOTPASS** (54 partidos, sin balón) exige NDA con aprobación manual. **GRF** son 3.000 partidos sintéticos sin licencia. No apareció ninguna otra publicación abierta en 2025-2026 con tracking de partido completo y posesión.

## 1. ¿Qué candidatos existen y qué ofrece cada uno?

### Takeaway
Fuera de los tres conjuntos que ya usamos, solo SoccerTrack v2 cumple las cuatro condiciones a la vez: partidos nuevos, señal vivo/muerto por frame, licencia abierta y acceso por autoaceptación. PFF es el conjunto más grande y el más parecido a un tracker que pierde el balón. Los demás sirven como contraste de etiquetas (PathCRF, ELASTIC) o como prueba de robustez (GRF).

### Cited Findings

| Candidato | Partidos | Hz | Balón | Señal de posesión / vivo-muerto | Licencia citada | Acceso | Tamaño | Uso |
|---|---|---|---|---|---|---|---|---|
| **SoccerTrack v2** | 10, universitarios; 932-934 min — [paper](https://arxiv.org/html/2508.01802), [HF](https://huggingface.co/datasets/atomscott/soccertrack-v2) | GSR a 25 fps — [format-gsr.md](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-gsr.md) | **Reconstruido**, interpolado entre eventos — [HF](https://huggingface.co/datasets/atomscott/soccertrack-v2) | 21.428 eventos BAS con `team` y `frame` (v1.2); `ballStatus` por frame en el XML bruto — [dataclass.py](https://github.com/AtomScott/SoccerTrack-v2/blob/main/scripts/tracking_data_utils/dataclass.py) | CC BY 4.0 (datos), MIT (código) — [LICENSE-DATA](https://github.com/AtomScott/SoccerTrack-v2/blob/main/LICENSE-DATA) | HF `gated: auto` [measured] | ~2,3 GB sin GSR ni vídeo; GSR 55,95 GB [measured] | Validar y entrenar |
| **PFF/Gradient WC2022** | 64, España en 4 — [Gradient](https://www.gradientsports.com/blog/pff-fc-release-2022-world-cup-data) | 29,97 — [arXiv 2609.24040](https://arxiv.org/html/2609.24040v1) | Sí, 3D; presente en el 93,4 % de los frames de las ventanas de evento [measured: tarjeta HF] — [SoccerNet-GAR](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR) | `game_event.home_ball` y OUT/END/KICKOFF/OTB — [kloppy pff.py](https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/tracking/pff.py) | Sin licencia publicada; los papers hablan de "terms of use" y "research licence" — [arXiv 2609.24040](https://arxiv.org/html/2609.24040v1) | Formulario, "instant access" — [Gradient 2025](https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset) | ~50 GB en total — [arXiv 2609.24040](https://arxiv.org/html/2609.24040v1) | Validar; entrenar según los términos |
| **PathCRF** (sobre IDSSE) | 7 (los de IDSSE) | 25 | Sí (IDSSE) | Script que genera "per-frame ground-truth possession states"; muerto vía BallStatus de IDSSE — [repo](https://github.com/hyunsungkim-ds/pathcrf) | MPL-2.0 (código) [measured] | Abierto | Repo 187 MB [measured] | Validar (segunda etiqueta) |
| **ELASTIC GT** (sobre IDSSE) | 3 (J03WMX, J03WN1, J03WPY) | 25 | Sí | Eventos con `frame_id`/`receive_frame_id` anotados a mano, mediana de 3 anotadores — [repo](https://github.com/hyunsungkim-ds/elastic) | CC BY 4.0 (`benchmark/`) | Abierto | Repo 22,6 MB [measured] | Validar instantes de cambio |
| **FOOTPASS / SN-PCBAS-2026** | 54 (48/3/3) — [arXiv 2511.16183](https://arxiv.org/html/2511.16183v1) | 25 | **No**; jugadores fuera de cámara imputados | Eventos (frame, clase, equipo, dorsal); sin vivo/muerto | "other"; "terms of use for research and evaluation" + NDA de SoccerNet — [arXiv 2511.16183](https://arxiv.org/html/2511.16183v1) | `gated: manual` [measured] — [HF](https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026) | `tactical_data_*` 2,77 GB [measured] | Validar H/A y entrenar, con acceso |
| **GRF simulated** | 3.000 sintéticos — [README](https://github.com/Dtradke/grf_tracking_data) | Sin documentar | Sí (`PLAYER_ID == -1`) [measured] | `ON_BALL` por jugador [measured: constants.py] | **Sin licencia** [measured] | Abierto (Google Drive) | 100 MB (12 partidos) a 25 GB | Solo robustez |

**Descartados para esta validación.**
- **SoccerNet-GAR**: los mismos 64 partidos de PFF en clips de 16 a 144 frames, sin campo de posesión y con NDA — [HF](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR).
- **TranSPORTmer**: 283 partidos de LaLiga con etiqueta por frame, pero no publica datos — [arXiv 2410.17785](https://arxiv.org/html/2410.17785v2).
- **Borghesi et al.**: usan exactamente nuestras clases (inactivo, local, visitante) con datos de la Serie A, pero no encontré datos publicados — [PoliTO](https://iris.polito.it/handle/11583/2976470).
- **DEFCON, Wide Open Gazes, GenTac**: datos privados — [arXiv](https://arxiv.org/abs/2602.18519).

**Sin novedades abiertas en 2025-2026.**
- Las búsquedas en Zenodo, Figshare y HF no dieron nada nuevo [measured]. En Figshare solo aparecen penaltis femeninos — [Figshare](https://figshare.com/articles/dataset/Women_s_Soccer_Penalty_Kick_Data_Captured_by_Computer_Vision_Models/31464526).
- open-football (CC0, actualizado el 2026-10-03) solo lista conjuntos ya conocidos — [README](https://github.com/withqwerty/open-football).
- La DFL captura 3D desde 2025-26 sin publicación abierta — [SVG](https://svgplay.sportsvideo.org/videos/3d-player-tracking-in-the-bundesliga).
- Gradient solo anuncia acuerdos comerciales en 2026 — [blog](https://www.gradientsports.com/blog/).

**SkillCorner.** El `matches.json` y el árbol del repo listan 20 partidos [measured en la sesión del 2026-11-01] — [matches.json](https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches.json). Hay formato v3 desde septiembre de 2026 — [open-football](https://github.com/withqwerty/open-football). El README sigue diciendo 10, así que está desactualizado.

### Inferences
- Para el reto de "inspeccionable", SoccerTrack v2 encaja limpio: cualquier organizador acepta el gate automático y descarga la misma revisión (`v1.2`). PFF exige que cada organizador rellene el formulario. Basta con documentar en el repo el `game_id` y la ruta de obtención, sin subir los datos.
- PathCRF y ELASTIC no amplían el dominio. Si v6 se entrenó con IDSSE, su valor es contrastar nuestra definición de "controlada" con otra etiqueta independiente, no medir generalización.

### Gaps
- No se leyó `tactical_data_format.txt` de SN-PCBAS-2026: hace falta acceso para saber si trae balón o estado del juego.
- No se consultaron las APIs de Kaggle, Mendeley Data y OSF; en esas plataformas solo se hicieron búsquedas web.

## 2. ¿Cuáles parecen un tracker automático con huecos de balón?

### Takeaway
PFF es el único conjunto nuevo con balón detectado desde la emisión y huecos medidos. Hay un matiz: kloppy entrega `ballsSmoothed`, así que los huecos reales hay que leerlos del campo `balls` raw. FOOTPASS es broadcast pero sin balón, lo que equivale a un 100 % de hueco. SoccerTrack v2 usa cámara panorámica fija y su balón es una reconstrucción, de modo que no tiene huecos auténticos.

### Cited Findings
- **PFF**. Balón presente en el 93,4 % de los frames de las ventanas de evento. El 85,9 % de los eventos tiene el balón completo en su ventana, y existe un flag `tracking_valid` [measured: tarjeta HF] — [SoccerNet-GAR](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR). G2G cita "intermittent ball visibility" y "smoothing artifacts" — [arXiv 2609.24040](https://arxiv.org/html/2609.24040v1).
- **Lector de kloppy para PFF** [measured: código raw] — [pff.py](https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/tracking/pff.py):
  - Lee solo `ballsSmoothed` y los jugadores smoothed, emparejados por `jerseyNum`.
  - Ignora `balls` y `possession_event`.
  - Fija la frecuencia con `frame_rate = metadata["fps"]`.
- **FOOTPASS**: "game-state" desde la emisión, con jugadores fuera de plano imputados y sin tracking de balón — [arXiv 2511.16183](https://arxiv.org/html/2511.16183v1).
- **SoccerTrack v2**: posiciones del sistema BePro con cámara panorámica que cubre todo el campo — [paper](https://arxiv.org/html/2508.01802). El balón es "a reconstruction rather than a per-frame measurement… anchored at the annotated events and interpolated in straight constant-speed lines" — [HF](https://huggingface.co/datasets/atomscott/soccertrack-v2).
- **SkillCorner** (ya en uso): broadcast a 10 fps, `is_detected` para distinguir detectado de extrapolado — [README](https://github.com/SkillCorner/opendata).

### Inferences
- En SoccerTrack v2, v6 debe recibir **solo jugadores**. Hay que quitar `ball/` y `ball@loc`/`ball@speed` del XML, porque el balón lleva el timing de los mismos eventos que generan la etiqueta. Esa ejecución mide justo el modo "sin balón" de v6, que es el caso extremo de hueco.
- En PFF hay dos variantes complementarias: `balls` raw con sus huecos (lo más parecido al tracker del reto) y `ballsSmoothed` (techo con balón casi continuo). La diferencia entre ambas mide cuánto pierde v6 por los huecos.

### Gaps
- No hay documentación pública de si `ballsSmoothed` interpola los frames sin balón. Tampoco de qué significa `possession_event` frente a `home_ball`. Ambas cosas están en los PDF de especificación que vienen con los datos.
- No hay cifra pública de frames sin balón en partido completo para PFF: el 93,4 % se mide solo en ventanas de evento.

## 3. PFF/Gradient WC2022: términos reales y acceso

### Takeaway
El acceso es un formulario web, y la página de 2025 dice "instant access". No hay licencia abierta publicada. El formulario remite a unos Terms of Use cuyo texto no se pudo leer, y los papers que usan los datos declaran que no pueden redistribuirlos. Para validar sirve si se documenta cómo reproducirlo. Para entrenar depende de esos términos.

### Cited Findings
- **Páginas de Gradient**:
  - Post de 2025: "Simply fill out the form below to gain instant access to this invaluable dataset" — [Gradient 2025](https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset).
  - Única petición de uso en el post original: etiquetar @PFF_FC en redes — [Gradient](https://www.gradientsports.com/blog/pff-fc-release-2022-world-cup-data).
- **Formulario**: "By submitting this form you confirm that you agree to our Terms of Use" (fragmento del buscador) — [fc.pff.com](https://fc.pff.com/). La página de términos solo muestra "© 2025 All Rights Reserved" vía fetch — [terms-conditions](https://www.gradientsports.com/terms-conditions). La sesión del 2026-11-01 solo encontró en ella un acuerdo de API Sandbox de 2022, que es otro servicio.
- **Lo que declaran quienes los usan**:
  - "obtained from the provider on request under its terms of use and are not redistributed"; "obtained under the provider's research licence" — [arXiv 2609.24040](https://arxiv.org/html/2609.24040v1).
  - "subject to third-party licensing restrictions and cannot be publicly shared" — [arXiv 2603.28916](https://arxiv.org/html/2603.28916v1).
- **Derivado SoccerNet-GAR**: licencia "other", NDA personalizado — [HF](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR).
- **Formato**:
  - Un `{game_id}.jsonl.bz2` por partido, más `metadata.csv`, `rosters.csv`, `events.json` y los PDF de especificación — [Gradient](https://www.gradientsports.com/blog/pff-fc-release-2022-world-cup-data).
  - Kloppy usa `pff_metadata_{id}.json` y `pff_rosters_{id}.json`; el 10517 es Argentina–Francia — [kloppy docs](https://kloppy.pysport.org/user-guide/loading-data/pff/).
- **España**: 7-0 a Costa Rica, 1-1 con Alemania, 1-2 con Japón, y octavos contra Marruecos — [Olympics.com](https://olympics.com/en/news/fifa-world-cup-2022-spain-results-scores-and-standings).

### Inferences
- Siguiente paso: rellenar el formulario hoy y leer los términos que aparezcan al hacerlo. Si permiten el uso interno para investigación, se valida v6 en los 4 partidos de España más una muestra aleatoria. En el repo se dejan el script y la lista de `game_id`, sin datos.
- Que la expresión "research licence" y el NDA del derivado restrinjan la redistribución es coherente con publicar solo métricas agregadas.

### Gaps
- No se ha leído el texto de los Terms of Use. Hay snapshot en Wayback del 2026-08-01, pero la herramienta no lo abre — [snapshot](http://web.archive.org/web/20260801150927/https://www.gradientsports.com/terms-conditions).
- No se sabe si tras el formulario llega una descarga directa, un enlace por email o una carpeta de Drive.
- Los `game_id` de España están en `metadata.csv`, que solo se ve tras el acceso.

## 4. SoccerTrack v2: detalle técnico y regla de etiquetas

### Takeaway
El formato encaja con v6:
- 22 jugadores por frame a 25 fps, en metros (105 x 68, origen en el centro).
- Lado `left`/`right` por mitad, con `player_id` persistente.

La descarga útil sin vídeo ronda 2,3 GB y no exige aceptar más que el gate automático. Hay que tener cuatro detalles en cuenta:
- La rejilla es de 1,05 x 0,68 m.
- El partido M9 (132831) tiene la calibración degenerada.
- Hay que fijar la revisión `v1.2`.
- No existe lector en kloppy ni en databallpy.

### Cited Findings
- **Formato real del GSR** — [format-gsr.md](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-gsr.md):
  - Es SoccerNet-COCO: `image_id` en string (prefijo + frame) y la posición en `bbox_pitch.x_bottom_middle`/`y_bottom_middle`.
  - `attributes` = {role, jersey, team, player_id}.
  - Ocupa ~2,7 GB por mitad y pide ~20 GB de RAM con `json.load`, así que hay que leerlo en streaming.
- **Precisión** — [paper](https://arxiv.org/html/2508.01802):
  - Rejilla de 1,05 x 0,68 m (el XML bruto da coordenadas 0-1 con 2 decimales).
  - Calibración autoconsistente a unos 2-3 m.
  - "No quantitative audit of positional accuracy".
- **Error de M9**: dos keypoints intercambiados (RMS de 1260 px); regenerarlo figura como pendiente — [data_corrections](https://github.com/AtomScott/SoccerTrack-v2/blob/main/data_corrections/README.md).
- **XML bruto**: por frame trae `frameNumber`, `matchTime`, `eventPeriod`, `ballStatus`, `player@loc`, `ball@loc`. Los metadatos traen `frameStart`/`frameEnd`/`fps` por periodo y `teamId` — [dataclass.py](https://github.com/AtomScott/SoccerTrack-v2/blob/main/scripts/tracking_data_utils/dataclass.py). En el EDA se ve `ballStatus = BALLOUT` en el frame 251 — [eda.ipynb](https://github.com/AtomScott/SoccerTrack-v2/blob/main/notebooks/eda.ipynb).
- **Eventos BAS**:
  - 12 clases: Pass 9.315, Drive 8.255, High Pass 1.157, Out 771, Cross 394, Throw In 385, Block 353, Tackle 307, Shot 266, Free Kick 150, Goal 44, Header 31 — [paper](https://arxiv.org/html/2508.01802).
  - "Out" es sintetizado y "marks play going dead"; una quinta parte sigue a faltas — [paper](https://arxiv.org/html/2508.01802).
  - `player_nodes.csv` tiene 79 tipos de evento del proveedor, entre ellos `possession`, `cornerKick` y `goalKick*` — [raw_event_data.ipynb](https://github.com/AtomScott/SoccerTrack-v2/blob/main/notebooks/raw_event_data.ipynb).
- **Tamaños** [measured] — [árbol HF](https://huggingface.co/api/datasets/atomscott/soccertrack-v2/tree/main?recursive=true):
  - `bas` 4,56 MB, `ball` 2,60 MB, `raw` 2,61 GB (con `tracker_box_data.xml` de 215-318 MB por partido).
  - `gsr` 55,95 GB, `videos` 82,68 GB.
- **Versión y splits**:
  - En v1.2 el frame k del GSR es el frame k del vídeo, y los eventos traen `frame` — [HF](https://huggingface.co/datasets/atomscott/soccertrack-v2).
  - Test: M7 y M9. Validación: M2 y M10.
- **Lectores**: kloppy no tiene lector BePro/SoccerTrack [measured] — [serializers](https://github.com/PySport/kloppy/tree/master/kloppy/infra/serializers/tracking). databallpy solo lee inmotio, metrica y tracab [measured] — [parsers](https://github.com/Alek050/databallpy/tree/main/databallpy/data_parsers/tracking_data_parsers).

### Inferences
- **Descarga mínima**, tras `huggingface-cli login` y aceptar el gate:
  ```python
  snapshot_download("atomscott/soccertrack-v2", repo_type="dataset", revision="v1.2",
      allow_patterns=["bas/*", "raw/*/*_tracker_box_*.xml", "raw/*/*_player_nodes.csv",
                      "raw/*/*_padding_info.csv", "README.md", "CHANGES_v1_2.md"])
  ```
  `ball/*` sobra para validar sin fuga.
- **Regla de etiquetas propuesta** (por mitad, en frames de vídeo v1.2):
  - **D**:
    - Preferente: `ballStatus` del XML, alineado con `frameStart`/`padding_info`.
    - Respaldo, con BAS: muerto desde el inicio de la mitad hasta el primer evento, desde cada `Out` o `Goal` hasta el siguiente evento, y en corners y saques de puerta recuperados de `player_nodes.csv`.
  - **H/A**: el `team` del último evento con `frame` ≤ f.
    - Un `Tackle` cambia la posesión.
    - Un `Block` cuenta solo si el siguiente evento es del mismo equipo; si no, se marca como disputado.
    - Shot y Goal en el mismo frame se ordenan con Goal al final.
  - **Conversión a local/visitante**: left/right a H/A con `player_id`/`teamId`, porque el lado cambia entre mitades.
  - **Métrica**:
    - Se aplica el mismo filtro de "controlada" que al ctrl y se mide con la misma métrica de siempre, a la frecuencia de v6.
    - M9 se informa aparte.
    - Opcionalmente, se excluye ±0,5 s alrededor de cada cambio.
- **Cobertura de eventos**: con unos 23 eventos por minuto (uno cada ~2,6 s), la etiqueta reconstruida tiene huecos cortos. Es una buena base para la posesión de equipo, menos para la individual.

### Gaps
- No se sabe la frecuencia nativa del XML ni si los 25 fps del GSR están interpolados.
- No están claros los valores completos de `ballStatus` (solo se vio `BALLOUT`), ni si ese campo lo deriva BePro de sus propios eventos.
- No se leyó `CHANGES_v1_2.md` (está detrás del gate), ni se sabe la proporción de eventos con `visibility: "not shown"`.

## 5. Orden de descarga para validar v6

### Takeaway
Primero SoccerTrack v2 (10 partidos, hoy mismo). Después, la segunda etiqueta de PathCRF sobre IDSSE (sin permisos). En paralelo, el formulario de PFF para tener 64 partidos con huecos de balón broadcast.

### Inferences

| Prioridad | Dataset | Acceso hoy | Lector (esfuerzo) | Regla de etiquetas (esfuerzo) | Qué mide en v6 | Siguiente paso |
|---|---|---|---|---|---|---|
| 1 | **SoccerTrack v2** (10 partidos, CC BY 4.0) | Gate automático; ~2,3 GB | Parser propio en streaming del XML bruto (215-318 MB por partido) o del GSR; mapear left/right a H/A con `teamId`. Unas horas | `ballStatus` para D y último `team` BAS para H/A, con las reglas de Tackle y Block. Unas horas, más medir los valores de `ballStatus` | Generalización a otro proveedor y nivel, sin balón (hueco del 100 %) | Aceptar el gate, bajar `bas/` y `raw/*xml`, contar los valores de `ballStatus` en un partido |
| 2 | **PathCRF + ELASTIC** (IDSSE) | Abierto; 187 + 22,6 MB | Ya existe: `datatools/preprocess.py`. Menos de una hora | Jugador a equipo, más el BallStatus de IDSSE para D. Menos de una hora | Sensibilidad de la métrica a la definición de "controlada"; error temporal en cambios (3 partidos anotados a mano) | Clonar el repo y generar las etiquetas por frame de los 7 partidos |
| 3 | **PFF/Gradient WC2022** (64 partidos) | Formulario, "instant access"; ~50 GB | kloppy `pff.load_tracking`, más un lector propio de `balls` raw para los huecos reales. De horas a un día | `home_ball` arrastrado entre eventos; D con OUT/END hasta KICKOFF/OTB. Unas horas | Tracker broadcast con huecos reales; España (4 partidos) | Rellenar el formulario hoy, leer los términos y bajar primero los partidos de España |
| 4 | **FOOTPASS** (54 partidos, sin balón) | NDA, aprobación manual | Pendiente del formato | Último equipo con evento; sin D | Solo H/A, broadcast | Pedir acceso si se quiere para después de la entrega |
| 5 | **GRF simulated** (3.000) | Abierto, sin licencia | CSV de esquema simple. Horas | `ON_BALL`, mapeado a equipo | Robustez en dominio sintético | Solo si sobra tiempo; no incluirlo en el pipeline entregado |

- **Para entrenar**, además de validar, valen SoccerTrack v2 (CC BY 4.0) y PFF solo si sus términos lo permiten. FOOTPASS vale solo con NDA. PathCRF/ELASTIC y GRF quedan para validación o diagnóstico.
- **Inspección por los organizadores**: con SoccerTrack v2 basta citarlo y fijar la revisión `v1.2`. Con PFF hay que documentar el formulario y los `game_id`. En ningún caso se suben datos al repo.

### Gaps
- Ninguna de estas validaciones está medida todavía. Las cifras de v6 en cada dataset salen del siguiente paso de cada fila.
