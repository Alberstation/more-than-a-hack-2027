# SoccerTrack v2 como validación externa de un modelo de posesión por frame (local / visitante / balón muerto)

Fuentes principales: [ficha HF](https://huggingface.co/datasets/atomscott/soccertrack-v2), [API HF](https://huggingface.co/api/datasets/atomscott/soccertrack-v2), [árbol HF](https://huggingface.co/api/datasets/atomscott/soccertrack-v2/tree/main?recursive=true), [paper arXiv 2508.01802](https://arxiv.org/html/2508.01802), [repo GitHub](https://github.com/AtomScott/SoccerTrack-v2).
[measured] = comprobado por mí el 2026-11-13 contra la API pública de HF o GitHub, sin descargar datos ni aceptar el gate.

## 1. Tracking: frecuencia, coordenadas, cobertura e identidad de equipo

### Takeaway
GSR da ~22 jugadores por frame a 25 fps, en metros sobre un campo de 105x68 con origen en el centro, durante el partido entero. Cada registro trae el lado (left/right, fijo por mitad), el dorsal y un `player_id` entero persistente. Hay dos limitaciones: la precisión es baja (rejilla de 1,05 x 0,68 m) y la exactitud no está auditada.

### Cited Findings
- Hay 10 partidos universitarios grabados con cámara panorámica fija (BePro) que cubre todo el campo. La ficha de HF dice "934 minutes" — [README HF](https://huggingface.co/datasets/atomscott/soccertrack-v2). El paper da 932 minutos, 1,4 M de frames y 30,7 M de registros jugador-frame; las duraciones van de 90 a 97 min por partido. Las dos fuentes no coinciden (934 frente a 932) — [paper](https://arxiv.org/html/2508.01802).
- Las posiciones las proporciona el sistema BePro, "synchronised to video timestamps". Se etiquetan por frame a 25 fps y salen ~22 registros por frame ("players on the pitch"). Los porteros tienen su propio rol. El paper dice que el GSR "no ball or referee annotations" — [paper](https://arxiv.org/html/2508.01802). En cambio, la especificación del repo contempla `role: referee`, así que hay que comprobar qué roles aparecen en los ficheros — [format-gsr.md](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-gsr.md).
- Las coordenadas son métricas: 105 x 68 m con origen en el círculo central. x crece hacia la mitad derecha vista desde la cámara principal; y crece hacia la banda cercana a la cámara — [README HF](https://huggingface.co/datasets/atomscott/soccertrack-v2). El repo afirma lo contrario para y ("towards the top"), así que el signo de y hay que comprobarlo con datos — [format-gsr.md](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-gsr.md).
- El formato real de los ficheros no es el que describe la especificación. Es un objeto SoccerNet-COCO (`info`, `images`, `annotations`, `categories`) con estas particularidades — [format-gsr.md, "As shipped"](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-gsr.md):
  - `image_id` es un string ("3000001" = prefijo + frame de 6 dígitos que empieza en 1).
  - `attributes` = {role, jersey (str), team: "left"/"right", player_id (int, p. ej. 170959)}.
  - La posición va en `bbox_pitch.x_bottom_middle/y_bottom_middle`.
  - `info.frame_rate: 25` y `seq_length: 67650` (45 min).
- Los ficheros GSR ocupan ~2,7 GB por mitad. Cargarlos con `json.load` pide unos 20 GB de RAM, así que hay que leerlos en streaming (ijson o similar) — [format-gsr.md](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-gsr.md).
- El lado de equipo se fija por mitad según la dirección de ataque en el saque inicial, así que left/right puede invertirse entre mitades — [format-gsr.md](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-gsr.md); [paper](https://arxiv.org/html/2508.01802).
- Limitaciones que reconoce el propio paper — [paper](https://arxiv.org/html/2508.01802):
  - Las posiciones están cuantizadas en una rejilla de 1,05 x 0,68 m.
  - La calibración solo es autoconsistente a unos 2-3 m.
  - "No quantitative audit of positional accuracy or identity consistency".
  - No se documenta cómo mantuvo el proveedor la identidad bajo oclusión.
- En el export bruto del proveedor (`raw/*_tracker_box_data.xml`) las posiciones son coordenadas normalizadas de 0 a 1 con 2 decimales (p. ej. `x=0.28, y=0.18`). Eso explica la rejilla 105/100 x 68/100 m. Cada jugador lleva `teamId` (p. ej. 9701 / 9834) — [eda.ipynb](https://github.com/AtomScott/SoccerTrack-v2/blob/main/notebooks/eda.ipynb), [convert_raw_to_pitch_plane.py](https://github.com/AtomScott/SoccerTrack-v2/blob/main/src/coordinate_conversion/convert_raw_to_pitch_plane.py).
- Hay un error de calibración en 132831 (M9, test): dos keypoints intercambiados. Con ellos la calibración da RMS 1260 px, y el GSR de ese partido se generó con esa calibración degenerada. En el repo figura como pendiente regenerarlo — [data_corrections/README.md](https://github.com/AtomScott/SoccerTrack-v2/blob/main/data_corrections/README.md).

### Inferences
- Para un modelo que parte del tracking de jugadores, el formato encaja: 25 fps, metros, 22 jugadores y equipo por jugador. Si el modelo trabaja a 10 fps (u otra frecuencia), basta con submuestrear.
- Los ficheros pueden cambiar entre versiones, así que conviene fijar la revisión (tag `v1.2`).
- left/right se convierte en home/away con `player_id`, que es persistente entre mitades, o con el `teamId` de los metadatos del XML bruto. Un equipo tiene el mismo id en las dos mitades, y así se elige cuál es "local".
- La rejilla de ~1 m y el error de 2-3 m bastan para la posesión de equipo. Para la posesión individual de cada jugador son poco precisos.
- Conviene excluir M9 (132831) del análisis, o tratarlo aparte, hasta confirmar que su GSR se ha regenerado.

### Gaps
- No sé la frecuencia nativa del XML bruto de BePro. Los metadatos tienen un atributo `fps` por periodo, pero no he podido leer el valor sin aceptar el gate. Tampoco sé si los 25 fps del GSR están interpolados.
- No sé el porcentaje de frames con menos de 22 jugadores (expulsiones, salidas del encuadre). El paper solo dice "~22".

## 2. Estructura en Hugging Face y descarga solo de anotaciones

### Takeaway
Las anotaciones útiles caben en unos 2,3 GB (bas + ball + XML bruto de tracking y metadatos), frente a 56 GB de GSR y 83 GB de vídeo. Se pueden bajar sin vídeo con `allow_patterns`, pero todo exige aceptar el gate de HF con sesión iniciada.

### Cited Findings
Tamaños por carpeta [measured] — [árbol HF](https://huggingface.co/api/datasets/atomscott/soccertrack-v2/tree/main?recursive=true). Revisión sha `325bd4c5…`, lastModified 2026-09-24:

| Carpeta | Ficheros | Tamaño total | Contenido |
|---|---|---|---|
| `bas/<id>/<id>_12_class_events.json` | 10 | 4,56 MB (410-519 KB c/u) | eventos 12 clases |
| `ball/<id>_{1st,2nd}_ball.npz` | 20 | 2,60 MB (~125-135 KB c/u) | balón reconstruido |
| `gsr/<id>/<id>_{1st,2nd}.json` | 20 | 55,95 GB (2,70-2,92 GB c/u) | tracking GSR |
| `mot/` | 21 | 3,59 GB | 10 txt de ~6 MB + 10 clips mp4 de 0,27-0,60 GB + README |
| `raw/<id>/` | 102 | 2,61 GB | ver detalle debajo |
| `videos/` | 20 | 82,68 GB | panorámicas 3,3-6,9 GB por mitad |
| raíz | 4 | ~24 KB | README, CHANGES_v1_1.md, CHANGES_v1_2.md, .gitattributes |

Detalle de `raw/<id>/`:
- `tracker_box_data.xml`, 215-318 MB por partido.
- `tracker_box_metadata.xml`, 7-25 KB.
- `player_nodes.csv`, 0,50-0,87 MB, con los eventos brutos del proveedor.
- `12_class_events.json`, 0,33-0,54 MB.
- `homography.npy`, `camera_intrinsics.npz`, `keypoints.json`, `padding_info.csv`.
- `mapx/mapy.npy`, 9-29 MB.
- 132831 y 132877 tienen además `8_class_events.json`.

Acceso:
- `gated: "auto"` [measured] — [API HF](https://huggingface.co/api/datasets/atomscott/soccertrack-v2). Sin autenticar, incluso `CHANGES_v1_2.md` y `/refs` responden "Access … is restricted. You must have access to it and be authenticated" [measured]. El README sí se puede leer sin autenticar [measured].
- El paper dice "available after accepting the dataset terms" — [paper](https://arxiv.org/html/2508.01802).
- El loader oficial usa `huggingface_hub.snapshot_download(..., allow_patterns=[f"gsr/{mid}/*", f"bas/{mid}/*", f"mot/{mid}/*", f"raw/{mid}/*", f"videos/{mid}*"])` — [load_from_hf.py](https://github.com/AtomScott/SoccerTrack-v2/blob/main/src/data_utils/load_from_hf.py).
- Versiones: tag `v1.2` (actual), con `v1.1` y `v1.0` disponibles. En v1.2 el frame k del GSR es el frame k del vídeo, y los eventos traen `frame`. Corrige un desfase de ~1 s en 132877 (M10). Se eliminan el tercer periodo no filmado de tres partidos y los eventos fuera de lo filmado; quedan 21.428 eventos — [README HF](https://huggingface.co/datasets/atomscott/soccertrack-v2).

### Inferences
- La descarga mínima recomendada (tras aceptar el gate con `huggingface-cli login`) es esta:
  ```python
  snapshot_download("atomscott/soccertrack-v2", repo_type="dataset", revision="v1.2",
      allow_patterns=["bas/*", "ball/*", "raw/*/*_tracker_box_*.xml", "raw/*/*_player_nodes.csv",
                      "raw/*/*_padding_info.csv", "README.md", "CHANGES_v1_2.md"])
  ```
  Ocupa unos 2,3 GB. Si hacen falta las coordenadas métricas ya sincronizadas al vídeo, se añade `gsr/*`, que son unos 56 GB y se procesan en streaming.
- Para leer `ball/*.npz` en Python, `np.load` debe ir con `allow_pickle=False`, porque los ficheros descargados no son de confianza.

### Gaps
- No he leído CHANGES_v1_2.md, porque exige el gate. Puede documentar cambios en `raw/` o en la sincronía XML↔vídeo.

## 3. Eventos (BAS): clases, actor, tiempos, sincronía. ¿Hay estado del balón o posesión?

### Takeaway
Hay 21.428 eventos de 12 clases. Cada uno trae el equipo que actúa ("left"/"right") y, en el 99,5 % de los casos, el jugador. En v1.2 traen también el `frame` exacto del vídeo y del GSR. BAS no tiene campo de posesión ni de estado del balón. Sí lo tiene el XML bruto del proveedor: un `ballStatus` por frame.

### Cited Findings
- Las 12 clases, con su número de eventos — [paper](https://arxiv.org/html/2508.01802):

  | Clase | Eventos |
  |---|---|
  | Pass | 9.315 |
  | Drive | 8.255 |
  | High Pass | 1.157 |
  | Out | 771 |
  | Cross | 394 |
  | Throw In | 385 |
  | Ball Player Block | 353 |
  | Player Successful Tackle | 307 |
  | Shot | 266 |
  | Free Kick | 150 |
  | Goal | 44 |
  | Header | 31 |

- Los eventos salen del event data de BePro con un mapeo "many-to-one, hand-built" de las etiquetas del proveedor. Se descartaron "possession spells", tackles y duelos fallidos, faltas y sustituciones. Las líneas de tiempo se revisaron a mano contra el vídeo — [paper](https://arxiv.org/html/2508.01802).
- "Out" es un evento sintetizado: "marks play going dead rather than a measured boundary crossing". Aproximadamente una quinta parte de los Out siguen a faltas. El balón sale del campo en el 3 % de los frames de M7 y nunca en M9 y M10 — [paper](https://arxiv.org/html/2508.01802).
- Cada evento lleva `team` ("left"/"right", con el mismo criterio de lado que el GSR en esa mitad) y `player_id`, que comparte identificadores con el GSR — [README HF](https://huggingface.co/datasets/atomscott/soccertrack-v2); [format-bas.md](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-bas.md).
- El campo `visibility` vale "visible" o "not shown" — [format-bas.md](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-bas.md).
- Hay un solo fichero por partido para las dos mitades, y `gameTime` ("1 - 12:30") indica la mitad — [format-bas.md](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-bas.md).
- Tiempos en v1.2: "`position` is milliseconds from the first decoded frame of the half's video, and `frame` is the video (and game state) frame the event falls on" — [README HF](https://huggingface.co/datasets/atomscott/soccertrack-v2). La documentación del repo, escrita antes de v1.2, avisa de que en versiones anteriores `position` era absoluto desde el inicio del partido; en 117093 la 2ª mitad va de 2.700.760 a 5.506.280 ms — [format-gsr.md](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-gsr.md), [soccertrack_v2.py](https://github.com/AtomScott/SoccerTrack-v2/blob/main/src/data_utils/soccertrack_v2.py). Hay que usar el campo `frame` de v1.2 y no recalcularlo.
- Particularidades de las etiquetas — [format-bas.md](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-bas.md):
  - Shot y Goal aparecen duplicados en el mismo instante.
  - `Throw In` se marca en el momento de la reanudación.
  - No hay clase Penalty: un penalti es un Shot.
  - Tampoco hay Corner ni Goal Kick entre las 12 clases.
- El XML bruto trae por frame `frameNumber`, `matchTime` (ms), `eventPeriod` (FIRST_HALF/SECOND_HALF) y `ballStatus`, además de `player@loc`, `player@speed`, `ball@loc` y `ball@speed` — [dataclass.py](https://github.com/AtomScott/SoccerTrack-v2/blob/main/scripts/tracking_data_utils/dataclass.py), [convert_raw_to_pitch_plane.py](https://github.com/AtomScott/SoccerTrack-v2/blob/main/src/coordinate_conversion/convert_raw_to_pitch_plane.py).
  - Los valores de `ballStatus` no están claros. La docstring dice "e.g., IN_PLAY, OUT_OF_PLAY", pero el valor que se ve en la salida del EDA es `BALLOUT` (frame 251, matchTime 0) — [eda.ipynb](https://github.com/AtomScott/SoccerTrack-v2/blob/main/notebooks/eda.ipynb).
  - Los metadatos traen por periodo `frameStart`, `frameEnd`, `matchTimeStart`, `matchTimeEnd` y `fps`, además de los equipos (`side`) y los jugadores con `teamId` y `shirtNumber` — [dataclass.py](https://github.com/AtomScott/SoccerTrack-v2/blob/main/scripts/tracking_data_utils/dataclass.py).
- El CSV bruto `player_nodes.csv` tiene 79 tipos de evento del proveedor. Entre ellos están `possession`, `turnoverWon`/`turnoverLost`, `intercept`, `clearance`, `cornerKick`, `goalKickSucceeded/Failed`, `freeKick`, `throwIn` y `penaltyKick` — [raw_event_data.ipynb](https://github.com/AtomScott/SoccerTrack-v2/blob/main/notebooks/raw_event_data.ipynb).

### Inferences
- La cobertura de eventos es densa: 21.428 en ~932 min, unos 23 por minuto, es decir, uno cada ~2,6 s. Drive aporta 8.255. Eso permite reconstruir la posesión de equipo con huecos cortos.
- El `ballStatus` del XML es la etiqueta directa de balón vivo/muerto que se buscaba. Antes de usarlo hay que medir dos cosas: qué valores toma y cómo se alinea con el vídeo (los offsets de `frameStart` y `padding_info.csv`).
- Los eventos brutos (`player_nodes.csv`) sirven para recuperar corners y saques de puerta, que no están entre las 12 clases.

### Gaps
- No sé si `ballStatus` lo genera BePro a partir de sus propios eventos o de forma independiente. Es probable que venga de los eventos, pero no está documentado.
- No sé la proporción de eventos `visibility: "not shown"`.

## 4. Balón: ¿detección real o reconstrucción? (riesgo de fuga)

### Takeaway
El balón es una reconstrucción. Está anclado en los eventos y se interpola en línea recta a velocidad constante entre ellos, así que no sirve como entrada de un modelo que se valida contra etiquetas derivadas de esos mismos eventos.

### Cited Findings
- "It is a reconstruction rather than a per-frame measurement: it is anchored at the annotated events and interpolated in straight constant-speed lines between anchors, so it carries event timing. Positions are clamped to the pitch rectangle in matches 132831 and 132877" — [README HF](https://huggingface.co/datasets/atomscott/soccertrack-v2).
- El paper dice lo mismo, añade que "most of which [anchors] are annotated events" y advierte de que puede inflar las puntuaciones de spotting con balón — [paper](https://arxiv.org/html/2508.01802).
- El repo tiene código de detección de balón (TrackNetX, BoT-SORT), pero no se publica ningún track de balón detectado — [árbol repo](https://github.com/AtomScott/SoccerTrack-v2/tree/main/src/ball_tracking).

### Inferences
- Para validar sin fuga, el modelo debe recibir solo las posiciones de los jugadores (GSR o XML), sin `ball/` ni `ball@loc`/`ball@speed` del XML, que probablemente tienen el mismo origen.
- Si nuestro modelo usa normalmente el balón como entrada, la validación en SoccerTrack v2 mide la variante "solo jugadores", y hay que declararlo así.

### Gaps
- No he confirmado que `ball@loc` del XML bruto sea la misma reconstrucción que `ball/`, aunque es probable.

## 5. Licencia, gate, nivel de juego y volumen

### Takeaway
Los datos son CC BY 4.0, con uso comercial permitido y obligación de atribución, y el código es MIT. Para descargar hay que aceptar los términos en HF, con aprobación automática. Son 10 partidos de una liga universitaria amateur, japonesa por lo que indican las afiliaciones, aunque el texto no nombra el país.

### Cited Findings
- Texto literal de LICENSE-DATA, extracto: "The SoccerTrack v2 *dataset* — comprising the panoramic 4K videos and the associated GSR, BAS, and MOT annotation files distributed via Hugging Face … and Google Drive — is released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0). … Attribution — you must give appropriate credit, provide a link to the licence, and indicate if changes were made. When you use the dataset in academic work, please cite the SoccerTrack v2 paper" — [LICENSE-DATA](https://github.com/AtomScott/SoccerTrack-v2/blob/main/LICENSE-DATA).
- README HF: "CC BY 4.0. The release is pseudonymised: players are identified by jersey numbers and integer identifiers, and no personal names are included. Please cite the SoccerTrack v2 paper (in submission; citation to be added)". Etiqueta `license:cc-by-4.0` [measured] — [README HF](https://huggingface.co/datasets/atomscott/soccertrack-v2).
- Gate: `"gated":"auto"` [measured] — [API HF](https://huggingface.co/api/datasets/atomscott/soccertrack-v2). No he visto el texto de los términos del gate, porque solo aparece al pedir acceso.
- Ética: hay aprobación del comité de ética universitario (Universidad de Tsukuba según el paper) y consentimiento que "covers public release and redistribution" — [paper](https://arxiv.org/html/2508.01802); [LICENSE-DATA](https://github.com/AtomScott/SoccerTrack-v2/blob/main/LICENSE-DATA).
- Nivel de juego: "university-level amateur teams", "single league". La financiación es de JST/JSPS y los autores son de la Universidad de Nagoya y Playbox — [paper](https://arxiv.org/html/2508.01802); [README repo](https://github.com/AtomScott/SoccerTrack-v2).
- Splits: test M7 (128057) + M9 (132831); validación M2 (117093) + M10 (132877); entrenamiento los seis restantes — [README HF](https://huggingface.co/datasets/atomscott/soccertrack-v2).
- Cita: Scott, Uchida, Kuroda, Kim, Fujii, arXiv:2508.01802 (2025) — [README repo](https://github.com/AtomScott/SoccerTrack-v2).

### Inferences
- CC BY 4.0 permite usarlo en el hackathon citando la fuente. Primero hay que comprobar que RULES.md admite datos externos con esta licencia.
- Es otro dominio: fútbol amateur, cámara panorámica y otro proveedor. Por eso sirve como prueba de generalización, no como comparación directa con datos profesionales.

### Gaps
- El país no se nombra en el texto.

## 6. Trabajos previos de posesión y loaders (kloppy / databallpy)

### Takeaway
No he encontrado ningún trabajo que derive la posesión por frame de SoccerTrack v2, ni loaders en kloppy o databallpy. Habría que escribir el parser, que es sencillo.

### Cited Findings
- Los deserializadores de tracking de kloppy son hawkeye, metrica, pff, secondspectrum, signality, skillcorner, sportec, statsperform y tracab; no hay ninguno de BePro ni de SoccerTrack [measured] — [kloppy tracking serializers](https://github.com/PySport/kloppy/tree/master/kloppy/infra/serializers/tracking).
- Los parsers de tracking de databallpy son inmotio, metrica y tracab [measured] — [databallpy parsers](https://github.com/Alek050/databallpy/tree/main/databallpy/data_parsers/tracking_data_parsers). databallpy calcula la posesión de equipo a partir de eventos sincronizados, pero solo para los proveedores que soporta — [databallpy PyPI](https://pypi.org/project/databallpy/).
- Repos en GitHub que mencionan SoccerTrack v2, ninguno sobre posesión [measured]:
  - kinesis-alpha-performance-lab: métricas físicas.
  - Synapsehack: scouting.
  - ProjetoPDS-AI: CNN de tracking.

  [búsqueda GitHub](https://github.com/search?q=soccertrack+v2&type=repositories).
- El loader oficial es `src.data_utils.soccertrack_v2.load_match`, con `gsr_frames()`, `bas_events()` y `Event.image_id` — [soccertrack_v2.py](https://github.com/AtomScott/SoccerTrack-v2/blob/main/src/data_utils/soccertrack_v2.py).
  - Con v1.2 conviene comprobar que `Event.image_id` no vuelve a restar la mitad, porque el `position` de v1.2 ya es relativo a la mitad. Lo más seguro es usar el campo `frame`.
  - La búsqueda web no arrojó otros resultados — [búsqueda](https://arxiv.org/abs/2508.01802v1).

### Inferences
- Lo práctico es un parser propio: streaming de GSR o XML para los jugadores, y BAS o `ballStatus` para las etiquetas.

### Gaps
- La búsqueda de código dentro de kloppy y databallpy pide autenticación. Me he basado en el listado de carpetas.

## Regla concreta para derivar las etiquetas por frame (propuesta)

Todo se hace por mitad y en frames de vídeo v1.2, que coinciden con los del GSR. `T` = equipo que actúa (`team`), que se convierte a home/away con `player_id`/`teamId`.

**A. Vivo/muerto**
- **Opción preferida:** usar `ballStatus` del XML bruto, una vez alineado con el frame de vídeo mediante `frameStart`/`padding_info`. Hay que medir antes qué valores toma (BALLOUT frente al valor de juego).
- **Si no hay `ballStatus`, usar BAS:**
  1. Muerto desde el inicio de la mitad hasta el primer evento, que es el saque.
  2. Muerto desde cada `Out` hasta el siguiente evento de reanudación. Si el siguiente evento es `Throw In` o `Free Kick`, se toma su frame. En caso contrario se toma el frame del siguiente evento, que cubre corners y saques de puerta, ausentes de las 12 clases. Se pueden recuperar de `player_nodes.csv` (`cornerKick`, `goalKick*`).
  3. Muerto desde cada `Goal` hasta el siguiente evento, que es el saque de centro.
  4. Opcional: frente a un Out sin reanudación clara, añadir un margen de 0,5-1 s.
- **Aviso:** una quinta parte de los `Out` son en realidad faltas, porque Out marca el balón parado, no solo la salida del campo. Eso favorece usarlo como etiqueta de "muerto".

**B. Posesión cuando el balón está vivo**
- `pos(f)` = `T` del último evento con `frame` ≤ f. Para esto Pass, High Pass, Cross, Drive, Header, Shot, Throw In y Free Kick cuentan igual.
- `Player Successful Tackle`: la posesión pasa a `T`, porque el tackle "wins possession".
- `Ball Player Block`: se asigna a `T` (el que bloquea) solo si el siguiente evento también es de `T`. Si no, se mantiene la posesión anterior. También se puede marcar como "disputado" y excluir de la métrica.
- Cambio entre eventos consecutivos de equipos distintos (pase fallido, intercepción): el cambio se produce en el frame del primer evento del nuevo equipo, igual que en el convenio del event data. Opcionalmente, se excluye una ventana de ±0,5 s alrededor del cambio para no penalizar la ambigüedad del timing.
- Los eventos duplicados en el mismo frame (Shot + Goal) se ordenan dejando Goal al final, para que el tramo siguiente quede muerto.

**C. Métrica**
- Accuracy y F1 por clase (home/away/muerto) frame a frame, a 25 fps o submuestreado al fps del modelo.
- Además, accuracy con tolerancia de ±N frames en los cambios.
- Excluir M9 (calibración) o informar de él por separado.
- Entradas del modelo: solo posiciones de jugadores, sin `ball/` ni `ball@*` del XML, para evitar la fuga.
