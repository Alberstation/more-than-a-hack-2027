# PFF FC / Gradient WC2022 y otros datasets de tracking broadcast con posesión

Fecha de consulta: 2026-11-13. [measured] = comprobado leyendo directamente el fichero raw o la página fuente (sin intermediarios de búsqueda).

## 1. Términos/licencia y vía de acceso de PFF/Gradient WC2022

### Takeaway
No hay una licencia abierta publicada (ni CC ni MIT). El acceso es por formulario y el post de 2025 promete "acceso instantáneo". Los blogs no ponen restricciones explícitas, pero el formulario remite a unos "Terms of Use" cuyo texto no pude leer, y los papers que usan los datos hablan de "terms of use", de "research licence" y de "third-party licensing restrictions" que impiden redistribuir. Que sea "inspeccionable por organizadores" depende de que cada organizador rellene el formulario: no hay URL pública de descarga.

### Cited Findings
- Post original (rebrandeado como Gradient, datado en septiembre de 2024): el acceso es "simply fill out the form below and follow the easy-to-use instructions". Única petición de uso: "If you intend to publish your findings on social media, please tag us @PFF_FC". No hay texto de licencia en la página, solo enlaces a /terms-conditions y /privacy-policy — [Gradient blog](https://www.gradientsports.com/blog/pff-fc-release-2022-world-cup-data); copia antigua en [blog.fc.pff.com](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data)
- Post "enhanced" (13-feb-2025): "Simply fill out the form below to gain instant access to this invaluable dataset". No menciona aprobación manual ni Google Drive, y no pone restricciones de uso (solo invita a etiquetar @PFF_FC) — [Gradient blog](https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset); [blog.fc.pff.com](https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset)
- fc.pff.com, junto al formulario: "By submitting this form you confirm that you agree to our Terms of Use" (fragmento del buscador; no se abrió la página entera) — [fc.pff.com](https://fc.pff.com/)
- La página de Terms & Conditions de Gradient no muestra cuerpo legible por fetch: solo "© 2025 All Rights Reserved". Hay snapshot en Wayback del 2026-08-01, pero la herramienta no puede abrir web.archive.org — [gradientsports.com/terms-conditions](https://www.gradientsports.com/terms-conditions); [snapshot](http://web.archive.org/web/20260801150927/https://www.gradientsports.com/terms-conditions)
- arXiv 2609.24040 (G2G), Data availability: "The raw PFF FC World Cup tracking and event data are obtained from the provider on request under its terms of use and are not redistributed by us". Competing interests: "The World Cup data were obtained under the provider's research licence" — [arXiv 2609.24040](https://arxiv.org/html/2609.24040v1)
- arXiv 2603.28916 (Karakuş & Arkadaş): "The dataset was publicly released by PFF FC" (enlaza al blog). Su Data availability dice: "The tracking and event data used in this study are subject to third-party licensing restrictions and cannot be publicly shared" — [arXiv 2603.28916](https://arxiv.org/html/2603.28916v1)
- arXiv 2506.06666 (mismos autores): la página abs no nombra al proveedor, solo "synchronised event and tracking data" del Mundial 2022. Según el buscador, describe "29.97Hz player and ball positions" — [arXiv 2506.06666](https://arxiv.org/abs/2506.06666)
- arXiv 2606.19931: según el snippet del buscador, lista "PFF FC (30 fps)". La versión HTML da 404 y el PDF no se pudo extraer — [arXiv 2606.19931 PDF](https://arxiv.org/pdf/2606.19931)
- La documentación de kloppy dice que los datos "can be requested via" el blog de PFF. No menciona ninguna carpeta de Google Drive ni pone nota de licencia — [kloppy docs PFF](https://kloppy.pysport.org/user-guide/loading-data/pff/)
- SoccerNet-GAR (derivado de estos datos) está en HF con licencia "other" y acceso con gating: request form más un NDA personalizado (enlazado en Google Drive) que hay que aceptar. Lo atribuye a "Gradient Sports (formerly PFF FC)" [measured: tarjeta HF] — [HF SoccerNet-GAR](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR)

### Inferences
- El formulario da acceso inmediato (según Gradient), así que el plazo de 4 días no es el riesgo. El riesgo está en la regla de "inspeccionable": no podemos redistribuir los datos, y los organizadores tendrían que registrarse con su propio email. Lo prudente es usarlo solo como validación secundaria y documentar en el repo la ruta de obtención y los game_id, sin subir datos.
- Que SoccerNet exija un NDA para redistribuir un derivado, y que G2G hable de "research licence", indica que los términos reales restringen la redistribución y probablemente el uso comercial. Es coherente con "third-party licensing restrictions".

### Gaps
- No pude leer el texto de los Terms of Use: el cuerpo no se renderiza y Wayback está bloqueado para la herramienta. Para leerlo hay que abrir en un navegador gradientsports.com/terms-conditions o el snapshot de 2026-08-01.
- No verifiqué si, tras el formulario, se envía un enlace por email, una carpeta de Google Drive o una descarga directa (no se envió el formulario, por restricción). Tampoco encontré la "carpeta de Google Drive enlazada desde kloppy": la documentación actual solo enlaza al blog.
- No vi los PDFs de especificación ni los términos que vienen con los datos.

## 2. Formato de datos y comportamiento del loader de kloppy

### Takeaway
Hay un `{game_id}.jsonl.bz2` por partido (~30 Hz, 29.97). El ball status y la posesión salen de `game_event`: `home_ball` y los tipos OUT/END/KICKOFF/OTB. Kloppy solo usa `ballsSmoothed` y los jugadores smoothed, e ignora el `balls` raw y `possession_event`.

### Cited Findings
- Ficheros: tracking `{game_id}.jsonl.bz2` (uno por partido), eventos en `events.json`, y `metadata.csv`, `players.csv`, `rosters.csv`, `competitions.csv`, más PDFs de especificación y grades — [Gradient blog](https://www.gradientsports.com/blog/pff-fc-release-2022-world-cup-data). El post de 2025 es inconsistente: habla también de una carpeta "Event Data" por partido — [Gradient blog 2025](https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset)
- Kloppy usa metadata y rosters por partido (`pff_metadata_10517.json`, `pff_rosters_10517.json`, `pff_10517.jsonl.bz2`; el 10517 es Argentina–Francia). Parámetros: `coordinates`, `sample_rate`, `limit`, `only_alive` — [kloppy docs](https://kloppy.pysport.org/user-guide/loading-data/pff/)
- Loader de kloppy [measured: código fuente raw] — [pff.py](https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/tracking/pff.py):
  - `frame_rate = metadata["fps"]`. El timestamp sale de `periodElapsedTime`.
  - El balón se lee solo de `ballsSmoothed` (el `balls` raw no se usa). Si falta `x` o `y`, la coordenada queda en `None`. Los jugadores salen de `homePlayersSmoothed`/`awayPlayersSmoothed`, emparejados por `jerseyNum`.
  - Ball state: empieza en DEAD. Pasa a DEAD con `OUT` (BALL_OUT) y `END` (PERIOD_END), y a ALIVE con `FIRSTKICKOFF`…`FOURTHKICKOFF` y `OTB` (ON_THE_BALL). Los demás eventos no lo cambian.
  - Ball owning team: `home_team if game_event["home_ball"] else away_team`, y solo cuando `home_ball` no es nulo. El valor se arrastra hasta el siguiente evento. `possession_event` no se usa.
  - Los periodos 3 y 4 van a `et_frames`. Flags: `BALL_OWNING_TEAM | BALL_STATE`.
- Frame rate: "our copy stores smoothed tracking at 29.97 Hz"; "about 50 GB and on the order of 10⁷ frames" — [arXiv 2609.24040](https://arxiv.org/html/2609.24040v1). También "29.97 Hz" en [arXiv 2603.28916](https://arxiv.org/html/2603.28916v1). Un proyecto externo dice "30 Hz" y submuestrea a 5 Hz — [wc2026-tracking-transformer](https://stranger9977.github.io/wc2026-tracking-transformer/methodology.html)

### Inferences
- La posesión de kloppy es event-driven: un escalón que se mantiene entre eventos, sin neutral ni disputa. Sirve como etiqueta de equipo por frame comparable a nuestro modelo, siempre que se trate igual que el ctrl (mismo filtrado de balón vivo).
- Si queremos los huecos reales de balón, hay que leer `balls` (raw) por nuestra cuenta, porque kloppy solo da el smoothed, que probablemente rellena huecos.

### Gaps
- No hay documentación pública que confirme si `ballsSmoothed` interpola los frames sin balón, ni qué semántica tiene `possession_event` frente a `home_ball`. Eso estará en los PDFs de especificación, que no vi.
- El tamaño exacto por partido no se sabe; solo hay el total de ~50 GB según G2G.

## 3. Partidos de España y frames sin balón

### Takeaway
España juega 4 de los 64 partidos. La única cifra pública de cobertura del balón es la de SoccerNet-GAR: balón presente en el 93.4 % de los frames de sus ventanas por evento.

### Cited Findings
- España en Qatar 2022: 7–0 a Costa Rica, 1–1 con Alemania y 1–2 contra Japón (Grupo E), y octavos contra Marruecos — [Olympics.com](https://olympics.com/en/news/fifa-world-cup-2022-spain-results-scores-and-standings); [Wikipedia Grupo E](https://en.wikipedia.com/wiki/2022_FIFA_World_Cup_Group_E). El 0–0 y la derrota en penaltis ante Marruecos vienen del conocimiento del buscador, no de una fuente abierta.
- SoccerNet-GAR (tracking PFF): "Ball data is present in 93.4% of frames"; el 85.9 % de los eventos tiene el balón completo en su ventana; balón 3D con z en [-8, 25] m; existe un flag `tracking_valid` [measured: tarjeta HF] — [HF SoccerNet-GAR](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR)
- G2G cita como limitación "intermittent ball visibility" y "smoothing artifacts", y pone en cuarentena los vuelos no resueltos — [arXiv 2609.24040](https://arxiv.org/html/2609.24040v1)

### Inferences
- Como el tracking viene de broadcast, el balón se pierde (~6,6 % de frames sin balón en ventanas de evento, que son cercanas a la acción, así que en el partido completo la ausencia será probablemente mayor). Es el dominio más parecido a un tracker automático que pierde el balón.

### Gaps
- No se sabe el `game_id` de los partidos de España. Hay que sacarlo de `metadata.csv` tras el acceso.
- No hay estadística pública de frames sin balón por partido completo.

## 4. Otros datasets con tracking automático/broadcast y señal de posesión

### Takeaway
No encontré ninguna release abierta nueva en 2025–2026 con tracking broadcast y posesión. SkillCorner opendata sigue siendo la única con licencia MIT, y hoy su README habla de 10 partidos de A-League 2024/25 a 10 fps, no de 20. SoccerNet-GAR usa tracking PFF pero exige NDA.

### Cited Findings
- SkillCorner opendata: MIT, "10 matches… a sample of 2024/2025 league matches in the Australian A-League", tracking a 10 fps, dict `possession` con `player_id` y `group` (home/away), `is_detected` para jugadores (detectado o extrapolado), Body Pose a 25 fps para 2 partidos en HF, spec de dynamic events del 2025-02-16 [measured: README] — [GitHub SkillCorner/opendata](https://github.com/SkillCorner/opendata). Esto choca con el dato previo de "20 partidos": hay que comprobarlo contando los ficheros del repo.
- SoccerNet-GAR: 87 939 eventos de 64 partidos WC2022 (45/9/10 en train/valid/test), 1 485 008 frames de tracking, clips de 16 o 144 frames por evento (no partidos completos), ~29.97 fps, sin campo de posesión, gating con NDA — [HF](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR); [arXiv 2511.12606](https://arxiv.org/pdf/2511.12606)
- Gradient no tiene releases gratuitas nuevas en 2026: sus anuncios de 2026 son comerciales (Marquee, Servette, financiación) — [Gradient blog](https://www.gradientsports.com/blog/); [Marquee](https://www.gradientsports.com/blog/gradient-data-to-support-marquees-ai-native-football-platform)
- Mills et al. (Sports Engineering): auto-eventing de pases, recepciones e intercepciones con tracking derivado de broadcast a partir de la distancia jugador–balón. Es un paper de método, sin dataset abierto aparente — [SHU repo](https://shura.shu.ac.uk/37410/1/Mills-AutomaticEventDetection%28VoR%29.pdf)
- Estudio de duelos con Hawk-Eye WC2022 (330 duelos, etiqueta de último toque): no vi que publique los datos — [Sports Engineering 2025](https://link.springer.com/article/10.1007/s12283-025-00518-3)

### Inferences
- Para validar fuera del dominio con un tracker que pierde el balón, el orden práctico es: (1) SkillCorner, que es abierto, MIT, tiene posesión y está ya integrado; (2) PFF WC2022, de acceso inmediato pero sin redistribución, solo como evidencia complementaria con instrucciones para reproducir.

### Gaps
- No revisé a fondo Sportlogiq, las muestras públicas de Second Spectrum, Bepro, TRACAB/ChyronHego ni las secuencias largas de SoccerNet-GSR (límite de llamadas). En las búsquedas no apareció ninguna release abierta con posesión de estos proveedores.
