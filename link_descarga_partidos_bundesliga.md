# Descarga de los 7 partidos de la Bundesliga (IDSSE / DFL)

Los datos no están en este repositorio porque son demasiado pesados: unos 2,5 GB en total, y cada
archivo de tracking pesa entre 330 y 400 MB (GitHub rechaza archivos de más de 100 MB).

No forman parte de los datos del reto. Son datos abiertos de otros partidos, con licencia
**CC-BY 4.0**, que debe citarse al usarlos.

## Enlaces de descarga

- **Fuente oficial (figshare):** <https://doi.org/10.6084/m9.figshare.28196177>
- **Réplica en Hugging Face:** <https://huggingface.co/datasets/pysport/idsse-data>

Archivos por partido (temporada 2022/23):

| ID | Competición | Fecha | Partido | Resultado | Información | Eventos | Tracking |
|---|---|---|---|---|---|---|---|
| J03WMX | 1. Bundesliga | 2023-05-27 | 1. FC Köln – FC Bayern München | 1:2 | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml) | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WMX.xml) | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml) |
| J03WN1 | 1. Bundesliga | 2023-05-27 | VfL Bochum 1848 – Bayer 04 Leverkusen | 3:0 | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml) | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WN1.xml) | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml) |
| J03WOH | 2. Bundesliga | 2022-08-26 | Fortuna Düsseldorf – SSV Jahn Regensburg | 4:0 | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml) | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOH.xml) | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml) |
| J03WOY | 2. Bundesliga | 2022-09-10 | Fortuna Düsseldorf – F.C. Hansa Rostock | 3:1 | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOY.xml) | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOY.xml) | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOY.xml) |
| J03WPY | 2. Bundesliga | 2022-10-15 | Fortuna Düsseldorf – 1. FC Nürnberg | 0:1 | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml) | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WPY.xml) | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml) |
| J03WQQ | 2. Bundesliga | 2022-11-05 | Fortuna Düsseldorf – FC St. Pauli | 1:0 | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WQQ.xml) | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WQQ.xml) | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WQQ.xml) |
| J03WR9 | 2. Bundesliga | 2022-11-11 | Fortuna Düsseldorf – 1. FC Kaiserslautern | 1:2 | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WR9.xml) | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WR9.xml) | [xml](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WR9.xml) |

## Documentación de los datos

- **Paper:** Bassek, M., Rein, R., Weber, H. y Memmert, D. (2025). *An integrated dataset of
  spatiotemporal and event data in elite soccer*. Scientific Data, 12, 195.
  DOI: [10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y) ·
  [texto completo en PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/)
- **Lectura con Python:** [kloppy](https://github.com/PySport/kloppy) (proveedor `sportec`) y
  [floodlight](https://floodlight.readthedocs.io/en/latest/modules/io/dfl.html).

Lo esencial del dataset:

- **Tracking:** óptico TRACAB a 25 Hz, con jugadores y balón.
- **Formato:** tres XML por partido (información del partido, eventos y posiciones), en el
  formato DFL/Sportec.
- **`BallPossession` por frame:** 1 = local, 2 = visitante.
- **`BallStatus` por frame:** 0 = balón muerto, 1 = balón vivo.
- **Eventos:** 11.137 en total. Sus relojes son independientes de los del tracking, así que pueden
  estar desfasados varios segundos.

En kloppy, usa `only_alive=False` al cargar el tracking. El valor por defecto (`True`) descarta los
frames de balón muerto.
