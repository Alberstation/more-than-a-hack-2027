# Bitácora

Registro compacto de todos los cambios e intentos, para continuar sin releer el proyecto.
Actualizar al cerrar cada bloque. Detalle numérico: `data/experiments/improve/` (local) y `docs/plan/continuar_m4_gpu.md`.

## Estado

- **Reto:** H/A/D por frame, Barça–Athletic, 199.319 frames. Ranking `spellF1`, desempate `macroF1`.
- **Mejor versión: v3** (`mixed_all3`). `python possession/train_v3.py` y `python possession/pipeline.py --out <equipo>.csv`.
- **CSV:** `evaluation/results/possession_v3.csv`, SHA-256 `e034f5d7…dd72ce`. **Respaldo:** `possession_v2.csv`, intacto.
- **CV v3:** spellF1 0,556 (huecos uniformes) y 0,596 (en muerto), frente a 0,490 y 0,510 de `ref`.
- **Metrica v3:** 0,601 y 0,643, frente a 0,541 y 0,558.
- **Partido del reto:** 309 posesiones de 11,2 s; 49/59 balones parados con reanudación.
- **5 min públicos (solo comprobación):** v3 0,645 con 14 posesiones; v2 0,463 con 21; reales 17. Muestra mínima, no decide nada.
- **Reproducibilidad:** dos procesos separados dan el mismo modelo; el CSV se regenera idéntico en local. Falta máquina limpia.
- **Mayor riesgo:** definición de posesión. Elegimos etiquetas `raw`; si el reto sigue `ctrl`, se dejan unas 0,035.

## Reglas

- Sin etiquetas a mano en el partido. Los 5 min públicos solo comprueban y explican; nunca entrenan ni deciden.
- Nada del reto a la nube ni al repo. `data/` y `evaluation/results/` están ignorados.
- El CSV debe regenerarse idéntico (regla 3). Entrega en CPU. Sin fechas salvo que se pidan.
- Aceptar: +0,03 pareado, ≥5/7 partidos, sin empeorar en Metrica. Ruido por partido: ~0,04.
- Medir cambios pequeños solo contra `mixed_all3` y `ref`, sin repetir todo el banco.

## Versiones

| Versión | Contenido | Resultado |
|---|---|---|
| Baseline | Equipo más cercano + suavizado | Fragmenta posesiones |
| v1 | Viterbi H/A/D, hueco sin evidencia, muerto heurístico | Superada |
| v2 | HGB, 46 características, IDSSE degradado, Viterbi (10, 40) | CV 0,4805; respaldo |
| **v3** | v2 + toque, forma y control (85 características), 3 degradaciones × 2 patrones de huecos, Viterbi (5, 40) | Actual |

Dominio corregido en v2: ruido suave, velocidades normalizadas por partido, sin posiciones extremas de equipo.

## Intentos y decisiones

Diferencias de spellF1 frente a `ref` contra etiquetas `raw`: huecos uniformes / huecos en muerto. Partidos que mejoran entre paréntesis.

| Intento | Medida | Decisión | Motivo |
|---|---|---|---|
| `all_feats` | +0,049 / +0,050 (7/7) | **Entra** | Forma es lo que más aporta |
| Huecos mixtos | `mixed_all` +0,048 / +0,076 (7/7) | **Entra** | Protege ante el patrón real desconocido |
| 3 degradaciones | `mixed_all3` +0,066 / +0,087 (7/7) | **Entra** | +0,011–0,018 sin componente nuevo; Metrica 11/12 |
| Detector de cambios | Sobre v3: −0,006 / +0,013; Metrica +0,007 | Fuera | Signo mixto, bajo el umbral, un modelo más |
| Etiquetas de control | −0,070 / −0,077; Metrica −0,06 | Fuera | Solo ganan contra `ctrl` |
| Huecos solo en muerto | `dead_all` −0,010 / +0,083 | Fuera | Falla si los huecos son uniformes |
| `es`, `two_heads`, `symmetric`, `weighted`, `seeds3`, `lightgbm`, `self_train` | Dentro de ±0,01 | Fuera | Sin aporte |
| MBR | −0,005 (1/7) | Fuera | Tras corregir el muestreo, tampoco aporta |
| Smart Tagging en decodificador | −0,001 a +0,005, simulado | Fuera | Ni con etiquetas perfectas |
| Grid de costes más amplio en v2 | 0,482 | Fuera | Sin ganancia |
| M4 (WASB, balón del vídeo) | `noisy20` +0,026 (7/7) | Fuera | No supera la puerta; solo 7 % de frames es juego sin balón |
| PathCRF | Sin ejecutar | Bloqueado | Entorno bloquea código externo; pide identidades y portero, que el reto no da |
| ELASTIC | Sin ejecutar | Sin sentido | Solo afina `ctrl`, rechazadas |
| Vídeo y audio | Sin ejecutar | Bloqueado | No hay `.mp4` en este PC |

## Hechos útiles

- Oráculos: equipo real 0,909; balón muerto real 0,516. El límite es decidir H o A.
- Posesiones por partido: IDSSE 437 de 7,2 s; `ctrl` 227 de 13,9 s; Metrica ~380 de 9 s; 5 min públicos 17 de 14,2 s.
- Huecos del reto: 150 tramos, mediana 10,7 s; 83 % en tramos que v2 predice como muertos (predicción, no etiqueta).
- Smart Tagging: los códigos vienen en pares espejo; `DEFENDING*` y `DEFENSIVE*` marcan al equipo sin balón.
- `sample_weight` de unos cambia los árboles de HGB; `train_v3.py` lo pasa para coincidir con lo medido.
- `train.py`: su early stopping no actúa (`n_iter_`=300). Sin efecto en v3, que usa `early_stopping=False`.
- Entorno: `.venv` Python 3.12 con `requirements-solution.txt`; `numba` y `lightgbm` solo para experimentos. RTX 4060, 16 hilos, 15,7 GB.

## Código

- Producción: `possession/train_v3.py`, `features_v3.py`, `pipeline.py` (`--version v3` por defecto).
- Banco: `possession/experiments/improve.py` (brazos), `decoder_exp.py` (MBR y Smart Tagging), `metrica_exp.py` (ensayo), `dose_exp.py` (puerta M4).
- Datos locales: `data/tactical/`, `data/external/{idsse,idsse_events,metrica}`. Metrica se descarga de `metrica-sports/sample-data`.

## Pendiente

1. Repositorio público y documento técnico.
2. Comprobar la regeneración en una máquina limpia.
3. Subir el CSV solo cuando se decida; la entrega oficial es el 17/11/2026.
