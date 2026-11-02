# Bitácora

Registro compacto de todos los cambios e intentos, para continuar sin releer el proyecto.
Actualizar al cerrar cada bloque. Detalle numérico: `data/experiments/improve/` (local) y `docs/plan/continuar_m4_gpu.md`.

## Estado

- **Reto:** H/A/D por frame, Barça–Athletic, 199.319 frames. Ranking `spellF1`, desempate `macroF1`.
- **Mejor versión: v4** (`mixed_all3` con `LIVE_DEAD=1`, balón muerto móvil). `python possession/train_v4.py` y `python possession/pipeline.py --out <equipo>.csv`.
- **CSV:** `evaluation/results/possession_v4.csv`, SHA-256 `e6a371b2…cc538e`; 212 posesiones de 15,8 s, 47,4 % muerto, 52/59 balones parados con reanudación. **Respaldos:** `possession_v3.csv` (`e034f5d7…dd72ce`) y `possession_v2.csv`, intactos.
- **CV v3:** spellF1 0,556 (huecos uniformes) y 0,596 (en muerto), frente a 0,490 y 0,510 de `ref`.
- **Metrica v3:** 0,601 y 0,643, frente a 0,541 y 0,558.
- **Partido del reto:** 309 posesiones de 11,2 s; 49/59 balones parados con reanudación.
- **5 min públicos (solo comprobación):** v3 0,645 con 14 posesiones; v2 0,463 con 21; reales 17. Muestra mínima, no decide nada.
- **Reproducibilidad:** dos procesos separados dan el mismo modelo; el CSV se regenera idéntico en local. Falta máquina limpia.
- **Estimación corregida:** las cifras CV y Metrica de arriba usan balón muerto congelado, que no ocurre en el reto. Con balón muerto en movimiento, v3 da CV 0,478/0,556 y Metrica 0,514/0,596 (costes 5, 40).
- **Reproducibilidad v4:** predicciones idénticas al brazo medido (diferencia máxima 0,0) y CSV igual byte a byte en dos ejecuciones de `pipeline.py`.
- **Verdad de referencia:** `ctrl` pasa a ser la principal (definición y 5 min públicos); `raw` queda como contraste. v4 usa costes (40, 40), decididos contra `ctrl`.

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
| v3 | v2 + toque, forma y control (85 características), 3 degradaciones × 2 patrones de huecos, Viterbi (5, 40) | Respaldo |
| **v4** | v3 entrenado con balón muerto móvil, Viterbi (40, 40) | Actual |

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
| Coste H/A 14 / 20 (cobertura de definición, solo decodificador, v3 guardado) | Contra `raw` −0,014/−0,008 y −0,022/−0,020 (0–1/7); contra `ctrl` +0,042/+0,058 y +0,062/+0,074 (7/7); Metrica eventos hasta −0,005, control +0,023 a +0,037 | **Pendiente del usuario** | No pasa la puerta contra `raw`; declarar π y coste antes de mirar. Barrido en `data/experiments/hedge/`; falta repetir en producción |
| Saerens EM y prior H/A, incluso real | ≤ +0,004 | Fuera | Sin ganancia |
| Fusión por duración | < 3 s pierde 0,04 en Metrica | Fuera | Dominada por subir el coste |
| Mezcla `raw`+`ctrl` | Misma frontera (una degradación `ctrl`) | Fuera | Sin cobertura extra |

### Revisión con ground truth y balón muerto en movimiento

Diagnóstico: `possession/experiments/gt_review.py`; solo agregados y sin elegir por la puntuación de los 5 min. Rejilla: `possession/experiments/live_review.py`, resultado en `data/experiments/improve_live/review.json`. Test con balón móvil, huecos uniformes / en muerto.

| Intento | Medida | Decisión | Motivo |
|---|---|---|---|
| Reentrenar v3 con balón muerto móvil (`moving_dead_ball`) | Frente a v3, costes (5, 40): `raw` +0,065/+0,030 (7/7), `ctrl` +0,078/+0,041 (7/7); Metrica `raw` +0,063/+0,028, `ctrl` +0,077/+0,029 (6/6). Recall de muerto 0,80/0,90 → 0,96 | **Entra** como v4 | Corrige un supuesto falso; gana en todos los partidos, aunque el patrón en muerto queda en el límite de la puerta |
| Etiquetas de control reentrenadas (`ctrl_mixed_all3`) | Contra `ctrl`, cada uno en sus mejores costes: IDSSE +0,026/+0,024; Metrica −0,027/−0,035. Contra `raw` −0,09/−0,11 | Fuera | Subir el coste de `mixed_all3` da lo mismo sin perder en Metrica |
| Detector de cambios contra `ctrl` | +0,007/+0,006, balón congelado | Fuera | Bajo la puerta también con la nueva referencia |
| Otros descartados contra `ctrl` | Sin aporte, balón congelado | Fuera | No cambia su decisión |
| Coste H/A sobre el modelo reentrenado | Sin π: tabla abajo. Con π=0,7 para `ctrl`: 0,594 (5), 0,625 (14), 0,632 (20), 0,638 (28), 0,643 (40); Metrica 0,626, 0,644, 0,647, 0,646, 0,648 | **Entra coste 40** (corrige 28) | `raw` contradice la definición: se decide contra `ctrl`. Costes altos: 56 da IDSSE 0,691/0,721 y Metrica 0,669/0,697; 80 da 0,691/0,719 y 0,655/0,680. 40 y 56 empatan; 40 cede menos contra `raw` y es mejor en Metrica en muerto |

| Coste H/A | IDSSE `raw` | IDSSE `ctrl` | Metrica `raw` | Metrica `ctrl` |
|---|---|---|---|---|
| 5 | 0,544/0,586 | 0,594/0,618 | 0,577/0,624 | 0,621/0,652 |
| 14 | 0,534/0,579 | 0,640/0,669 | 0,583/0,623 | 0,644/0,678 |
| 20 | 0,526/0,568 | 0,657/0,681 | 0,576/0,615 | 0,655/0,682 |
| 28 | 0,513/0,557 | 0,671/0,694 | 0,561/0,603 | 0,658/0,688 |
| 40 | 0,495/0,534 | 0,680/0,716 | 0,542/0,575 | 0,666/0,705 |

Cautela: entrenamiento y test comparten simulador, y el simulador sigue más tranquilo que el reto (velocidad en muerto: mediana 1,45 frente a 2,7 m/s; p90 3,85 frente a 9,5). No se afina más para no ajustarlo a la ventana pública. `metrica_exp.py run ctrl_mixed_all3` con balón congelado ya no hace falta: la medida con balón móvil lo sustituye.

## Hechos útiles

- Definición (CHALLENGE.md): exige ganancia **controlada**; balón disputado o suelto sigue con el último equipo; toque, desvío y duelo aéreo sin control no cambian posesión. Apunta a `ctrl`. La verdad sale mecánicamente del dueño del tracking corregido, sin suavizar; `ctrl` es solo nuestra aproximación.
- 5 min públicos: permitidos para entender la etiqueta; prohibido elegir versiones o constantes por su puntuación.
- 5 min públicos frente a ventanas IDSSE: duración media 14,19 s (`raw` p95, `ctrl` p58), posesiones por minuto vivo 4,23 (`raw` p5, `ctrl` p42), cambios directos por minuto 2,98 (`raw` p5, `ctrl` p75). Se parecen a `ctrl`.
- Balón muerto: congelado en IDSSE y Metrica (100 % de frames), móvil en el reto (mediana 2,7 m/s, p90 9,5). `ball_speed` y `ball_moved_1s` aprendían una señal falsa; recall público 0,68 frente a 0,97 en IDSSE.
- Balón ausente en el reto: la ventana pública es el mejor bloque de 20 (falta 11,8 % frente a 40,7 % del partido; bloques 0,12–0,80), así que 0,645 no representa el partido. P(falta|muerto) 0,336, P(falta|vivo) 0,065: apoya `mixed`.
- Fronteras H↔A: error ≈1,0 s, parecido a IDSSE (0,72–0,84). El cambio de etiqueta coincide con el cambio de equipo más cercano (mediana +0,00 s).
- Oráculos: equipo real 0,909; balón muerto real 0,516. El límite es decidir H o A.
- Posesiones por partido: IDSSE 437 de 7,2 s; `ctrl` 227 de 13,9 s; Metrica ~380 de 9 s; 5 min públicos 17 de 14,2 s.
- Huecos del reto: 150 tramos, mediana 10,7 s; 83 % en tramos que v2 predice como muertos (predicción, no etiqueta).
- Smart Tagging: los códigos vienen en pares espejo; `DEFENDING*` y `DEFENSIVE*` marcan al equipo sin balón.
- `sample_weight` de unos cambia los árboles de HGB; `train_v3.py` lo pasa para coincidir con lo medido.
- `train.py`: su early stopping no actúa (`n_iter_`=300). Sin efecto en v3, que usa `early_stopping=False`.
- Entorno: `.venv` Python 3.12 con `requirements-solution.txt`; `numba` y `lightgbm` solo para experimentos. RTX 4060, 16 hilos, 15,7 GB.

## Código

- Producción: `possession/train_v4.py` (usa `train_v3.py` con `moving=True`), `features_v3.py`, `pipeline.py` (`--version v4` por defecto).
- Banco: `possession/experiments/improve.py` (brazos), `decoder_exp.py` (MBR y Smart Tagging), `metrica_exp.py` (ensayo), `dose_exp.py` (puerta M4).
- Datos locales: `data/tactical/`, `data/external/{idsse,idsse_events,metrica}`. Metrica se descarga de `metrica-sports/sample-data`.

## Pendiente

Orden razonado y fuentes en `reports/Posesión en tracking fuentes alternativas.md`. Los rangos de ganancia son estimaciones sin medir.

1. Decisiones del usuario: formulario PFF y consulta a organizadores (Metrica, PFF, definición).
2. Diagnósticos: oráculo de tramos y etiquetas; curva de aprendizaje; dosis v3 con balón de 3 y 5 m y con equipo de reanudación.
3. Según diagnósticos: HGB de dos etapas, rescore de tramos, reanudaciones sin balón, lote de features H−A, SkillCorner (20 partidos, MIT), codificador de conjuntos.
4. Repositorio público y documento técnico.
5. Comprobar la regeneración en una máquina limpia.
6. Subir el CSV solo cuando se decida; la entrega oficial es el 17/11/2026.
