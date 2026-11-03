# Bitácora

Registro compacto de todos los cambios e intentos, para continuar sin releer el proyecto.
Actualizar al cerrar cada bloque. Detalle numérico: `data/experiments/improve/` y `data/experiments/improve_live/` (locales).

## Estado

- **Reto:** H/A/D por frame, Barça–Athletic, 199.319 frames. Ranking `spellF1`, desempate `macroF1`.
- **Mejor versión y entrega por defecto: v5** = HGB de v4 entrenado con `ctrl` (`ctrl_mixed_all3`) mezclado con el codificador de conjuntos (3 semillas), w 0,75, Viterbi (20, 20). `python possession/idsse.py` (también baja los eventos), `python possession/v5.py` y `python possession/pipeline.py`. Pesos del codificador en `possession/models/` (git), inferencia NumPy float64 de un hilo; sin torch. CSV `possession_v5.csv`, SHA-256 `866cb21a…a32dcc`, idéntico en dos ejecuciones (50 s); 182 posesiones de 18,1 s, 48,1 % muerto, 52/59 balones parados. HGB idéntico al medido. 5 min públicos (solo comprobación): spellF1 0,707 (v4 0,637), macroF1 0,910.
- **Versión anterior: v4** (`mixed_all3` con `LIVE_DEAD=1`, balón muerto móvil). `python possession/train_v4.py` y `python possession/pipeline.py --out <equipo>.csv`.
- **CSV:** `evaluation/results/possession_v4.csv`, SHA-256 `e6a371b2…cc538e`; 212 posesiones de 15,8 s, 47,4 % muerto, 52/59 balones parados con reanudación. **Respaldos:** `possession_v3.csv` (`e034f5d7…dd72ce`) y `possession_v2.csv`, intactos.
- **CV v3:** spellF1 0,556 (huecos uniformes) y 0,596 (en muerto), frente a 0,490 y 0,510 de `ref`.
- **Metrica v3:** 0,601 y 0,643, frente a 0,541 y 0,558.
- **Partido del reto:** 309 posesiones de 11,2 s; 49/59 balones parados con reanudación.
- **5 min públicos (solo comprobación):** v3 0,645 con 14 posesiones; v2 0,463 con 21; reales 17. Muestra mínima, no decide nada.
- **Reproducibilidad:** dos procesos separados dan el mismo modelo; el CSV se regenera idéntico en local. Falta máquina limpia.
- **Estimación corregida:** las cifras CV y Metrica de arriba usan balón muerto congelado, que no ocurre en el reto. Con balón muerto en movimiento, v3 da CV 0,478/0,556 y Metrica 0,514/0,596 (costes 5, 40).
- **Reproducibilidad v4:** predicciones idénticas al brazo medido (diferencia máxima 0,0) y CSV igual byte a byte en dos ejecuciones de `pipeline.py`.
- **Verdad de referencia:** solo `ctrl` (definición y 5 min públicos); `raw` ya no se mide. v4 usaba costes (40, 40), decididos contra `ctrl`.
- **v5 frente a v4, contra `ctrl`:** IDSSE 0,767/0,779 frente a 0,690/0,721 (7/7); Metrica 0,672/0,693 frente a 0,669/0,697 (empate). Frente a su HGB solo: +0,065/+0,044 IDSSE, +0,031/+0,033 Metrica.

## Reglas

- Sin etiquetas a mano en el partido. Los 5 min públicos solo comprueban y explican; nunca entrenan ni deciden.
- Nada del reto a la nube ni al repo. `data/` y `evaluation/results/` están ignorados.
- El CSV debe regenerarse idéntico (regla 3). Entrega en CPU. Sin fechas salvo que se pidan.
- Aceptar: +0,03 pareado, ≥5/7 partidos, sin empeorar en Metrica. Ruido por partido: ~0,04.
- Medir cambios pequeños solo contra `mixed_all3` y `ref`, sin repetir todo el banco.
- Medir solo contra `ctrl`. `raw` contradice CHALLENGE.md (exige control); las cifras `raw` de abajo son históricas.

## Versiones

| Versión | Contenido | Resultado |
|---|---|---|
| Baseline | Equipo más cercano + suavizado | Fragmenta posesiones |
| v1 | Viterbi H/A/D, hueco sin evidencia, muerto heurístico | Superada |
| v2 | HGB, 46 características, IDSSE degradado, Viterbi (10, 40) | CV 0,4805; respaldo |
| v3 | v2 + toque, forma y control (85 características), 3 degradaciones × 2 patrones de huecos, Viterbi (5, 40) | Respaldo |
| v4 | v3 entrenado con balón muerto móvil, Viterbi (40, 40) | Respaldo |
| **v5** | HGB v4 con `ctrl` + codificador de conjuntos (3 semillas, w 0,75), Viterbi (20, 20) | Actual |

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
| Coste H/A 14 / 20 (cobertura de definición, solo decodificador, v3 guardado) | Contra `raw` −0,014/−0,008 y −0,022/−0,020 (0–1/7); contra `ctrl` +0,042/+0,058 y +0,062/+0,074 (7/7); Metrica eventos hasta −0,005, control +0,023 a +0,037 | Sustituido | Resuelto en v4 con coste 40 contra `ctrl` (ver revisión abajo). Barrido en `data/experiments/hedge/` |
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

### Diagnósticos sobre v4

Fuera de pliegue, balón móvil, costes (40, 40); huecos uniformes / en muerto. Script `possession/experiments/oracle_exp.py`, salida `improve_live/oracle.json`.

| Oráculo | `ctrl` | `raw` |
|---|---|---|
| v4 | 0,680/0,716 | 0,495/0,534 |
| Tramos de v4, etiqueta real mayoritaria | 0,800/0,833 | 0,510/0,542 |
| Tramos reales, voto de v4 | 0,829/0,859 | 0,555/0,588 |
| Tramos reales, probabilidad media de v4 | 0,878/0,904 | 0,655/0,690 |
| Muerto real forzado | 0,741/0,752 | 0,543/0,574 |

Lectura contra `ctrl`: fronteras y etiquetas pesan parecido (+0,15 y +0,12). Agregar la probabilidad por tramo supera al voto (+0,05), lo que apoya el rescore de tramos; su techo sobre los cortes de v4 es +0,12. Muerto real: +0,06.

Curva de aprendizaje (`learning_exp.py`, `improve_live/learning/curve.json`): `mixed_all3` con 2, 4, 6 y 7 partidos IDSSE (3 subconjuntos), medido en Metrica, costes (40, 40). `ctrl`: 0,634/0,661 → 0,664/0,691 → 0,669/0,701 → 0,666/0,705; `raw`: 0,524/0,550 → 0,542/0,572 → 0,546/0,580 → 0,542/0,575. Satura desde 4–6 partidos; los 7 reproducen la medida anterior. Más partidos del mismo tipo baja de prioridad.

Rescore de tramos sin entrenar (`rescore_exp.py`, `improve_live/rescore.json`), solo `ctrl`: cortar con Viterbi más barato y dar a cada tramo la clase de mayor probabilidad media (o log-verosimilitud). IDSSE +0,003 (5/7) / −0,001 (0–1/7); Metrica +0,001 / 0,000. La validación anidada elige siempre coste 40: sobresegmentar no ayuda, y sobre los cortes de Viterbi el voto agregado coincide casi siempre con su etiqueta. **Fuera.** El margen del oráculo (+0,12 en etiquetas) no sale de reagregar las mismas probabilidades: hace falta evidencia nueva (contexto largo, HGB de dos etapas, o rescore entrenado con agregados por tramo).

HGB de dos etapas (`stack_exp.py`, `improve_live/stack/`), solo `ctrl`: etapa 1 `mixed_all3` anidada (42 ajustes con 5 partidos), etapa 2 HGB sobre probabilidades en ventanas ±1/5/15/60 s, laterales 5/15 s y agregados del tramo, entrenada con `ctrl` y aumentada con H/A intercambiados. IDSSE −0,004 (2/7) / −0,013 (1/7); Metrica −0,021 / −0,041; costes elegidos (20, 40) y (20, 20). **Fuera.** Mismo patrón que reentrenar con `ctrl`: aprende la aproximación `ctrl` de IDSSE y no transfiere a Metrica. Además la etapa 1 interna (5 partidos) es más débil que la de test (6), lo que desajusta las probabilidades.

Lote de geometría H−A (`geom_exp.py`, `improve_live/geom.json`, brazo `ctrl_geom3`, entrenado con `ctrl`): mínimo suave de distancias (1 y 3 m), llegada proyectada a 0,5 y 1 s, presión, Voronoi (todo el campo, tercios, junto al balón), rejilla 6×4 de ocupación H−A; 44 columnas. Frente a `ctrl_mixed_all3`: IDSSE −0,001/−0,005 (3/7), Metrica +0,004/+0,005. **Fuera**: el HGB ya saca esa señal de lo que tiene.

Codificador de conjuntos (`encoder_data.py`, `encoder_train.py` en `C:\scripts\hackathon_barsa\venv-torch`, `encoder_score.py`; salida `improve_live/encoder.json`). DeepSets por equipo (media, máximo, atención por cercanía al balón) + TCN dilatada ±50 s a 5 pasos/s, 451 k parámetros, **entrenado con `ctrl`**, ambas vistas H/A, 3000 iteraciones, ~2 min por pliegue en la RTX 4060. Mezcla con v4 por un peso, peso y costes anidados. Solo `ctrl`:

| Referencia | Brazo | IDSSE | Metrica |
|---|---|---|---|
| v4 actual (0,690/0,721; Metrica 0,669/0,697) | Codificador solo | +0,072/+0,053 (7/7) | +0,007/−0,007 |
| | Mezcla (w 0,75; costes 28/20) | +0,081/+0,067 (7/7) | +0,014/+0,012 |
| v4 con `ctrl` (0,702/0,735; Metrica 0,641/0,660) | Codificador solo | +0,059/+0,039 (7/7) | +0,036/+0,030 |
| | Mezcla | +0,066/+0,053 (7/7) | +0,036/+0,035 |

**Pasa la puerta** frente a las dos referencias. Semillas 2028 y 2029: mezcla frente a `ctrl_mixed_all3` +0,058/+0,055 y +0,057/+0,046 (7/7), Metrica +0,034/+0,032 y +0,025/+0,029. Promedio de 3 semillas: +0,065/+0,044 (7/7), Metrica +0,031/+0,033. Constantes de producción (`encoder_score.py pick`, media de 7 partidos × 2 huecos): w 0,75, (20, 20), 0,781; cima plana (±0,001). Inferencia NumPy frente a torch: diferencia máxima 1,1e-6, argmax 100 %. Etiquetas `ctrl` movidas a `possession/control.py` (reproducen las cacheadas); `improve.py` las importa.

Fallo del codificador en el reto (`error_review.py`, solo agregados): tras el pitido del descanso el vídeo sigue ~80 s con 0–2 jugadores; un equipo sin jugadores dejaba el máximo en el relleno −1e4 (la condición usaba el recuento con mínimo 1), y GroupNorm sobre la mitad entera aplanaba **toda la primera parte** (confianza 0,366 frente a 0,955 en trozos de 512). Corregido en `encoder.py` y `encoder_train.py`: equipo vacío → 0. Ningún paso de entrenamiento ni test tenía un equipo vacío, así que pesos y cifras medidas no cambian. Tras corregir: confianza en el reto 0,944 (IDSSE 0,94), frames con max p < 0,6 del 51,7 % al 3,4 %. CSV anterior `fa9d73de…` sustituido.

Errores que quedan en la ventana pública (entender, no ajustar): 10/17 posesiones bien; fallan posesiones H muy cortas (0,8, 1,3 y 3,5 s) dentro de tramos A, que v5 trata como toques, y las A vecinas quedan fusionadas. v5 da 3,2–3,3 posesiones por minuto vivo frente a 4,23 de la ventana; en IDSSE da 0,89–0,90 veces el ritmo de `ctrl`. Es decir, la verdad del reto conserva posesiones más cortas que nuestra aproximación `ctrl`.

Posesiones cortas (`short_review.py`, solo agregados). Duraciones: ventana pública 12 % < 2 s y 24 % de 2–4 s (n 17); IDSSE `raw` 30 % y 23 %; `ctrl` 6 % y 14 %. Recall de v5 en IDSSE contra `ctrl` (IoU ≥ 0,5): < 2 s 21 %, 2–4 s 52 %, 4–8 s 68–71 %, 8–16 s 83–85 %, > 16 s 91–92 %. En las cortas perdidas, p(equipo real) ≈ 0,45: el modelo duda, no es solo el coste. En la ventana, las dos < 2 s tienen p(real) 0,07: el modelo ve al otro equipo, coherente con `ctrl`, que borra toques de < 1,5 s. Coste 20: una posesión de 1 s necesita p ≈ 0,83 de media. spellF1 pesa igual cada posesión y la pérdida por frame pesa por duración: las cortas pesan poco en entrenamiento. Literatura: PathCRF (arXiv 2602.12080, mismos datos IDSSE, 5 partidos) usa set attention + temporal + CRF con transiciones dinámicas: F1 65,3 sin CRF, 73,1 con Viterbi, 75,7 con CRF aprendido; precisión de equipo por frame 92,4 %. Siguientes candidatos: pérdida ponderada por posesión y transiciones aprendidas.

Variantes del codificador (semilla 2027, `encoder_train.py <root> 2027 <variante>`, `variant_score.py`, `improve_live/variants.json`), frente al codificador base de la misma semilla, ambos mezclados con `ctrl_mixed_all3`, peso/costes/ganancia anidados, solo `ctrl`:

| Variante | IDSSE | Metrica | Recall < 2 s / 2–4 s |
|---|---|---|---|
| base | 0,768/0,788 | 0,676/0,695 | 19 %/47 % · 21 %/53 % |
| `w1` (peso por posesión, α 1) | −0,003 (3/7) / −0,007 (3/7) | +0,008/+0,004 | 18 %/45 % · 19 %/50 % |
| `w05` (α 0,5) | −0,003 (4/7) / −0,006 (2/7) | +0,011/+0,008 | 16 %/48 % · 19 %/51 % |
| `trans` (logit de cambio en el coste) | +0,001 (2/7) / −0,007 (1/7) | −0,008/+0,004 | 20 %/50 % · 20 %/50 % |

**Las tres fuera**: dentro de ±0,01 y el recall de las cortas no se mueve. Con pérdida ponderada la validación sube los costes a 40 y compensa. Lectura: las posesiones cortas no se pierden por cómo se pesa la pérdida ni por el decodificador, sino porque el tracking no da evidencia suficiente para ellas (p ≈ 0,45).

Recall de v5 según balón visible en la posesión (`short_review.py ball`, IDSSE, contra `ctrl`), huecos uniformes / en muerto, balón < 20 % → > 80 % de la posesión: < 2 s 20 % → 21 % / 11 % → 27 %; 2–4 s 42 % → 57 % / 38 % → 50 %; 4–8 s 59 % → 75 % / 52 % → 72 %; 8–16 s 78 % → 85 % / 69 % → 86 %; > 16 s 91–96 % sin diferencia. p(equipo real) en < 2 s con balón: 0,53–0,57. **El balón explica +10–20 puntos en 2–16 s, pero las < 2 s se pierden aun viéndolo**: su límite no es el balón. Hipótesis sin comprobar: parte de las `ctrl` < 2 s son artefactos de alinear eventos con frames y no se pueden aprender.

Datos: SkillCorner Open Data (20 partidos, MIT, 3,6 GB) clonado en `data/external/skillcorner`. Sin formularios: solo fuentes de descarga directa.

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
- Reto: tras el pitido del descanso el vídeo sigue ~80 s con 0–2 jugadores; hasta 24 pistas locales a la vez (p99 16). Todo modelo debe tolerar equipos vacíos y más de 20 pistas.
- Datos: solo fuentes de descarga directa, sin formularios ni peticiones (decisión del usuario). PFF queda fuera. SkillCorner (MIT) descargado y sin usar: la curva satura.
- Entrenar y medir contra `ctrl`. Entrenar con `ctrl` falló en el HGB de dos etapas (Metrica −0,02/−0,04), pero funciona en el codificador: el fallo era del modelo, no de las etiquetas.
- `BallPossession` de DFL: sin definición pública localizada (README de IDSSE no la da; el artículo de Bassek et al. no se pudo leer).

### Literatura (revisada 2026-11-02)

- **A favor del codificador:** PathCRF (Kim et al., arXiv 2602.12080, código `hyunsungkim-ds/pathcrf`) usa los mismos 7 partidos IDSSE (5 de entrenamiento): set attention por equipo + módulo temporal + CRF. Ablaciones: F1 65,3 sin CRF, 73,1 con Viterbi fijo, 75,7 con CRF de transiciones dinámicas; separar interacciones dentro/entre equipos 68,2 → 75,7; 25 fps 77,2 frente a 75,7 a 5 fps. Equipo por frame 92,4 %. Usa tracking limpio con balón. Borghesi, Costa, Morra y Lamberti (ESWA 2023): TCN + capas invariantes a permutación, posesión 86,2 % (+7,3 sobre reglas), vivo/muerto 89,2 %. Kim et al. (KDD 2023, `pientist/ballradar`) y TranSPORTmer (ACCV 2024): Set Transformers para poseedor y estado de juego.
- **Matiza:** Grinsztajn et al. (NeurIPS 2022): los árboles siguen ganando en tabular medio; coherente con que la mezcla HGB + red sea lo mejor. Ninguna fuente evalúa tracking automático degradado ni desglosa posesiones cortas; con 40 % de balón ausente no hay precedente. Lo que da a PathCRF su ganancia (transiciones aprendidas) aquí no aportó.
- **Decodificación:** la segmentación temporal trata la sobre- y subsegmentación con priors de duración (Constraint-Aware Decoding, arXiv 2605.10149; Lea et al., arXiv 1602.02995).

### Investigación de posesiones cortas (2026-11-02, `reports/Posesiones cortas en tracking.md`)

- **Diagnóstico:** las cortas se pierden por falta de evidencia de equipo (p ≈ 0,45), no por pérdida ni decodificador; ningún coste o prior de duración las recupera sin inventar otras. Casi todo lo publicado en segmentación combate el exceso de segmentos, no la falta de cortos.
- **DFL `BallPossession`:** sin algoritmo ni duración mínima públicos; Bassek et al. 2025 solo dice "definido por el jugador que controla el balón" (TRACAB Gen5, operadores en directo, catálogo DFL v5.1 no público). En `raw` el 30 % dura < 2 s: no hay mínimo efectivo. databallpy discrepa del campo en ~10 % de frames vivos de un partido Sportec.
- **Definiciones:** Wyscout, Opta y FIFA EFI ("In Contest"): un toque sin control no cambia posesión. Link y Hoernig 2017: toque por aceleración ≥ 4 m/s², kappa de control 0,67 (humanos 0,72). Vidal-Codina 2022: zona 0,5–1 m, ganancia solo con cambio de dirección o velocidad del balón, duelo con 2+ rivales a 1 m. Yamamoto 2023 (J1, 95 partidos): posesión individual media 2,49 s; 11,3 % de las de equipo < 1 s.
- **Ruido de reloj:** Biermann 2023: error mediano 0,52 s de los pases del proveedor; con desfase constante (como el nuestro) solo el 49,9 % queda a < 0,5 s; con sincronización por trayectoria 70–81 %. Vidal-Codina: eventos Sportec hasta > 10 s desplazados. ELASTIC 2025: 88,4 % al frame exacto, 95 % a < 1 s. Acuerdo humano en bordes de posesión: IoU 0,72–0,78.
- **Balón inferido:** Ball Radar (KDD 2023) sin balón: 3,66 m, equipo 85,9 %, jugador 64,7 %; con 20 % de balón visible: 1,31 m, jugador 93,1 %. En IDSSE, balón regresado + reglas da F1 48,4 frente a 75,7 de PathCRF. TranSPORTmer (283 partidos LaLiga): 2,57–2,71 m. Peral 2025 necesita vídeo.
- **Cambio de proveedor:** PathCRF Sportec→K League F1 49,2 (73,9 en el mismo proveedor); pierde más recall que precisión, como nosotros. Crang et al. 2025: tracking de retransmisión, RMSE de posición 1,7–16,4 m según proveedor.
- **Auditoría de `ctrl` (`label_audit.py`, tracking limpio, umbrales fijados antes de mirar: toque = Δv del balón ≥ 2 m/s en ±3 frames con el más cercano a ≤ 1,5 m; duelo = ambos equipos a ≤ 1 m).** Toque al inicio (±0,5 s) propio / rival: < 2 s 87 %/29 %, 2–4 s 89 %/18 %, 4–8 s 87 %/21 %, > 16 s 90 %/17 %: las cortas empiezan con un toque propio igual que las largas. Sin toque propio: 8 % de las < 2 s, 2 % de las 2–4 s. Cercanía media del equipo: 0,46 (< 2 s), 0,56, 0,64, 0,70, 0,81 (> 16 s): en las cortas el rival está igual de cerca, y coincide con p ≈ 0,45 del modelo. Tipos < 2 s: control 29 %, duelo 26 %, toque con el rival más cerca 36 %, sin toque 8 %. Reloj de eventos ±0,5 / ±1 s: cambian 0,13–0,17 % / 0,39–0,41 % de frames y sobreviven 95–97 % / 92 % de las cortas. **Conclusión: las cortas de `ctrl` son posesiones reales y robustas al reloj; lo que las distingue es quién toca el balón, no quién está más cerca.**
- **Código reutilizable:** PathCRF (`hyunsungkim-ds/pathcrf`, MPL-2.0 según GitHub; trae eventos DFL sincronizados), databallpy, ELASTIC (licencia no declarada), Ball Radar (sin licencia: no reutilizable).

## Código

- Producción: `possession/train_v4.py` (usa `train_v3.py` con `moving=True`), `features_v3.py`, `pipeline.py` (`--version v5` por defecto). v5: `v5.py` (constantes y HGB con `labels="ctrl"`), `encoder.py`, `control.py`, `models/*.npz`.
- Codificador (entorno aparte `C:\scripts\hackathon_barsa\venv-torch`, torch 2.11 cu128): `experiments/encoder_data.py` → `encoder_train.py <improve_live> <seed> [w05|w1|trans]` → `encoder_export.py` → `encoder_score.py` (`pick` da las constantes de v5) y `variant_score.py`.
- Diagnósticos v4/v5: `oracle_exp.py`, `learning_exp.py`, `rescore_exp.py`, `stack_exp.py`, `geom_exp.py`, `error_review.py` (ventana pública, ritmo, confianza), `short_review.py` (duraciones y recall por duración).
- Banco: `possession/experiments/improve.py` (brazos), `decoder_exp.py` (MBR y Smart Tagging), `metrica_exp.py` (ensayo), `dose_exp.py` (puerta M4).
- Datos locales: `data/tactical/`, `data/external/{idsse,idsse_events,metrica}`. Metrica se descarga de `metrica-sports/sample-data`.

## Pendiente

Orden razonado y fuentes en `reports/Posesión en tracking fuentes alternativas.md`. Los rangos de ganancia son estimaciones sin medir.

1. Sin decisiones abiertas sobre datos: sin formularios (PFF fuera) y sin consulta a organizadores (RULES.md y CHALLENGE.md lo resuelven).
2. Diagnósticos cerrados (oráculo y curva). La dosis de balón y reanudación se salta: no hay vídeo y el muerto real solo da +0,06.
3. Mejoras opcionales cerradas: rescore, dos etapas y lote H−A fuera; codificador dentro. v5 elegida sobre la mezcla con el HGB `raw` (empate dentro del ruido; coherencia con `ctrl`). Reanudaciones sin balón y SkillCorner bajan (muerto real +0,06; curva saturada). Pérdida ponderada y transiciones aprendidas fuera.
3b. Hecho: recall de posesiones cortas según el balón visible (abajo). El balón ayuda poco; las < 2 s se pierden incluso con balón.
3c. Investigación de posesiones cortas: `reports/Posesiones cortas en tracking.md` (notas en `research_notes/Posesiones cortas en tracking/`).
3d. Auditoría de etiquetas hecha (`label_audit.py`, 3 s, sin modelo; abajo): las `ctrl` cortas son reales, no artefactos. Siguiente: evidencia de toque (quién golpea el balón) en el modelo.
4. Repositorio público y documento técnico.
5. Comprobar la regeneración en una máquina limpia.
6. Subir el CSV solo cuando se decida; la entrega oficial es el 17/11/2026.
