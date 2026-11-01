# Bitácora de avance

Registro de lo hecho en la Fase 1 de More than a Hack 2027: predecir H/A/D por frame en Barça–Athletic.
Incluye lo que se integró, lo que se descartó y por qué. Las etapas van en orden; se añade una nueva al
final de cada bloque de trabajo y no se reescriben las anteriores.

**Estado actual:** la entrega de respaldo sigue siendo v2 (`evaluation/results/possession_v2.csv`). La
mejor variante medida es `mixed_all`, que aún no está integrada en `possession/train.py` ni en
`possession/pipeline.py`.

**Cómo leer las cifras**

- `spellF1` es la métrica de ranking; `macroF1` desempata. Salvo que se indique, las cifras son de
  validación dejando un partido fuera sobre IDSSE/DFL degradado, no notas del reto.
- Una diferencia inferior a unos 0,04 en un partido completo es ruido (`EVALUATION.md`).
- **Criterio de aceptación:** +0,03 de spellF1 pareado, mejora en al menos 5 de 7 partidos y no empeorar
  en el ensayo con Metrica.

## Resumen de decisiones

| Etapa | Cambio | Decisión |
|---|---|---|
| 1 | Viterbi H/A/D, balón ausente sin evidencia, balón muerto heurístico (v1) | Integrado |
| 2 | Validación en IDSSE degradado (M5) | Integrado |
| 3 | Modelo aprendido con 46 características y costes por validación (v2) | Integrado, entrega de respaldo |
| 3 | Ruido suave, velocidades normalizadas, sin posiciones extremas por equipo | Integrado |
| 3 | Señales de toque | Descartado por separado: dentro del ruido |
| 3 | Detector de cambios | No integrado: sin superar el ruido |
| 3 | Entrenar con 3 degradaciones por partido | Prometedor, sin integrar ni combinar con `mixed_all` |
| 4 | Recuperar el balón del vídeo con WASB (M4) | Descartado: no supera la puerta |
| 5 | Etiquetas de control | Descartado |
| 5 | Forma de equipo, toque y control juntos (`all_feats`) | Aceptado, sin integrar |
| 5 | Entrenar con ambos patrones de huecos (`mixed_all`) | Elegido, sin integrar |
| 5 | Huecos solo en balón muerto (`dead_all`) | Descartado: depende de un patrón sin confirmar |
| 5 | Early stopping, dos modelos, simetría, pesos, semillas, LightGBM, autoentrenamiento | Descartados: sin aporte |
| 5 | MBR y prior de Smart Tagging | Descartados: sin aporte |
| 5 | PathCRF, ELASTIC, vídeo y audio | Sin probar |

## Etapa 0: preparación

**Hecho**

- Cargador del tracking con caché y características por frame sin identidades de jugador.
- Lectura de las normas: nada etiquetado a mano en el partido, 5 minutos públicos solo para comprobar y
  entender, datos de otros partidos permitidos, ningún dato del reto en la nube ni publicado.

## Etapa 1: v1 sin entrenamiento

**Integrado** en `possession/pipeline.py` y `possession/decode.py`:

- **M1:** Viterbi sobre H/A/D con coste por cambio de estado, en lugar de voto por mayoría.
- **M2:** un frame sin balón no aporta evidencia, así que la posesión continúa a través del hueco.
- **M3:** balón muerto por ritmo colectivo, último balón visto fuera y tramo previo al saque de la
  segunda parte.

**Motivo:** el baseline pierde por fragmentar posesiones, no por fallar frames.

## Etapa 2: validación externa

**Integrado** en `possession/idsse.py` y `possession/train.py`:

- **M5:** 7 partidos abiertos de IDSSE/DFL, degradados para parecerse al tracking del reto, con
  validación dejando un partido fuera y la métrica oficial.

**Relegado:** SkillCorner y las muestras de Metrica quedaron como referencia. Metrica se recuperó en la
etapa 5 para el ensayo.

## Etapa 3: v2 con modelo aprendido

**Integrado**

- HistGradientBoosting con 46 características y Viterbi con costes elegidos por validación
  (switch=10, dead=40). Resultado: spellF1 0,4805 y macroF1 0,8633.
- Tres correcciones del desplazamiento de dominio, detectadas sin etiquetas: ruido suave en el tiempo,
  velocidades normalizadas por partido y retirada de las posiciones extremas de cada equipo. En el
  partido del reto, v2 pasó de 388 posesiones de 6,7 s a 317 de 11,1 s.

**Medido y no integrado**

| Variante | spellF1 | macroF1 | Por qué |
|---|---|---|---|
| Señales de toque | 0,488 | 0,869 | Mejora dentro del ruido |
| Toque + detector de cambios | 0,496 | 0,871 | 5 partidos suben y 2 bajan |
| Ampliar el grid de costes | 0,482 | | Sin ganancia con `(switch=10, dead=80)` |
| 3 degradaciones, sin detector | 0,506 | 0,874 | Aportan más que el detector; sin integrar |
| 3 degradaciones + toque + detector | 0,514 | 0,876 | Sube en 7 de 7 frente a 1 degradación; sin integrar |

El 0,514 es algo optimista: sus costes se eligieron con los mismos 7 partidos. Estas dos filas se midieron
con `change_exp.py` y no tienen ensayo con Metrica.

**Oráculos, que orientan pero no son alcanzables**

| Oráculo | spellF1 | Lectura |
|---|---|---|
| Balón muerto real | 0,516 | El balón muerto no es el límite |
| Equipo real | 0,909 | El límite es decidir H o A |
| Sin ninguna degradación | 0,614 | Cota holgada: también recupera jugadores |

## Etapa 4: puerta de M4 con la GPU local

**Preparado**

- Entorno `.venv` con Python 3.12 y las versiones fijadas; datos IDSSE y eventos descargados.
- Copiados a mano solo los ficheros ligeros del reto a `data/tactical/`; no se repitió la descarga de 13 GB.
- CSV de v2 generado y validado: 317 posesiones de 11,1 s de media. Se regenera idéntico desde el modelo
  guardado; falta comprobarlo desde `train.py`.
- Consultas a la organización retiradas: las normas ya cubren los 5 minutos y los datos externos,
  `CHALLENGE.md` resuelve la definición de posesión y la entrega seguirá en CPU.

**Descartado: M4 (WASB).** La puerta exigía +0,03 en `noisy20` y mejora en al menos 5 de 7 partidos.
Con los huecos reales del reto se quedó en +0,026 (7/7).

| Brazo | Balón ausente | spellF1 | Frente a `ref41` |
|---|---|---|---|
| `ref41` | 40 % | 0,488 | |
| `hard41` (huecos en los cambios) | 40 % | 0,467 | −0,021 |
| `ideal20` | 19 % | 0,527 | +0,040 |
| `noisy20` | 19 % | 0,514 | +0,026 |
| `ball_all` | 0 % | 0,568 | +0,080 |
| `dead41` (huecos en balón muerto) | 41 % | 0,539 | +0,051 |
| `dead_m4` | 36 % | 0,564 | +0,076 |

El 83 % de los frames sin balón del reto cae en tramos que v2 predice como muertos. Con ese patrón, la
ganancia comparable de M4 es de +0,025 aun con recuperación perfecta. Es una predicción, no una etiqueta.
`dead41` no es ganancia alcanzable: mide un problema más fácil.

**Corrección de rumbo:** el early stopping de `train.py` no actuaba (`n_iter_` llegaba a 300), y en
`degrade()` los huecos caen al azar. Ambos puntos pasaron a la etapa 5.

## Etapa 5: investigación y prueba de todas las mejoras

**Investigación:** `reports/Posesión en tracking cinco mejoras.md`, con notas en `research_notes/`.

**Banco:** `possession/experiments/improve.py`, `decoder_exp.py`, `metrica_exp.py` y `tags_check.py`.
Cada variante se puntuó en los mismos frames, cruzando dos verdades (`raw`, `BallPossession` tal cual, y
`ctrl`, solo tramos con `Play` o `ShotAtGoal` del equipo a ±1,5 s) con dos patrones de huecos (`uniform` y
`dead`). Los costes de cada partido se eligieron con los otros seis.

### Resultados en IDSSE

| Variante | Qué cambia | Uniform/raw | Uniform/ctrl | Dead/raw | Dead/ctrl |
|---|---|---|---|---|---|
| `ref` | Referencia | 0,490 | 0,634 | 0,510 | 0,639 |
| `es` | Iteraciones con partido retenido | 0,483 | 0,650 | 0,504 | 0,649 |
| `two_heads` | Modelos separados de muerto y equipo | 0,490 | 0,648 | 0,502 | 0,653 |
| `symmetric` | Promedio con equipos intercambiados | 0,490 | 0,639 | 0,512 | 0,644 |
| `weighted` | Más peso cerca de los cambios | 0,490 | 0,639 | 0,514 | 0,643 |
| `seeds3` | 3 semillas | 0,491 | 0,640 | 0,509 | 0,640 |
| `lightgbm` | Otra librería | 0,484 | 0,641 | 0,514 | 0,652 |
| `self_train` | Autoentrenamiento | 0,489 | 0,610 | 0,504 | 0,634 |
| `touch` | Señales de toque | 0,500 | 0,654 | 0,527 | 0,654 |
| `control` | Retención y cadenas de pases | 0,494 | 0,646 | 0,521 | 0,649 |
| `shape` | Forma de equipo | 0,523 | 0,680 | 0,537 | 0,664 |
| `change` | Toque + detector de cambios | 0,516 | 0,652 | 0,541 | 0,666 |
| `all_feats` | Toque, forma y control | 0,539 | 0,680 | 0,559 | 0,674 |
| `dead_gaps` | Entrenar con huecos en muerto | 0,425 | 0,593 | 0,534 | 0,690 |
| `dead_all` | `all_feats` + huecos en muerto | 0,480 | 0,633 | 0,592 | 0,701 |
| **`mixed_all`** | `all_feats` + ambos patrones | **0,538** | **0,676** | **0,585** | **0,715** |
| `ctrl_labels` | Etiquetas de control | 0,420 | 0,670 | 0,432 | 0,677 |
| `ctrl_all` | Control + `all_feats` | 0,452 | 0,709 | 0,467 | 0,699 |
| `ctrl_dead` | Control + huecos en muerto | 0,366 | 0,600 | 0,459 | 0,713 |
| `ctrl_dead_all` | Control + huecos en muerto + `all_feats` | 0,406 | 0,643 | 0,496 | 0,753 |
| `ctrl_mixed_all` | Control + ambos patrones + `all_feats` | 0,453 | 0,711 | 0,483 | 0,750 |
| `ctrl_mixed_change` | Lo anterior + detector de cambios | 0,456 | 0,708 | 0,482 | 0,751 |

Cada columna solo se compara consigo misma: `ctrl` es una verdad distinta y más fácil.

### Ensayo con Metrica

Se entrenó con los 7 partidos de IDSSE y se puntuó en las partidas 1 y 2 de Metrica, con 3 degradaciones
cada una y costes medianos de IDSSE. Las etiquetas salen de sus eventos y de su balón ausente, así que
sustituyen de forma aproximada a las del reto. Sus posesiones duran unos 9 s.

| Variante | Huecos uniformes | Huecos en muerto |
|---|---|---|
| `ref` | 0,541 | 0,558 |
| `ctrl_labels` | 0,479 | 0,499 |
| `dead_gaps` | 0,479 | 0,593 |
| `shape` | 0,568 | 0,586 |
| `all_feats` | 0,591 | 0,610 |
| `dead_all` | 0,532 | 0,636 |
| **`mixed_all`** | **0,585** | **0,632** |
| `ctrl_all` | 0,510 | 0,528 |
| `ctrl_dead` | 0,439 | 0,521 |
| `ctrl_dead_all` | 0,469 | 0,550 |
| `ctrl_mixed_all` | 0,521 | 0,548 |

`mixed_all` mejora a `ref` en +0,044 y +0,073, en 6 de 6 repeticiones en ambos casos.

### Decisión por mejora

**Aceptadas**

- **`all_feats`:** +0,05 frente a `ref` en casi todas las pruebas y sin perder ningún partido en Metrica.
- **`mixed_all`:** añade el entrenamiento con ambos patrones. Gana hasta +0,03 si los huecos caen en balón
  muerto y casi no cuesta si son uniformes.

**Descartadas**

- **Etiquetas de control:** suben contra su propia verdad, pero bajan unas 0,06 solas contra `raw` y en
  Metrica. Riesgo asumido: si el reto sigue esa definición, se dejan unas 0,035 sin ganar. Los 14,2 s de
  los 5 minutos públicos apuntan a ella, pero salen de solo 17 posesiones. Esos 5 minutos no se usaron
  para decidir.
- **`dead_gaps` y `dead_all`:** caen con huecos uniformes (−0,065 y −0,010) y el patrón real no está
  confirmado.
- **`es`, `two_heads`, `symmetric`, `weighted`, `seeds3`, `lightgbm`:** dentro de ±0,01 contra `raw`.
- **`self_train`:** sin ganancia contra `raw` y baja contra `ctrl`.
- **`change`:** aporta frente a `ref`, pero no suma sobre la mejor base (+0,001 y −0,003 en
  `ctrl_mixed_change`).
- **MBR:** −0,005 (1/7) sobre `mixed_all`. El primer intento dio −0,276 por un error de diseño, que
  dividía también los costes por la temperatura. Tras corregirlo, las rutas muestreadas son realistas y
  sigue sin aportar.
- **Prior de Smart Tagging en el decodificador:** entre −0,001 y +0,005 en una simulación sobre IDSSE, sin
  mejorar ni con etiquetas perfectas. Incluye el coste reducido en transiciones. No se aplicó al partido
  del reto. De su análisis queda que los códigos `DEFENDING` y `DEFENSIVE` marcan al equipo sin balón y
  son pares espejo de los ofensivos.

**Sin probar**

- **PathCRF:** el entorno bloqueó ejecutar código externo. Solo se clonó y se leyó, sin ejecutar nada.
  Pide jugadores identificados y portero marcado, que el reto no da.
- **ELASTIC:** solo afina las etiquetas de control, ya descartadas.
- **Vídeo y audio:** los `.mp4` no están en este PC.

## Pendiente

1. Medir `mixed_all` con 3 degradaciones por partido, con costes anidados y ensayo con Metrica.
2. Integrar el resultado en `possession/train.py`, `possession/features.py` y `possession/pipeline.py`.
3. Comprobar que el fichero se regenera idéntico desde `train.py`, con un solo comando y en CPU.
4. Generar el CSV nuevo sin sobrescribir `possession_v2.csv` hasta validarlo.
5. Preparar el repositorio público y el documento técnico.
