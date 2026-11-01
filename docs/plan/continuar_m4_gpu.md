# Continuar en el PC con GPU: M4 y refinamiento de spellF1

Estado al 2026-10-31, rama `feature-alberto`. Entrega: 17 de noviembre de 2026.

M4 (recuperar el balón del vídeo) queda descartado: no superó la puerta 1. Las secciones de la puerta
2 y de integración solo se retoman si un dato nuevo cambia ese resultado.

## Orden de trabajo

| # | Tarea | Necesita | Estado |
|---|---|---|---|
| 1 | Generar el CSV de v2 como respaldo | Datos del reto | Generado y validado; no se sube aún |
| 2 | Consultas a la organización (ver abajo) | Nada | Ninguna pendiente |
| 3 | **Puerta 1:** dosis-respuesta, `dose_exp.py` | IDSSE y ATD | Hecho: no superada |
| 4 | Medir `mixed_all` con 3 degradaciones e integrar en `train.py` y `pipeline.py` | IDSSE y ATD | Siguiente paso |
| 5 | Techo de PathCRF | Permiso para ejecutar código externo | Sin ejecutar |
| 6 | Repositorio público y documento técnico | Todo lo anterior | Empezar antes del 10/11 |

El CSV de v2 (`evaluation/results/possession_v2.csv`) da 317 posesiones de 11,1 s de media, valida el
formato y se regenera con el mismo SHA-256 desde el modelo guardado. Falta comprobar la regeneración
desde `train.py`.

## Preparar el PC

Python 3.12 con las versiones fijadas. Con `uv`:

```bash
git clone -b feature-alberto https://github.com/Alberstation/more-than-a-hack-2027.git
cd more-than-a-hack-2027
uv python install 3.12
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements-solution.txt
```

Git no incluye los datos ni el modelo:

1. Copia `data/tactical/` a mano. Es el bundle del reto (vídeo, ATD, homografía, Smart Tagging) y
   su licencia prohíbe redistribuirlo, así que no lo subas a ningún servicio público.
2. Ejecuta `python possession/idsse.py` para descargar los 7 partidos (~2,5 GB) y crear sus cachés.
3. Ejecuta `python possession/train.py` para regenerar `data/models/hgb_v2.pkl`.

Todo se procesa en máquinas del equipo con GPU local. Ningún dato del reto va a la nube, como fija
`plan_de_mejora.tex`.

## Qué sabemos

Todo se midió con validación dejando un partido fuera, sobre IDSSE degradado y con la métrica
oficial. Ningún número se eligió con los 5 minutos públicos.

| Variante | spellF1 | macroF1 | Lectura |
|---|---|---|---|
| Modelo actual (v2), costes del grid original | 0,480 | 0,863 | Punto de partida |
| Modelo actual, balón muerto real | 0,516 | 0,900 | Oráculo: el balón muerto no es el límite |
| Modelo actual, equipo real | 0,909 | 0,957 | Oráculo: el equipo H/A es el límite |
| + señales de toque | 0,488 | 0,869 | Mejora dentro del ruido |
| + señales de toque + detector de cambios | 0,496 | 0,871 | 5 partidos suben y 2 bajan; no integrado |
| Igual, entrenando con 3 degradaciones por partido | 0,514 | 0,876 | Mejor resultado; sube en 7 de 7 frente a 1 degradación |
| 3 degradaciones, sin detector | 0,506 | 0,874 | Las degradaciones aportan más que el detector |
| Sin ninguna degradación (balón y jugadores) | 0,614 | 0,926 | Cota superior holgada de M4 |

- Ampliando el grid de costes, el modelo actual da 0,482 con `(switch=10, dead=80)`.
- Predecimos unas 280–300 posesiones por partido frente a 437 reales. El 30 % de las reales dura
  menos de 2 s y son las que se escapan.
- Los 0,614 no son el techo de M4: esa variante también recupera los jugadores, y M4 solo da balón.
  El techo es el brazo `ball_all` de `dose_exp.py`: 0,568, aún provisional.
- El oráculo de equipo usa información que ningún modelo tiene, así que 0,91 tampoco es alcanzable.
- `degrade()` coloca los huecos al azar, pero `DATA.md` dice que el balón falta sobre todo en los
  frames que importan. El 0,48 puede ser optimista; el brazo `hard41` mide cuánto.
- Mejores costes con 3 degradaciones: `(switch=5, gain=2, dead=40)`. Se eligieron con los mismos
  7 partidos, así que el 0,514 es algo optimista.
- Pendiente de integrar en `train.py` y `pipeline.py`: señales de toque y 3 degradaciones por
  partido. El detector es opcional. Tarda unos 3 min por partido y aporta 0,009, dentro del ruido.
- Para repetir la prueba: `python possession/experiments/change_exp.py 3` y después
  `python possession/experiments/decode_exp.py oof_change_k3`.

## Resultado de las mejoras

Medido con `improve.py`, `decoder_exp.py` y `metrica_exp.py`. Diferencia pareada de spellF1 frente a
`ref`; costes elegidos con los otros 6 partidos. En Metrica: partidas 1 y 2 con 3 degradaciones cada
una, costes medianos de IDSSE y etiquetas construidas desde sus eventos.

| Variante | IDSSE huecos uniformes | IDSSE huecos en muerto | Metrica uniformes | Metrica en muerto | Decisión |
|---|---|---|---|---|---|
| `all_feats` (toque, forma y control) | +0,049 (7/7) | +0,050 (7/7) | +0,050 (6/6) | +0,052 (6/6) | Aceptada |
| `mixed_all` (`all_feats` + entrenar con ambos huecos) | +0,048 (7/7) | +0,076 (7/7) | +0,044 (6/6) | +0,073 (6/6) | **Elegida** |
| `dead_all` (solo huecos en muerto) | −0,010 (2/7) | +0,083 (7/7) | −0,009 (2/6) | +0,078 (6/6) | Rechazada: depende del patrón real |
| `ctrl_labels` (etiquetas de control) | −0,070 (0/7) | −0,077 (0/7) | −0,062 | −0,060 | Rechazada, ver abajo |

Todas las filas puntúan contra `BallPossession` crudo en IDSSE. `mixed_all` sobre esa referencia da
0,538 con huecos uniformes y 0,585 con huecos en muerto, frente a 0,490 y 0,510.

Las etiquetas de control suben contra su propia verdad (`ctrl_mixed_all` 0,750 frente a 0,715 de
`mixed_all`, huecos en muerto), pero bajan contra crudo y en Metrica (0,521–0,548 frente a
0,585–0,632). No se adoptan. Si el reto define la posesión como `ctrl`, esta decisión cuesta unos 0,035.

Sin aporte: `es`, `two_heads`, `symmetric`, `weighted`, `seeds3`, `lightgbm` y `self_train` (dentro de
±0,01 contra etiquetas crudas), y `change` (no suma sobre la mejor base: +0,001 y −0,003). MBR: −0,005 (1/7) sobre
`mixed_all`. Smart Tagging simulado en el decodificador: entre −0,001 y +0,005, sin mejorar ni con
etiquetas perfectas; el coste por transición está incluido.

Sin probar: PathCRF, ELASTIC (solo afina las etiquetas de control, ya rechazadas) y vídeo o audio
(no hay `.mp4` en este PC). PathCRF pide jugadores identificados y portero marcado, que el reto no da.

## Riesgos abiertos

**Definición de posesión.** Las posesiones duran 7,2 s de media en IDSSE y 14,2 s en los 5 minutos
públicos. Si Metrica las define de otro modo, el 0,48 no se traslada al partido del reto, y el valor
de M4 medido en IDSSE tampoco. Esos 5 minutos sirven para entender el etiquetado (`CHALLENGE.md`),
pero no para ajustar. Antes de cambiar etiquetas o parámetros por ese motivo, pregunta a los
organizadores (`RULES.md`).

**Reproducibilidad.** La regla 3 exige regenerar el mismo fichero, no uno parecido. El resultado de
WASB no se puede publicar si se considera dato derivado por frame, así que los organizadores tendrían
que regenerarlo, y una GPU distinta puede dar otras detecciones. Ese cambio llegaría a `train.py`, al
modelo y al CSV. Ver los criterios de la puerta 2.

## Consultas a la organización

Issue en el repositorio oficial o `hackathon.talentarena@mobileworldcapital.com`:

Decisión del 2026-11-01: no se consulta la reproducibilidad, ni el uso de los 5 minutos públicos para
entender las etiquetas, ni los datos externos. Las normas ya cubren los dos últimos, y la entrega
seguirá en CPU. Tampoco se consulta la definición de posesión: `CHALLENGE.md` ya dice que un toque sin
control no cambia la posesión y que, ante cualquier duda, manda el ground truth público.

## Puerta 1: ¿compensa recuperar balón?

`dose_exp.py` mantiene siempre los jugadores degradados y reduce el balón ausente del 41 % al 30, 20
y 10 %, con balón recuperado ideal y con ruido y falsos positivos supuestos. Los brazos están
pareados por partido.

Sigue adelante solo si `noisy20` gana **al menos +0,03 de spellF1** a `ref41` y mejora en **5 o más
de los 7 partidos**. Se elige `noisy20` por ser una recuperación de la mitad del balón ausente.

Resultado del 2026-11-01, con los huecos reales del reto (150 tramos, mediana 10,7 s) y error supuesto
de 1,5 m y 3 % de falsos positivos:

| Brazo | Balón ausente | spellF1 | macroF1 | Frente a `ref41` | Partidos que mejoran |
|---|---|---|---|---|---|
| `ref41` | 40 % | 0,488 | 0,867 | — | — |
| `hard41` (huecos en los cambios) | 40 % | 0,467 | 0,860 | −0,021 | 1 de 7 |
| `ideal30` | 30 % | 0,508 | 0,877 | +0,020 | 7 de 7 |
| `ideal20` | 19 % | 0,527 | 0,888 | +0,040 | 7 de 7 |
| `ideal10` | 9 % | 0,549 | 0,897 | +0,061 | 7 de 7 |
| `noisy30` | 30 % | 0,511 | 0,875 | +0,023 | 6 de 7 |
| `noisy20` | 19 % | 0,514 | 0,879 | +0,026 | 7 de 7 |
| `noisy10` | 9 % | 0,533 | 0,889 | +0,046 | 7 de 7 |
| `ball_all` | 0 % | 0,568 | 0,911 | +0,080 | 7 de 7 |
| `dead41` (huecos en balón muerto) | 41 % | 0,539 | 0,892 | +0,051 | 7 de 7 |
| `dead_m4` (`dead41` sin huecos en juego) | 36 % | 0,564 | 0,907 | +0,076 | 7 de 7 |

**La puerta 1 no se supera y M4 queda descartado salvo dato nuevo.** `noisy20` se queda en +0,026.

El 83 % de los frames sin balón del reto cae en tramos que v2 predice como balón muerto: falta en el
75 % de esos frames y en el 12,7 % de los predichos en juego. Son predicciones, no etiquetas, y
`DATA.md` afirma que el balón falta sobre todo en frames importantes, así que la cifra no es
definitiva. Si se confirma, solo el 7 % de los frames puntuados es balón en juego perdido, y la
ganancia comparable de M4 es `dead_m4` frente a `dead41`: +0,025 aun con recuperación perfecta.

`dead41` también indica que `degrade()` no modela bien los huecos. El modelo actual no puede aprender
que un hueco largo sugiere balón muerto, porque en su entrenamiento los huecos caen al azar.
`dose_exp.py` no mezcla resultados con huecos sustitutos y reales.

## Puerta 2: prueba acotada de WASB

Máximo 2 días, sobre un tramo corto. WASB solo documenta la evaluación sobre sus datasets, así que
hay que escribir la inferencia sobre vídeo propio. Se probó con Python 3.8 y CUDA 11.3: usa un
entorno separado y comprueba que funciona con una RTX 4060. Registra versiones, enlace y hash de los
pesos. Están en Google Drive sin licencia propia indicada; confírmala, porque el MIT consta solo
para el repositorio.

Infiere sobre un tramo contiguo, no solo sobre los frames sin balón, y compara con el balón del ATD
donde exista. Es una comparación entre algoritmos, sin etiquetas humanas, aunque esos frames son más
fáciles que los perdidos. Mide:

- Velocidad real. La estimación de 1–2,5 h no está medida.
- Cobertura sobre los frames sin balón.
- Error en metros, que sustituye a `ERROR_M`, y falsos positivos, que sustituyen a `OUTLIER_SHARE`.
  Repite `dose_exp.py` con esos valores.
- Determinismo: dos ejecuciones, en CPU y GPU si se puede. Tras redondear, las posiciones deben ser
  idénticas. Comprueba que el vídeo decodificado da 199.319 frames.

Sigue adelante solo si el brazo con los valores medidos sigue cumpliendo la puerta 1 y el resultado
es determinista, o la organización ha aceptado otra vía.

## Integrar M4

1. Infiere sobre los frames puntuados y decodifica en secuencia, sin saltos. La referencia es
   `data/tactical/BAR-ATH_tactical.mp4`, cuya fila i es el frame i de la entrega.
2. Proyecta con `unproject()` de `starter/load_homography.py`: la matriz va de campo a píxeles y
   devuelve 0–1, así que luego hay que pasar a metros. Supone balón en el suelo, por lo que los
   balones aéreos quedan desplazados.
3. Filtra candidatos con un modelo de movimiento, rellena huecos cortos y descarta las detecciones
   fuera del campo que no enlacen con la trayectoria. `outside_distance` arrastra la última posición
   vista, así que un balón de repuesto simularía balón muerto durante todo un hueco.
4. Cachea en `data/` y carga con un módulo nuevo usado por `train.py` y `pipeline.py`. `atd.load()`
   queda intacto para que `train.py` no degrade IDSSE a una cuota antigua.
5. Haz que `degrade()` reproduzca la cuota nueva y el error medido de WASB. Si no, el modelo se
   entrena en condiciones que ya no son las reales.

La validación sigue siendo en IDSSE, y el vídeo solo es un dato de entrada. No se pueden corregir
detecciones a mano en el partido del reto (regla 1). `RULES.md` permite pseudo-etiquetas si ninguna
persona las decide, pero WASB no publica código de entrenamiento, así que afinarlo queda fuera de
este plan.

## Reproducir los experimentos

Están en `possession/experiments/` y escriben en `data/experiments/`:

```bash
python possession/experiments/dose_exp.py                     # puerta 1, solo IDSSE
python possession/experiments/build_oof.py                    # predicciones por partido retenido
python possession/experiments/diag.py                         # oráculos y conteos
python possession/experiments/team_exp.py                     # base / sin degradar / toques
python possession/experiments/change_exp.py 1                 # K degradaciones por partido
python possession/experiments/decode_exp.py oof_change_k1     # grid del decodificador
python possession/experiments/improve.py prep                 # mejoras: partidos degradados
python possession/experiments/improve.py train; python possession/experiments/improve.py score
python possession/experiments/decoder_exp.py mbr mixed_all dead raw
python possession/experiments/decoder_exp.py tags mixed_all dead raw
python possession/experiments/metrica_exp.py prep             # requiere data/external/metrica
python possession/experiments/metrica_exp.py run ref mixed_all
```

Salvo `dose_exp.py`, todos necesitan `data/tactical/`.
