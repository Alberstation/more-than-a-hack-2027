# Continuar en el PC con GPU: M4 y refinamiento de spellF1

Estado al 2026-10-31, rama `feature-alberto`.

## Preparar el PC

```bash
git clone -b feature-alberto https://github.com/Alberstation/more-than-a-hack-2027.git
cd more-than-a-hack-2027
pip install -r requirements-solution.txt
```

Git no incluye los datos ni el modelo:

1. Copia `data/tactical/` a mano. Es el bundle del reto (vídeo, ATD, homografía, Smart Tagging) y
   su licencia prohíbe redistribuirlo, así que no lo subas a ningún servicio público.
2. Ejecuta `python possession/idsse.py` para descargar los 7 partidos (~2,5 GB) y crear sus cachés.
3. Ejecuta `python possession/train.py` para regenerar `data/models/hgb_v2.pkl`.

Antes de procesar nada en una GPU en la nube, revisa los términos del reto. Con GPU local no hay
ese problema.

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
| Sin perder el balón (sin degradar) | 0,614 | 0,926 | Techo aproximado de M4 |

- Ampliando el grid de costes, el modelo actual da 0,482 con `(switch=10, dead=80)`.
- Predecimos unas 280–300 posesiones por partido frente a 437 reales. El 30 % de las reales dura
  menos de 2 s y son las que se escapan.
- Los 0,614 son un techo, no una previsión: WASB no recuperará todo el balón.
- El oráculo de equipo usa información que ningún modelo tiene, así que 0,91 tampoco es alcanzable.
- Mejores costes con 3 degradaciones: `(switch=5, gain=2, dead=40)`. Se eligieron con los mismos
  7 partidos, así que el 0,514 es algo optimista.
- Pendiente de integrar en `train.py` y `pipeline.py`: señales de toque y 3 degradaciones por
  partido. El detector es opcional. Tarda unos 3 min por partido y aporta 0,009, dentro del ruido.
- Para repetir la prueba: `python possession/experiments/change_exp.py 3` y después
  `python possession/experiments/decode_exp.py oof_change_k3`.

## Riesgo abierto

Las posesiones duran 7,2 s de media en IDSSE y 14,2 s en los 5 minutos públicos. Si Metrica las
define de otro modo, el 0,48 no se traslada al partido del reto. Esos 5 minutos sirven para entender
el etiquetado (`CHALLENGE.md`), pero no para ajustar. Antes de cambiar etiquetas o parámetros por
ese motivo, pregunta a los organizadores (`RULES.md`).

## Siguiente paso: M4

1. Instalar [WASB-SBDT](https://github.com/nttcom/WASB-SBDT) con sus pesos de fútbol (MIT).
2. Medir la velocidad en un tramo corto. La estimación de 1–2,5 h para el partido no está medida.
3. Inferir en los frames sin balón y unos pocos alrededor. La referencia es
   `data/tactical/BAR-ATH_tactical.mp4`, cuya fila i es el frame i de la entrega.
4. Proyectar al campo con `BAR-ATH_tactical_homography.csv`. La proyección supone balón en el suelo,
   así que los balones aéreos quedan desplazados.
5. Filtrar candidatos con un modelo de movimiento y rellenar huecos cortos.
6. Cachear el resultado en `data/` y hacer que `atd.load()` o `pipeline.py` lo usen.
7. Volver a medir la cuota de balón ausente y ajustar `train.py` para que IDSSE se degrade a esa
   nueva cuota. Si no, el modelo se entrena en condiciones que ya no son las reales.

La validación sigue siendo en IDSSE, y el vídeo solo es un dato de entrada. No se pueden corregir
detecciones a mano en el partido del reto (regla 1). Afinar WASB con pseudo-etiquetas del propio
tracking queda pendiente de consultar a la organización.

## Reproducir los experimentos

Están en `possession/experiments/` y escriben en `data/experiments/`:

```bash
python possession/experiments/build_oof.py                    # predicciones por partido retenido
python possession/experiments/diag.py                         # oráculos y conteos
python possession/experiments/team_exp.py                     # base / sin degradar / toques
python possession/experiments/change_exp.py 1                 # K degradaciones por partido
python possession/experiments/decode_exp.py oof_change_k1     # grid del decodificador
```
