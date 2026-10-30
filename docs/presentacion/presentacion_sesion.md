# Presentación al equipo — More than a Hack 2027, Fase 1 (posesión)

Guion diapositiva por diapositiva, alineado con `More than a Hack 2027 · Posesión.pptx`.
Cada diapositiva tiene **título**, **contenido** (lo que va en pantalla) y, cuando hace falta,
**notas** (lo que se cuenta).

---

## 1 — Portada

**Contenido:**
- MORE THAN A HACK 2027 · FASE 1
- **Posesión del balón frame a frame**
- Metrica Sports + FC Barcelona + MWCapital
- Entrega · 17/11/2026 · Estado a 29/10/2026

---

## 2 — El problema

**Título:** Decir, en cada frame del partido, quién tiene el balón

**Contenido:**
- **FC Barcelona – Athletic Club** · LaLiga 24/25, J2
- **199.319 frames** a 25 fps → un CSV con una etiqueta por frame
- **158.857 puntúan**: solo los de la 1ª y la 2ª parte (no cuentan antes del partido, el descanso
  ni después del final)

| Etiqueta | Significado |
|---|---|
| **H** | Barça (local) en posesión, balón vivo |
| **A** | Athletic (visitante) en posesión, balón vivo |
| **D** | Balón muerto (fuera, falta, córner, gol…), sin importar quién reanuda |
| **X** | Sin respuesta. Fuera de las dos partes no penaliza |

---

## 3 — Qué cuenta como posesión

**Título:** Las reglas de posesión

**Contenido:**
- **Siempre hay dueño:** con el balón en juego siempre lo tiene un equipo. Suelto o disputado,
  sigue siendo del último que lo controló
- **Cambia con control:** solo cambia cuando el rival consigue control (retener unos segundos o
  más de 2 pases seguidos)
- **No cambia por:** un toque, un desvío o un duelo aéreo sin control posterior
- `H H D D H H` = **dos posesiones** (posesión = tramo seguido de H o de A)
- La respuesta correcta la calcula Metrica a partir de **su tracking corregido**, que no tenemos

**Notas:** Nosotros solo tenemos la versión automática y sin corregir de ese tracking
(`BAR-ATH_tactical_atd.txt`). La respuesta correcta sale de la versión corregida, que es privada.

---

## 4 — Cómo nos puntúan

**Título:** La métrica premia posesiones enteras, no frames

**Contenido:**
- **spellF1 · ranking:** cada posesión que predecimos se compara con la real del mismo equipo
  que más se le solape. Cuenta como acierto si comparten suficiente tiempo; se repite exigiendo
  del 30 % al 70 % de solapamiento y se promedia
  - **1,00** · llegar 1 frame tarde: no penaliza
  - **0,98** · llegar 3 frames tarde: penaliza poco
  - **0,65** · llegar 1 s tarde: penaliza bastante
- **macroF1 · desempate:** acierto por frame, con el mismo peso para H, A y D
- **Ruido de la métrica:** ~0,04 en spellF1 · ~0,01 en macroF1

**Notas:** Las tres cifras de ejemplo son de la tabla de `EVALUATION.md`. Cada posesión cuenta
igual, dure 1 s o 40 s.

---

## 5 — Los datos que tenemos

**Título:** Qué nos dan (y qué problemas tiene)

| Fuente | Qué contiene | Problema |
|---|---|---|
| Tracking (ATD) | Dónde están los jugadores y el balón en cada frame, a qué velocidad se mueven y de qué equipo es cada jugador | Generado automáticamente y sin revisar. **El balón no aparece en el 41 % de los frames que puntúan** |
| Smart Tagging | 638 fases de juego con su equipo; entre ellas, 59 balones parados (banda, córner, puerta, falta, saque) | Automático, con errores. Ventanas de duración fija, no el instante exacto. Saques de banda sin equipo |
| Homografía | Correspondencia entre el campo y los píxeles del vídeo, por frame | — |
| Vídeo | Táctico (el de referencia) y TV (no sincronizado, 164.488 frames) | — |
| Ground truth público | Respuesta correcta de los 5 primeros minutos (17 posesiones) | Solo para comprobar que el pipeline funciona; no se puede usar para evaluar modelos |

---

## 6 — El baseline

**Título:** El modelo de referencia: una regla sencilla sobre el tracking

**Contenido:**
1. Frame con balón: la posesión es del equipo del **jugador más cercano** (distancia en metros
   del balón a cada jugador visible)
2. Sin balón → **D**
3. Suavizado: votación por mayoría en una ventana de ±2 s

- **Oficial · partido completo: spellF1 0,2433** · macroF1 0,6724

**Por qué falla:**
- Trata el 41 % de frames sin balón como balón muerto → trocea posesiones reales
- Cualquier rechace puede cambiar la posesión
- Ventana fija: si es corta, la etiqueta parpadea; si es larga, borra posesiones cortas

---

## 7 — Los entregables

**Título:** Qué hay que entregar antes del 17/11/2026

**Contenido:**
1. **El CSV** `<equipo>.csv`: cabecera `frame,label` · 199.319 filas · etiquetas H/A/D/X ·
   validado con `validate_submission.py`. **Es lo único que puntúa**
2. **Repo público** que regenere exactamente el mismo CSV: dependencias fijadas, instrucciones y
   un solo comando. **Sin los datos del partido**
3. **Documento técnico** `<equipo>.pdf`: enfoque, datos usados y descartados, datos de
   entrenamiento (fuente, quién etiquetó, cuánto), reglas de posesión, limitaciones y cómo
   reproducirlo

- Un formulario de Google por equipo (editable; vale la última respuesta)
- Equipos de 3 a 5 personas · Sin leaderboard ni feedback antes del cierre

---

## 8 — Reglas del juego

**Título:** Lo que podemos y no podemos hacer

| Permitido | Prohibido |
|---|---|
| Cualquier método, lenguaje o modelo (abierto o comercial) | Etiquetar o corregir a mano el partido del reto, aunque sea para entrenar o validar |
| Datos de otros partidos, incluso etiquetados a mano | Usar el ground truth privado, o los 5 min públicos para entrenar o ajustar |
| Mejorar o rehacer el tracking con algoritmos | Datos hechos por personas sobre este partido (Opta, StatsBomb, LaLiga…) |
| Etiquetas generadas automáticamente | Subir vídeo o datos del reto a un LLM en la nube |
| Ver el vídeo para entender y depurar | Modelos o datasets privados que la organización no pueda revisar |

- Todo debe poder inspeccionarse y explicarse en el documento técnico
- ¿Dudas? Preguntar antes: issue o hackathon.talentarena@mobileworldcapital.com

---

## 9 — Oportunidades de mejora

**Título:** Cinco mejoras, respaldadas por estudios publicados

| # | Qué mejoramos | En qué nos basamos |
|---|---|---|
| **M1** | **No cambiar la posesión por un rebote.** Solo pasa al otro equipo cuando el rival se queda con el balón un tiempo | [PathCRF, 2026](https://arxiv.org/abs/2602.12080) |
| **M2** | **Saber quién tiene el balón aunque no se vea.** Usamos dónde se vio por última vez y cómo están colocados los dos equipos: el que ataca está adelantado y abierto; el que defiende, replegado | [Vidal-Codina et al. 2022](https://arxiv.org/abs/2202.00804) · [Bischofberger et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11025972/) · [Ball Radar](https://arxiv.org/abs/2306.08206) |
| **M3** | **Detectar el balón muerto por lo que pasa en el campo,** no por si se ve el balón: jugadores andando, balón fuera de las líneas, jugadores colocados para un saque | [Lang et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/) |
| **M4** | **Buscar el balón en el vídeo** en los momentos en que el tracking lo pierde | [WASB-SBDT](https://arxiv.org/abs/2311.05237) |
| **M5** | **Entrenar y medir con 7 partidos de la Bundesliga** que sí traen la respuesta correcta en cada frame. Les quitamos el balón a ratos para que se parezcan a nuestros datos. El partido del reto nunca se usa para entrenar | [IDSSE, Bassek et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/) |

**Notas:** M5 es también nuestra forma legal de medir: cada partido de la Bundesliga se evalúa con
un modelo entrenado sin él, usando la métrica oficial.

---

## 10 — Resultados

**Título:** spellF1 ×2,5 frente al baseline

**Medido en los 7 partidos de la Bundesliga** (cada uno evaluado con un modelo que no lo vio):

| | spellF1 | macroF1 |
|---|---|---|
| Baseline | 0,191 | 0,439 |
| **Nuestro modelo** | **0,481** | **0,863** |

- Ganamos al baseline en **los 7 partidos** (0,42–0,56 frente a 0,15–0,25)

**Qué hemos aplicado:**

| | Estado |
|---|---|
| M1 · No cambiar por un rebote | ✅ |
| M2 · Posesión sin ver el balón | ✅ (sin PathCRF ni Ball Radar) |
| M3 · Balón muerto por lo que pasa en el campo | ✅ (del Smart Tagging solo usamos el saque inicial) |
| M4 · Balón desde el vídeo | ❌ Falta GPU |
| M5 · Entrenar con la Bundesliga | ✅ |

**Notas:** En los 5 minutos públicos (17 posesiones, solo orientativo): baseline 0,326 de
spellF1 con 26 posesiones; nuestro modelo 0,463 con 21. La referencia oficial del baseline en el
partido completo es 0,2433.

---

## 11 — Estado del entregable

**Título:** Qué tenemos listo

**Hecho:**
- Pipeline completo en `possession/`
- CSV válido; se regenera idéntico cada vez
- Dependencias fijadas e instrucciones (`SOLUTION.md`)
- Plan y checklist de cumplimiento (`docs/plan/plan_de_mejora.pdf`)

**Pendiente:**
- Repo público propio del equipo (ahora estamos en un clon del de Metrica)
- Documento técnico final
- Envío del formulario

---

## 12 — Próximos pasos

**Título:** Lo que falta y lo que se puede mejorar

**Pendiente:**
- **M4** · buscar el balón en el vídeo (WASB): necesita GPU
- **PathCRF** como señal extra cuando no se ve el balón
- Comprobar que el entrenamiento también da el mismo resultado al repetirlo

**Riesgos y mejoras posibles:**
- **Riesgo principal:** la Bundesliga marca el balón muerto ~45 % del tiempo; no sabemos si
  Metrica usa el mismo criterio
- **Afinar los bordes:** fijar el final de cada parada en el momento del saque (1 s de retraso
  baja el spellF1 a 0,65)
- **Usar todos los balones parados** del Smart Tagging, no solo el saque inicial
- **Más partidos de entrenamiento:** 3 partidos de muestra de Metrica (mismo proveedor),
  SkillCorner
- **Simular mejor nuestros datos:** quitar el balón sobre todo en jugadas con muchos jugadores
  y en balones aéreos, que es donde lo pierde nuestro tracking

---

## 13 — En una frase

> Pasamos de "el más cercano tiene el balón" a un modelo entrenado en partidos de la Bundesliga
> que entiende cuándo el balón está muerto y mantiene las posesiones enteras: **×2,5 en
> spellF1**, cumpliendo todas las reglas.

- Preguntas · Reparto de tareas para M4, PathCRF y el documento técnico
