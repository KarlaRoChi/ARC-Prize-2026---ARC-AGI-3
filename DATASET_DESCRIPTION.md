# 📊 ARC Prize 2026 — ARC-AGI-3: Resumen del Dataset Description

**Documentación completa:** [docs.arcprize.org](https://docs.arcprize.org) · **Licencia de los datos:** Apache 2.0 · **Tamaño:** ~49 MB, 148 archivos

---

## 1. 📁 Archivos

| Ruta | Contenido | Uso |
|---|---|---|
| `ARC-AGI-3-Agents/` | Copia local del repo oficial [ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents): `main.py`, `agents/agent.py` (clase base), `agents/swarm.py`, plantillas | Framework donde se registra el agente propio |
| `arc_agi_3_wheels/` | Wheels offline (`arc_agi`, `arcengine`, numpy, pydantic, flask, …) para Python 3.12 | `pip install --no-index --find-links` dentro de Kaggle |
| `environment_files/<juego>/<versión>/` | 25 juegos públicos: `<juego>.py` (lógica del juego) + `metadata.json` (incluye `baseline_actions` humanas por nivel) | Desarrollo y CV local en modo offline |

Juegos públicos: `ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30`.

La evaluación usa **110 juegos privados distintos**: la mitad para el LB público y la otra mitad para el privado.

---

## 2. 🎮 Cómo Funcionan los Juegos

- El agente recibe **frames** (`FrameData`): `frame` (lista de grids de animación, cada uno de hasta 64×64 con valores 0–15 y origen (0,0) arriba a la izquierda), `state`, `levels_completed`, `win_levels` y `available_actions`.
- Estados del juego: `NOT_PLAYED`, `NOT_FINISHED`, `WIN`, `GAME_OVER`.
- Cada juego tiene varios niveles de dificultad creciente.

### Acciones
| Acción | Descripción |
|---|---|
| `RESET` | Empezar o reiniciar el juego |
| `ACTION1`–`ACTION5` | Acciones simples (p. ej. mover arriba/abajo/izquierda/derecha, interactuar) |
| `ACTION6` | Acción compleja con coordenadas `(x, y)` (clic) |
| `ACTION7` | Acción simple adicional |

El significado de cada acción **cambia por juego**: el agente tiene que descubrirlo explorando.

---

## 3. 🤖 Arquitectura de un Agente

Una subclase de `agents.agent.Agent` implementa:
- `is_done(frames, latest_frame)`: cuándo dejar de jugar.
- `choose_action(frames, latest_frame)`: la siguiente `GameAction` (con `set_data({"x":…, "y":…})` para `ACTION6`).

Un `Swarm` lanza un agente por juego en paralelo (un thread cada uno), abre un *scorecard*, juega y lo cierra.

**Modos del toolkit `arc_agi`** (`OPERATION_MODE`): `online` (API / gateway de Kaggle), `offline` (solo `environment_files` locales) y `normal` (ambos).

---

## 4. 🧮 Scoring

- Por nivel completado: `min(baseline_humano / acciones_agente, 1)²`.
- Por juego: media ponderada por índice de nivel.
- Total: media de juegos, de 0% a 100%.
- `baseline_actions` de cada juego público está en su `metadata.json` (p. ej. `ls20`: `[22, 123, 73, 84, 96, 192, 186]`).
