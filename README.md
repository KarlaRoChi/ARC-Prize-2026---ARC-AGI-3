# 🕹️ ARC Prize 2026 — ARC-AGI-3 (Interactive Reasoning)

> **Competencia Kaggle:** [ARC Prize 2026 - ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)  
> **Recurso de Cómputo:** 💻 CPU (0 slots GPU) en v1. El pool de la competición incluye RTX 6000 (`g4-standard-48`) para agentes con LLM.  
> **Internet:** ❌ Desactivado  
> **Métrica:** completitud × eficiencia frente a humanos, 0–100 (ver [OVERVIEW.md](OVERVIEW.md))  
> **Entregable:** lo genera Kaggle automáticamente durante el rerun; el notebook solo tiene que jugar.  
> **Fechas clave:** 26-oct-2026 (aceptar reglas / merge de equipos) · **2-nov-2026 (sumisión final)**

---

## 🎯 Objetivo de la Competencia

A diferencia de ARC-AGI-2 (predecir un grid a partir de ejemplos), aquí el agente **juega videojuegos desconocidos**:
recibe frames (grid de hasta 64×64 con colores 0–15), elige acciones (`RESET`, `ACTION1`–`ACTION7`, donde `ACTION6` es un clic con coordenadas `x, y`) y tiene que descubrir por exploración qué hace cada acción, cuál es el objetivo y cómo superar cada nivel.

La evaluación usa **110 juegos privados** que el agente nunca vio (55 para el LB público, 55 para el privado). Hay 25 juegos públicos en los datos para desarrollo local.

---

## 🏗️ Enfoque Técnico (v1: Graph Explorer)

[`src/graph_explorer_agent.py`](src/graph_explorer_agent.py) trata cada nivel como una máquina de estados determinista desconocida:

1. **Estado** = hash del grid visible.
2. **Acciones no probadas primero** en cada estado.
3. **BFS sobre el grafo de transiciones** conocido hasta el estado más cercano con acciones pendientes.
4. **Clics dirigidos:** `ACTION6` apunta al centro de cada objeto de color conectado.
5. **Memoria de fracasos:** las transiciones que llevan a `GAME_OVER` se vetan.
6. **Reinicio por nivel:** al completar un nivel se borra el grafo.

Es un baseline sin LLM. Los equipos punteros (LB 60+) usan agentes LLM (Qwen3.x) en las RTX 6000 sobre un *harness* de exploración ("duck harness" de Tufa Labs, solución pública del Milestone 2).

---

## 📁 Estructura del Directorio

```text
ARC Prize 2026 - ARC-AGI-3/ARC Prize 2026 - ARC-AGI-3/
├── kaggle_arc3_agent.ipynb      # Notebook de Kaggle (GENERADO desde src/, no editar a mano)
├── kernel-metadata.json         # Metadatos del kernel remoto (CPU, sin internet)
├── src/
│   └── graph_explorer_agent.py  # Código del agente (fuente única)
├── tools/
│   ├── build_notebook.py        # src/ → kaggle_arc3_agent.ipynb
│   └── local_eval.py            # Juega los 25 juegos públicos offline y calcula el score oficial
├── EXPERIMENT_LOG.md            # Registro de versiones y resultados
├── OVERVIEW.md                  # Resumen del overview oficial
├── DATASET_DESCRIPTION.md       # Resumen de datos, API de juegos y scoring
├── data/                        # (ignorado) datos de la competición para pruebas locales
└── submissions/                 # (ignorado) salidas descargadas del kernel
```

---

## 🚀 Cómo Trabajar

```powershell
# 1. Descargar los datos de la competición (una vez)
python -m kaggle competitions download arc-prize-2026-arc-agi-3 -p data
Expand-Archive data\arc-prize-2026-arc-agi-3.zip data

# 2. Entorno local (Python >= 3.12) con el toolkit oficial
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install data\arc_agi_3_wheels\arc_agi-0.9.8-py3-none-any.whl data\arc_agi_3_wheels\arcengine-0.9.3-py3-none-any.whl python-dotenv pandas pyarrow

# 3. Editar src/graph_explorer_agent.py y medir en los 25 juegos públicos
.venv\Scripts\python tools\local_eval.py --data data
.venv\Scripts\python tools\local_eval.py --data data --games ls20,vc33 --max-actions 1500

# 4. Regenerar el notebook y lanzarlo en Kaggle (desde la raíz Kaggle/)
.venv\Scripts\python tools\build_notebook.py
python kaggle_manager.py run arc3
```

El *commit run* en Kaggle juega 3 juegos públicos (`ls20`, `vc33`, `r11l`) como prueba de humo e imprime `*** Smoke score`. Para puntuar en el LB hay que **enviar la versión del notebook** (code competition):

```powershell
python -m kaggle competitions submit arc-prize-2026-arc-agi-3 -k rochikarla/arc-prize-2026-arc-agi-3-agent -v <versión> -f submission.parquet -m "v1 graph explorer"
```
