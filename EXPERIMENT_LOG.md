# 🕹️ Registro de Experimentos: ARC Prize 2026 — ARC-AGI-3

**Métrica:** promedio por juego de `min(acciones_humanas / acciones_agente, 1)²` ponderado por nivel (0–100).  
**Recurso:** CPU (0 slots GPU).  
**CV local:** `python tools/local_eval.py --data data` sobre los 25 juegos públicos de `environment_files/`.

| Versión | Fecha | Hipótesis / Descripción | Cambios Técnicos | CV Local (25 públicos) | LB Score | Estado / Conclusiones |
|---|---|---|---|---|---|---|
| `v1_graph_explorer` | 2026-10-17 | Exploración sistemática del grafo de estados: sin LLM, cada nivel como máquina de estados determinista | Estado = hash del grid; acciones no probadas primero; BFS al estado conocido más cercano con acciones pendientes; `ACTION6` sobre centros de objetos conectados; transiciones a `GAME_OVER` vetadas; grafo reiniciado por nivel; tope de 4000 acciones por juego | `0.28` (10 niveles en 8 de 25 juegos) | Pendiente | Baseline funcional. Completa niveles pero con muchas acciones, y la eficiencia al cuadrado castiga fuerte. Smoke de notebook (ls20, vc33, r11l): 2.27 |
