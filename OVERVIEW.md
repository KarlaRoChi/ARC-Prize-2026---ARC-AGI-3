# 🕹️ ARC Prize 2026 — ARC-AGI-3: Resumen del Overview Oficial de Kaggle

**Competición:** [ARC Prize 2026 - ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)  
**Organizador:** ARC Prize Foundation (François Chollet, Mike Knoop, Greg Kamradt y otros)  
**Competiciones hermanas en 2026:** [ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2) y un [paper track](https://www.kaggle.com/competitions/arc-prize-2026-paper-track) para documentar el enfoque de cualquiera de las dos.

---

## 1. 🎯 Propósito
Medir si un sistema **aprende habilidades nuevas sobre la marcha** en entornos interactivos ocultos, que no se pueden memorizar y que exigen exploración y razonamiento de varios pasos. Las tareas son resolubles por humanos.

## 2. 📏 Evaluación
- Score por juego de 0 a 100%. El 100% equivale a un humano: superar todos los niveles con el mismo número de acciones que los testers humanos (con tope en 100%).
- **Por nivel completado:** `min(acciones_humanas / acciones_agente, 1.0)` **al cuadrado** (0.5 → 0.25).
- **Por juego:** media ponderada de los niveles, con peso = índice del nivel (1-indexado). Los niveles tardíos pesan más.
- **Total:** media de los scores por juego.
- Detalle: [ARC-AGI-3 Scoring Methodology](https://docs.arcprize.org/methodology).

> Consecuencia práctica: completar niveles es imprescindible, pero cada acción de exploración extra en un nivel reduce su score de forma cuadrática.

## 3. 📤 Sumisión
- El archivo de sumisión **lo genera Kaggle automáticamente**: basta con que el agente actúe en algún juego.
- En el rerun, el notebook se conecta a un gateway local (`http://gateway:8001`) que sirve los juegos privados.

## 4. ⏱️ Requisitos de Código
- Solo vía Notebooks. CPU o GPU **≤ 9 h** de ejecución.
- **Internet desactivado.**
- Se permiten datos externos públicos y gratuitos, incluidos modelos preentrenados.
- **Aceleradores mejorados:** máquinas RTX 6000 (`g4-standard-48`) exclusivas de esta competición y siempre sin internet. Usarlas para otra cosa puede llevar a suspensión de la cuenta.

## 5. 🗓️ Calendario (23:59 UTC)
| Fecha | Hito |
|---|---|
| 25-mar-2026 | Inicio |
| 30-jun-2026 | Milestone 1 (opcional) |
| 30-sep-2026 | Milestone 2 (opcional) |
| **26-oct-2026** | Último día para aceptar las reglas y para fusionar equipos |
| **2-nov-2026** | Sumisión final |
| 4-dic-2026 | Anuncio de ganadores |

## 6. 🏆 Premios ($850,000)
- **Progress Prizes ($150,000):** LB final $40k / $15k / $10k / $5k / $5k, y milestones ($25k / $7.5k / $5k cada uno).
- **Bonus ($700,000):** solo si un equipo alcanza el 100%; se reparte entre los Top 5 con 100%.
- Los equipos premiados deben publicar su solución como open source.
