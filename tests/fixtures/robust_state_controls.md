# Controles de estado robusto — 2026-10-07

Derivados del runtime MATLAB R2025b ejecutado en una copia instrumentada de
fit.m y private/cfRobustNonlinearFit.m. Código fuente original, S8 y experimentos
de solo lectura. No se versionan copias del código MathWorks.

Orden: GT001/N5 (535,67,160), GT001/N10 (316,113,270), GT003/N5 (526,181,345),
gaussiana limpia y gaussiana con outlier. Incluye el estado final real MATLAB
(residuos, pesos, leverage, escala, SSE inicial, p, y) y su gof.rsquare, además
de los residuos anteriores usados para generar esos pesos.

Pruebas independientes de la optimización: R² absoluto 2e-12 (máximo observado
en los 12 controles 7.2e-14); pesos 1e-14 (máximo observado 4.5e-16).
El artefacto pequeño es una nueva referencia numérica, sin nuevo dataset biológico;
requiere revisión científica de procedencia. No cambia el ajuste de producción.

Generación: scripts/investigate_robust_solver.py prepare, ejecución de
run_investigation.m en scratch, después analyze --fixture
tests/fixtures/robust_state_controls.npz. Véase el informe de investigación.
