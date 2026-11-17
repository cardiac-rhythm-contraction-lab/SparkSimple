# Controles MATLAB R2025b, 2026-10-06

Generados por scripts/matlab_numeric_controls.py con copias nuevas de
sparkParametersFast2.m y Blob4Diameters.m, sin dibujo, y trazas S8 leídas.
Orden: Cell4/N5 (276,71,94), (124,53,193), (220,157,273);
GT001/N5 (399,113,98), (535,67,160); GT001/N10 (316,113,270);
GT002/N10 (843,123,285); GT003/N5 (737,151,75), (526,181,345);
Cell2/N5 (163,130,328). values=[amp,tau,ror,t2p,FDHM,AMP,mass,r2].

Perfiles gausianos: limpio, centro semipíxel, outlier, oscilación y próximo
al borde. Interpolantes: seno, impulso y gaussiana. Error FIR observado
1.78e-11 (impulso), <8e-14 (curvas); tolerancia de prueba 2e-11.
Error de curvas gaussianas <=1.09e-7; tolerancia 1.2e-7 para este control,
no para todas las FWHM. Tau/r2 no son oráculos de equivalencia del solver;
las diferencias se conservan y cuantifican en el informe.

El ROI adicional Cell2/N10 (808,69,299) se obtuvo del volum MATLAB persistente
con índices Python [45:92,784:831,293:310], conv2 con gaussiana2d(9), mínimo
restado y normalización por el máximo. Blob4Diameters original devuelve
1.0213807118745768 µm; su tercer diámetro vacío se omite de la media.
No se alteró el volumen ni el programa originales. El nuevo control pequeño
requiere revisión científica junto con los cambios de comportamiento.

Reproducción (rutas reales del manifest y scratch nuevo):

```powershell
python scripts/matlab_numeric_controls.py --reference MATLAB --s8 S8 --scratch CONTROLES
matlab -batch "run('CONTROLES/run_controls.m')"
python scripts/prepare_fwhm_rejection_control.py --input NUEVO/Cell2_NOISE10 --scratch CONTROLES --reference MATLAB --matlab-volume VOLUM_MATLAB
# Copiar scripts/matlab_fwhm_roi_control.m al scratch y ejecutarlo allí.
matlab -batch "cd('CONTROLES');run('matlab_fwhm_roi_control.m')"
python scripts/pack_matlab_controls.py --scratch CONTROLES --output tests/fixtures/matlab_parity_controls.npz
```

VOLUM_MATLAB es input/MatlabAudit/zData0.mat del caso en el manifest S8.
El script de preparación no escribe en input ni en MATLAB.
