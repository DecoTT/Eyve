export PYTHONIOENCODING=utf-8
PY="D:/Desarrollo/Claude Code/Eyve/2.1-dev/.venv/Scripts/python.exe"
S="flores|canasta,puntos|canasta,flores|sarga_fina,diamantes|sarga,rayas|sarga,puntos|ninguno"
for p in rampa:0.04 ganancia:0.9 escala:1.003 desliza:6; do
  n=${p/:/_}
  "$PY" medir.py dev resultados/rob_$n.json --solo "$S" --workers 6 --limpia 150 --perturba $p > resultados/rob_$n.log 2>&1
done
"$PY" medir.py dev resultados/rob_tilt_2.1.json --solo "$S" --workers 6 --limpia 150 --tilt 2.1 > resultados/rob_tilt.log 2>&1
for f in 0.30 0.20 0.10; do
  "$PY" medir.py dev resultados/fuerza_$f.json --solo "$S" --workers 6 --limpia 1 --K 6 --solo-fantasma --fuerza $f > resultados/fuerza_$f.log 2>&1
done
