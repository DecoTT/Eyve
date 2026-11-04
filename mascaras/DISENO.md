# Máscara aprendida: prototipo para Patrón

Rama `mascaras-proto`, sin mezclar a `2.1-dev`. Medido el 4-nov-2026 sobre la tela sintética de
`eyve.demo.textile`. **No se probó con tela real ni con cámara.**

## Resultado en corto

- Reproduje la línea base del video 6 sin diferencia: 237 de 240, 0 falsas alarmas en 1,280 cuadros
  sanos y los mismos 3 fantasmas escapados. Las 60 casillas salen idénticas a las del 28-oct.
- Con la máscara: **240 de 240** en esos mismos fallos, y **360 de 360** en fallos nuevos sobre un
  rollo que no se usó para afinar.
- El precio: **en rayas · canasta marca la costura del rollo**, 138 de 560 cuadros de tela limpia.
  En los otros 19 materiales, 0 de 10,640.
- Costo: 35 ms por cuadro, contra 33 ms de Patrón. Aprender tarda unos 4 s.

## Cómo funciona

`eyve/modules/mascara.py`, clase `MascaraAprendida`. No está en `MODULE_REGISTRY`, así que la
aplicación no cambia.

**Aprender.** Con 24 cuadros buenos, unos 8 s de línea:

1. La autocorrelación da la retícula: dos vectores, ajustados con todos los picos que caen en ella.
2. Los cuadros se doblan sobre una celda, en tres vueltas, dejando fuera los píxeles que no calzan.
   Así un fallo que se coló entre los cuadros buenos no se aprende.
3. Sale lo que se guarda: la celda, la tolerancia de cada punto, las zonas (tinta o tela limpia) y
   la viñeta de la cámara.
4. Uno de cada cuatro cuadros no entra a la celda: con esos se mide el piso de ruido.

**Aprobar.** `aprender()` deja la máscara «por aprobar», y así no marca nada. `vista()` muestra lo
aprendido, sus zonas y lo que no calza en un cuadro. `revisar_mascara.py` es la ventana con Aprobar
y Rechazar. La aprobación se guarda con una huella del contenido: una máscara alterada la pierde,
en memoria y en el archivo.

**Aplicar, por cuadro:**

1. Se busca la pose del material: traslación y deformación suave, de grueso a fino, sin los píxeles
   que no calzan.
2. Se pone la celda en esa pose y se saca la diferencia, ya sin la luz que cambió después de aprender.
3. Se promedia a la escala de una celda, por separado en tinta y en tela, y se mide la energía de más.
4. Si algún canal pasa 4.5 veces su piso (la energía, 2.5 veces eso), hay región. Si el cuadro no
   calza con la máscara, no juzga y lo avisa.

**Por qué ve lo que Patrón no ve.** Patrón compara contra copias del mismo material, y un fantasma es
una copia. Además toma valor absoluto, así que el hilo del tejido suma. La máscara conserva el signo:
el fantasma solo oscurece, el hilo sube y baja, y al promediar uno se acumula y el otro se cancela.

Los dos motores corren juntos: `fusionar()` manda la máscara y Patrón al mismo tracker, y lo que la
máscara ya marcó Patrón no lo repite, igual que hace con YOLO.

## Medición

`medir.py` repite la medición del video y corre tres seguimientos sobre los mismos cuadros: Patrón
solo («antes»), máscara sola, y los dos juntos («después»).

Separación de datos:

| Uso | Rollo | Semilla | Fallos |
|---|---|---|---|
| Aprender la máscara | 1 | 5 | Ninguno |
| Afinar | 3 | 3001 | 120 |
| Pruebas automáticas | 2 | Varias | Casos puntuales |
| Fallos del video | 0 | 2027 | 240 |
| Fallos nuevos | 4 | 4242 | 360 |

Hizo falta crear rollos: `TextilePattern` siembra su hilo con semillas fijas, y dos telas iguales son
la misma tela píxel a píxel aunque cambie `seed`. `rollos.py` cambia esas semillas: misma impresión,
otro hilo. La tela limpia se recorre una vuelta completa, 560 cuadros por material.

**Fallos del video** (rollo 0). «Lejos» cuenta cuadros con una marca a más de 40 px del fallo.

| Material | Fantasma antes | Fantasma después | Movido y falta antes | Movido y falta después | Falsas antes | Falsas después | Lejos antes | Lejos después |
|---|---|---|---|---|---|---|---|---|
| diamantes · sarga | 4/4 · 98 % | 4/4 · 98 % | 8/8 · 97 % | 8/8 · 98 % | 0/560 | 0/560 | 0 | 0 |
| diamantes · tafetán | 4/4 · 96 % | 4/4 · 98 % | 8/8 · 97 % | 8/8 · 98 % | 0/560 | 0/560 | 0 | 0 |
| diamantes · sarga fina | 4/4 · 90 % | 4/4 · 98 % | 8/8 · 97 % | 8/8 · 98 % | 0/560 | 0/560 | 0 | 0 |
| diamantes · canasta | 4/4 · 71 % | 4/4 · 95 % | 8/8 · 95 % | 8/8 · 98 % | 0/560 | 0/560 | 0 | 0 |
| diamantes · liso | 4/4 · 94 % | 4/4 · 98 % | 8/8 · 97 % | 8/8 · 98 % | 0/560 | 0/560 | 0 | 0 |
| flores · sarga | 4/4 · 93 % | 4/4 · 98 % | 8/8 · 98 % | 8/8 · 98 % | 0/560 | 0/560 | 0 | 0 |
| flores · tafetán | 4/4 · 98 % | 4/4 · 98 % | 8/8 · 98 % | 8/8 · 98 % | 0/560 | 0/560 | 1 | 1 |
| **flores · sarga fina** | **3/4 · 74 %** | **4/4 · 98 %** | 8/8 · 97 % | 8/8 · 98 % | 0/560 | 0/560 | 0 | 0 |
| **flores · canasta** | **3/4 · 43 %** | **4/4 · 98 %** | 8/8 · 97 % | 8/8 · 98 % | 0/560 | 0/560 | 0 | 0 |
| flores · liso | 4/4 · 80 % | 4/4 · 97 % | 8/8 · 98 % | 8/8 · 98 % | 0/560 | 0/560 | 0 | 0 |
| rayas · sarga | 4/4 · 91 % | 4/4 · 98 % | 8/8 · 96 % | 8/8 · 98 % | 0/560 | 0/560 | 1 | 1 |
| rayas · tafetán | 4/4 · 96 % | 4/4 · 98 % | 8/8 · 97 % | 8/8 · 98 % | 0/560 | 0/560 | 0 | 0 |
| rayas · sarga fina | 4/4 · 96 % | 4/4 · 98 % | 8/8 · 97 % | 8/8 · 98 % | 0/560 | 0/560 | 0 | 0 |
| rayas · canasta | 4/4 · 62 % | 4/4 · 98 % | 8/8 · 98 % | 8/8 · 98 % | 0/560 | **138/560** | 0 | 0 |
| rayas · liso | 4/4 · 90 % | 4/4 · 98 % | 8/8 · 98 % | 8/8 · 98 % | 0/560 | 0/560 | 0 | 0 |
| puntos · sarga | 4/4 · 86 % | 4/4 · 98 % | 8/8 · 97 % | 8/8 · 98 % | 0/560 | 0/560 | 0 | 0 |
| puntos · tafetán | 4/4 · 84 % | 4/4 · 95 % | 8/8 · 97 % | 8/8 · 98 % | 0/560 | 0/560 | 0 | 0 |
| puntos · sarga fina | 4/4 · 92 % | 4/4 · 98 % | 8/8 · 97 % | 8/8 · 98 % | 0/560 | 0/560 | 0 | 0 |
| **puntos · canasta** | **3/4 · 82 %** | **4/4 · 98 %** | 8/8 · 97 % | 8/8 · 98 % | 0/560 | 0/560 | 0 | 0 |
| puntos · liso | 4/4 · 96 % | 4/4 · 98 % | 8/8 · 96 % | 8/8 · 98 % | 0/560 | 0/560 | 0 | 0 |

**Totales**

| Conjunto | Fantasma antes | Fantasma después | Movido y falta antes | Movido y falta después | Falsas antes | Falsas después | Lejos antes | Lejos después |
|---|---|---|---|---|---|---|---|---|
| Video (rollo 0) | 77/80 · 86.3 % | **80/80 · 97.8 %** | 160/160 · 97.1 % | 160/160 · 98.1 % | 0/11,200 | **138/11,200** | 2 | 2 |
| Nuevos (rollo 4) | 117/120 · 87.5 % | **120/120 · 97.8 %** | 240/240 · 97.1 % | 240/240 · 98.0 % | 0/11,200 | **135/11,200** | 0 | 0 |

Los porcentajes son de cuadros con el fallo confirmado. Los 2 cuadros «lejos» del conjunto del video
vienen de Patrón, no de la máscara: el «después» no los agrega. Los 3 escapes del conjunto nuevo también
son fantasmas sobre canasta: diamantes, flores y rayas.

**Las falsas alarmas son la costura del rollo.** Las 138 y las 135 están todas en rayas · canasta. En
un muestreo de 80 cuadros por rollo (rollos 0 y 4), el centro de las 38 regiones cayó en la tela entre
x = 3,775 y 3,826, o en x = 7, ya del otro lado: la costura está en 3,840. La tela sintética de verdad tiene ahí una diferencia: al
redibujar el motivo en el empalme, la raya queda distinta (3.93 contra 3.64 en la diferencia contra el
periodo anterior). En una línea real sería una marca verdadera, pero en esta medición es tela
«limpia» y cuenta como falsa alarma. No la quité de la tabla.

**Hasta dónde llega.** Fantasmas más tenues que los de la demo (0.45), 6 materiales, 36 fantasmas por
fila. Aquí se ve que la prueba puede fallar:

| Fuerza del fantasma | Patrón | Máscara |
|---|---|---|
| 0.30 | 29/36 · 70 % | 36/36 · 97 % |
| 0.20 | 21/36 · 48 % | 34/36 · 94 % |
| 0.10 | 7/36 · 13 % | 25/36 · 67 % |

**Si algo cambia después de aprender.** Conjunto de afinar, 6 materiales, 150 cuadros limpios por
material. Estos cambios solo se aplican a los cuadros de prueba.

| Cambio | Fantasma con máscara | Falsas de la máscara | Falsas de Patrón |
|---|---|---|---|
| Rampa de luz de 4 % | 12/12 | 0/900 | 0/900 |
| Lámpara baja 10 % | 12/12 | 0/900 | 0/900 |
| Tela 0.3 % estirada | 12/12 | 0/900 | 185/900 |
| Tela 0.3° más inclinada | 12/12 | 0/900 | 0/900 |
| Cámara 6 px corrida | 12/12 | **209/900** | 0/900 |

Lo de la cámara corrida viene de la esquina superior izquierda: en una revisión de flores · sarga, las
regiones que vi caían ahí, donde la cuña de la rotación simulada entra a la zona que se juzga. No
revisé los otros dos materiales que marcaron (flores · canasta, puntos · canasta). Una cámara real no
tiene esa cuña, pero **eso no está probado**.

## Pruebas

`tests/test_mascara.py`, agregada a `run_all.py`. Comprueba, entre otras cosas, los 3 escapes uno por
uno, que la caja caiga sobre el fallo y deje de tocarlo si se corre 300 px, la tela limpia de un rollo
no visto, la aprobación, el material que no calza y un fallo entre los cuadros buenos.

**Suite completa en esta rama: 38 de 38**, con `test_demo_e2e.py` usando una copia local del modelo de
la demo (`resultados/suite_resumen.txt`).

`mutar.py` rompe el código a propósito en una copia: **11 mutaciones, las 11 detectadas**. Una primera
versión de la lista tenía una mutación que no cambiaba el comportamiento; la reemplacé.

Las pruebas encontraron fallos reales del prototipo: un cuadro liso «aprendía» una retícula, una falta
de tinta salía como tinta de más, y un fallo entre los cuadros buenos dejaba a la máscara ciega.

## Cómo se llegó aquí

El afinado se hizo con el rollo 3 y el umbral se subió después de una falsa alarma que apareció en el
rollo 2. El conjunto nuevo se midió dos veces, y conviene saberlo: una versión anterior dio 360/360 y
0 falsas, pero fallaba con 0.3° de inclinación, y al corregir eso apareció la costura. Esa medición
quedó en `resultados/v1/`. El conjunto nuevo no se usó para decidir ningún cambio. Lo que se ve en la
tabla es la versión final.

## Límites

- **Solo tela sintética.** Esa tela se repite a la perfección. Una tela real se estira de forma
  desigual, cambia de tono entre lotes y trae ruido de sensor. Es el riesgo principal.
- **Sensibilidad extrema.** El piso está cerca de 0.1 niveles de gris promediados. Por eso vio la
  costura. El piso se aprende del material, así que en una cámara real debería subir solo, pero no sé
  cuánto ni si el fantasma sigue por encima.
- **Deformación solo suave.** La pose es una sola para todo el cuadro. Un estiramiento local no se
  sigue, y no se midió.
- **Solo material que se repite.** Sin retícula, la máscara se niega a aprender.
- **Una máscara por material.** Un cambio de estampado pide aprender y aprobar otra vez. Si el material
  no calza, no juzga: sin aviso visible en la pantalla, la línea podría quedar sin inspección.
- **La aprobación es de una persona que mira.** No se probó si un operador reconoce una máscara mala
  en la vista, ni la ventana con un usuario. Solo se probaron sus botones por código.
- **Sin YOLO.** Igual que la medición del video, esta corre sin modelo.
- **Las cajas** salen ajustadas en el caso que revisé a ojo (flores · canasta), pero no medí su ajuste.

## Qué cambiaría para producto

1. **Primero, grabar tela real:** estampada, en movimiento, con fallos reales. Sin esto no recomiendo
   desarrollar más.
2. Resolver cómo tratar lo que es verdadero pero aceptable, como una costura: zonas que el operador
   excluye en la vista, o un mínimo de área o de contraste.
3. Módulo propio con `InspectionModule`, Pro como Patrón, y registro en `MODULE_REGISTRY`.
4. Flujo en Producción: capturar, revisar, aprobar y guardar en el Job con su `.npz`. Avisar en
   pantalla si la máscara no engancha.
5. Quién aprobó y cuándo ya se guarda. Falta mostrarlo en los reportes.
6. Una lectura más cuidadosa del margen de la orilla: hoy son 10 px fijos.

## Propuesta para Polaridad (solo diseño, sin código)

Hoy Polaridad decide con un solo rasgo: franja, punto o muesca. Si ese rasgo sale mal impreso o mal
iluminado, no hay con qué compararlo.

**Idea.** La misma mecánica de aprender, ver, aprobar y aplicar, pero sobre la pieza y no sobre una
retícula:

1. **Aprender.** De varias piezas buenas, ya recortadas por la detección de YOLO y alineadas a su caja,
   se saca una plantilla por promedio robusto, con su tolerancia por punto.
2. **Zonas.** La plantilla propone zonas estables: logo, marca de polaridad, serigrafía. El operador
   las ve, quita las que no sirven (un número de lote que cambia) y aprueba.
3. **Aplicar.** Cada pieza se compara contra la plantilla en sus orientaciones posibles (0° y 180°, o
   las cuatro), zona por zona. Cada zona vota y pesa según lo estable que fue al aprender.
4. **Decidir.** Gana la orientación con más voto, y solo si le saca margen a la segunda. Sin margen, el
   resultado es «no concluyente», no un OK.

**Qué aporta.** Varios puntos de referencia en vez de uno; un rasgo dañado ya no decide solo; y la zona
que no coincide dice además qué parte de la pieza está mal.

**Qué se reutiliza** de `mascara.py`: tolerancia por punto, vista y aprobación, huella, y guardado.
**Qué no:** la retícula, la pose por cuadro y el promedio a escala de celda. Ahí la pose la da la caja
de YOLO, y una caja imprecisa es el riesgo a medir primero.

**Qué medir antes de construirlo:** acierto contra el método actual, sobre piezas reales giradas y
sin girar, con el rasgo principal tapado a propósito. Es el caso que hoy falla y el que justifica el
cambio. También hace falta decidir qué hacer con piezas casi simétricas, donde ninguna zona desempata.

## Archivos

| Archivo | Qué es |
|---|---|
| `eyve/modules/mascara.py` | Núcleo |
| `tests/test_mascara.py` | Pruebas |
| `mascaras/medir.py`, `tabla.py`, `rollos.py` | Medición |
| `mascaras/medir_linea_base.py` | Guion del video, solo con otra ruta |
| `mascaras/mutar.py` | Mutaciones |
| `mascaras/revisar_mascara.py` | Ventana de aprobación |
| `mascaras/resultados/` | Cifras, vistas e intentos anteriores |
