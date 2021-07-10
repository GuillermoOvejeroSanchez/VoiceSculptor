# Socket Communication

Envio por Sockets (mas rapido, sin complejidad) Al ser todo en local no hace falta tener una conexion.
Puede ser que fuese un cliente y un servidor que fuesen en ordenadores distintos (pero en misma red local). EJ: Cliente, alumno dando discurso, Profesor, Analizando discurso

## Cliente

Cliente obtiene con un stream de PyAudio los datos en un buffer circular. Con Parselmoouth y el script de Sillabe Nuclei obtiene la informacion sobre:

```json
{
    "sound": "",
    "nsyll": 1,
    "npause": 0,
    "dur(s)": 2.9257142857142857,
    "phonationtime(s)": 2.9257142857142857,
    "speechrate(nsyll / dur)": 0.341796875,
    "articulation rate(nsyll / phonationtime)": 1.6447368421052633,
    "ASD(speakingtime / nsyll)": 0.608
}
````

Y graficas sobre:

- pitch
- intensidad
- velocidad
- pausas
- picos en pitch (TODO)

### TODO

- [ ] Algoritmo detector de pitch
- [ ] Media movil de las graficas

## Servidor

### TODO

- [ ] Mostrar informacion en tiempo real (PyQT o Web Dashboard)
- [ ] Imprimir informacion sobre pausas y numero de silabas
