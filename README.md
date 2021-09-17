# TFG 2020 : Analizador de Voz

## Iniciar Proyecto

```bash
choco install python --version=3.8.0
sudo apt-get install python
```

Windows: install pipenv

```bash
py -m ensurepip --upgrade
```

Windows: instalar PyAudio desde el fichero .whl

```bash
pip install wheel
pip install PyAudio-0.2.11-cp38-cp38-win_amd64
```

Instalar el resto de paquetes necesarios, desde requirements.txt o creando un pipenv y ejecutando:

```bash
pip install -r requirements.txt
```

## Report

Usa un servidor de Flask donde muestra la informacion sobre pausas, velocidad, intensidad, pitch

## Live Audio

Usa el callback de pyaudio para ir guardando los chunks del audio en un buffer circular que tiene 5 Segundos (es variable) para almacenar datos.
Los manda a traves de un Socket
Live plotter (basico, se puede mejorar con las funciones de animacion de matplotlib) recive la informacion de un Socket y plotea la informacion
Que mas informacion podemos sacar??
Para probar he instalado el Voicemeteer y cogiendo el audio de las locuciones y presentaciones que grabamos, para no tener que hablar yo si no poder analizar eso grabado como si fuese en tiempo real

## Performance

```ipython
In[1] %timeit snd = parselmouth.Sound("holamesa.wav")
170 µs ± 3.22 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

In [2]: snd.duration
Out[2]: 1.6370068027210884

In [3]: %timeit snd.to_pitch()
9.26 ms ± 94.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [4]: %timeit snd.to_intensity()
2.08 ms ± 5.26 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

------------

In [1]: %timeit snd = parselmouth.Sound("locucion2.wav")
10.7 ms ± 87.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [2]: snd.duration
Out[2]: 36.288

In [3]: %timeit snd.to_pitch()
64.9 ms ± 866 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

In [4]: %timeit snd.to_intensity()
52.3 ms ± 208 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

------------

In [1]: %timeit snd = parselmouth.Sound("00017.wav")
99.7 ms ± 1.21 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

In [2]: snd.duration
Out[2]: 314.624

In [3]: %timeit snd.to_pitch()
503 ms ± 8.15 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

In [4]: %timeit snd.to_intensity()
447 ms ± 595 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)


```
