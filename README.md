# TFG 2020 : Analizador de Voz

## Iniciar Proyecto

Windows: instalar PyAudio desde el fichero .whl

Instalar el resto de paquetes necesarios, desde requirements.txt o creando un pipenv y ejecutando:

```bash
pipenv install
```

## Report

Usa un servidor de Flask donde muestra la informacion sobre pausas, velocidad, intensidad, pitch

## Live Audio

Usa el callback de pyaudio para ir guardando los chunks del audio en un buffer circular que tiene 5 Segundos (es variable) para almacenar datos.
Los manda a traves de un Socket
Live plotter (basico, se puede mejorar con las funciones de animacion de matplotlib) recive la informacion de un Socket y plotea la informacion
Que mas informacion podemos sacar??
Para probar he instalado el Voicemeteer y cogiendo el audio de las locuciones y presentaciones que grabamos, para no tener que hablar yo si no poder analizar eso grabado como si fuese en tiempo real
