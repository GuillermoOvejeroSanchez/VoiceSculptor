# TFG 2020 : Analizador de Voz



## Report 

Usa un servidor de Flask donde muestra la informacion sobre pausas, velocidad, intensidad, pitch

## Live Audio

Usa el callback de pyaudio para ir guardando los chunks del audio en un buffer circular que tiene 5 Segundos (es variable) para almacenar datos.
Los manda a redis a traves de stream XADD, XREAD, etc
Live plotter (basico, se puede mejorar con las funciones de animacion de matplotlib) coge el stream de redis (en vez de redis se podria guardar en un fichero o en RabbitMQ "choco install rabbitmq") y plotea la informacion
Que mas informacion podemos sacar??
Para probar he instalado el Voicemeteer y cogiendo el audio de las locuciones y presentaciones que grabamos, para no tener que hablar yo si no poder analizar eso grabado como si fuese en tiempo real