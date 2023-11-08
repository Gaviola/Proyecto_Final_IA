# Predicción de resultados de partidas de LOL

## Integrantes
- Luciano Masuelli
- Facundo Gaviola

## Descripción
League of Legends (LoL) es un popular videojuego de estrategia en tiempo real que ha alcanzado una inmensa popularidad 
desde su lanzamiento en 2009. En este juego, dos equipos de cinco jugadores se enfrentan en un campo de batalla virtual, 
cada uno controlando un campeón con habilidades únicas. LoL ha dejado una marca indeleble en el mundo de los esports, 
destacándose por su enorme base de jugadores, la celebración del Campeonato Mundial anual con premios millonarios, y el 
establecimiento de ligas profesionales en diversas regiones. Además, la transmisión en plataformas como Twitch ha 
atraído a una audiencia global, contribuyendo a la profesionalización y popularidad de los deportes electrónicos.

## Objetivos
Predecir correctamente en base a los datos de una partida cuál de los dos equipos va a obtener la victoria. También se
busca determinar que factores son los que mas afectan al resultado de una partida y cuanto podria llegar a durar la 
partida dada una serie de factores iniciales.

## Alcance
Lograr predecir el resultado de una partida mediante un modelo de machine learning adecuado. Para esto se implementará 
un algoritmo de Random Forest o alguna implementacion de boosting. Se utilizará un dataset con datos de partidas 
clasificatorias de la temporada 2020 de League of Legends en la region de Corea. 

## Limitaciones
- Falta de datos, ya que hay factores que influyen en el juego que no son planteados en los datos.
- El constante cambio que tiene el juego a lo largo del tiempo dificulta que un modelo pueda mantenerse en el tiempo 
funcionando de manera correcta.

## Métricas
Para la evaluacion del desempeño de los algoritmos se utilizaran las siguientes metricas:
* Cantidad de partidas predichas correctamente / Cantidad total de partidas

## Justificación
La predicción del resultado de una partida implica considerar una amplia gama de factores y variables que resultan 
difíciles de gestionar mediante un software estocástico. Por esta razón, un modelo capaz de comprender la interacción 
entre los datos presentes en una partida y su influencia en las predicciones se presenta como una opción más conveniente 
en comparación con un programa tradicional.

## Listado de actividades a realizar
1. Recopilación de datos
2. Limpieza y puesta a punto del dataset
3. Estudio de los atributos más descriptivos
4. Investigación de modelos aplicables al problema
5. Evaluación de distintos modelos de machine learning
6. Comparar resultados de los distintos modelos frente a un algoritmo aleatorio
7. Obtención de métricas para evaluar resultados
8. Análisis de los resultados
9. Optimización del código

## Dataset
https://www.kaggle.com/datasets/gyejr95/league-of-legendslol-ranked-games-2020-ver1/data?select=match_winner_data_version1.csv
