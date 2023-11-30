# Modelo de Prediccion de Partidas de League of Legends

---

***Luciano Massuelli***

***Facundo Gaviola***

***Facultad de Ingeniería, Universidad Nacional de Cuyo***


---
## Introducción

El objetivo de este proyecto es buscar predecir, mediante algoritmos de inteligencia artificial, el resultado de una
partida de League of Legends.

Para dar un poco de contexto para aquel que no conozca el juego, League of Legends (LoL) es un popular videojuego de 
estrategia en tiempo real que ha alcanzado una inmensa popularidad desde su lanzamiento en 2009. En este juego, dos 
equipos de cinco jugadores se enfrentan en un campo de batalla virtual, cada uno controlando un campeón con habilidades 
únicas. El juego a dejado una marca indeleble en el mundo de los esports, destacándose por su enorme base de jugadores,
la celebración del Campeonato Mundial anual con premios millonarios, y el establecimiento de ligas profesionales en
diversas regiones.

Por todo lo mencionado anteriormente, se puede considerar de una gran importancia el poder predecir el resultado de una
partida de LoL, ya que esto puede ser utilizado, ya sea para apuestas deportivas, mejorar el rendimiento de un equipo o
simplemente comprender las distintas variables que se relacionan entre si durante una partida a la hora de conseguir la
victoria.

Para esto se implementaran modelos de inteligencia artificial, tales como Random Forest o alguna implementacion de
boosting, para poder predecir el resultado de una partida. La aplicacion de estas tecnicas de inteligencia artificial
son la mejor opcion para este problema, debido a una serie de factores:
* Complejidad del juego: El juego posee muchas variables que pueden influir de gran manera al desarrollo y resultado de
una partida.
* Gran cantidad de datos: Debido a la popularidad de LoL, es una tarea facil la recopilacion de una gran cantidad de 
partidas para poder entrenar los modelos.
* Analisis de variables: Mediante la aplicacion de tecnicas de inteligencia artificial, se puede analizar cuales son las
variables que mas influyen en el resultado de una partida, y cuales son las que menos influyen. Algortimos como Random
Forest permiten analizar la importancia de cada variable a la hora de realizar una prediccion.

Para la realizacion del proyecto se utilizaran partidas de la region de Corea de la temporada 2020 de LoL. Estas 
partidas fueron obtenidas de la pagina web de Kaggle, y se pueden encontrar en el siguiente link:
https://www.kaggle.com/datasets/gyejr95/league-of-legendslol-ranked-games-2020-ver1/data?select=match_winner_data_version1.csv

Se intentaran utilizar como variables predictoras a aquellos datos que puedan ser obtenidos al comienzo de una partida 
o en su defecto, en los primeros minutos de transcurrida la misma. Esto se debe a que se busca realizar una prediccion
lo mas temprana posible en la partidad.

---

## Marco teórico

**Arboles de Decision**

Los arboles de decision representan una funcion que toma como entrada un vector de atributos y valores y devuelve como 
resultado una decision. Cada nodo interno del arbol representa una prueba sobre un atributo, cada rama representa el
resultado de una prueba y cada nodo hoja representa una clase. El camino desde la raiz hasta una hoja representa una
regla de clasificacion.

El algoritmo en pseudocodigo utilizado para la creacion de un arbol de decision es el siguiente:
* plurality-value devuelve el valor mas comun entre los ejemplos.
* A es el atributo que mejor clasifica los ejemplos.
* IMPORTANCE es una funcion que mide la importancia de un atributo.
* v_k es el valor de A que corresponde a la rama del arbol.
* e.A es el valor del atributo A en el ejemplo e.
```
function decision-tree-learning(examples, attributes, parent_examples) returns a tree
    if examples is empty then return plurality-value(parent_examples)
    else if all examples have the same classification then return the classification
    else if attributes is empty then return plurality-value(examples)
    else
        A <- argmax a in attributes IMPORTANCE(a, examples)
        tree <- a new decision tree with root test A
        for each value v_k of A do
            exs <- {e : e in examples and e.A = v_k}
            subtree <- decision-tree-learning(exs, attributes - A, examples)
            add a branch to tree with label (A = v_k) and subtree subtree
        return tree
```

**Random Forest**

Random Forest es un algoritmo de aprendizaje basado en arboles de decision. El algoritmo crea una serie de arboles de
decision, de la misma manera que lo hace un algoritmo de bagging, pero en este caso, en el momento de en que cada arbol
del modelo va a crear nuevas ramas, se selecciona un subconjunto aleatorio de m variables predictoras del total de 
variables predictoras p. En cada creacion de ramas se seleccionan m variables predictoras distintas. Esto se hace para
evitar que los arboles esten altamente correlacionados entre si y asi tener un modelo mas confiable. Generalmente se
utiliza m = sqrt(p) para la cantidad de variables seleccionadas.

**Boosting**

Boosting es un algoritmo de aprendizaje que, al igual que Random Forest, se basa en arboles de decision para realizar 
regresiones o clasificaciones. En boosting no se involucran tecnicas de boostraping, sino que se utilizan distintas
versiones modificadas de los datos de entrenamiento. El algoritmo combina multiples arboles de decision, ˆf 1,..., ˆf B
para crear un modelo de prediccion f(x) = sum(b=1 to B) f_b(x). Es importante destacar que los arboles de decision son
pequeños, es decir, tienen pocos niveles y pocos nodos. Esto mejora lentamente el rendimiento de f(x). El pseudocodigo
del algoritmo es el siguiente:
* ri es el residuo de la prediccion en el ejemplo i.
* lambda es un parametro de contraccion.
* d es la cantidad de divisiones que se realizan en cada arbol
* fb es el arbol de decision b
* f es el modelo de prediccion

```
1. Initialize f_0(x) = 0
2. For b = 1 to B:
    (a) Fit a tree ˆf b with d splits (d + 1 terminal nodes) to the training data (X, r).
    (b) Update ˆf by adding in a shrunken version of the new tree: ˆf(x) ← ˆf(x) + λ ˆfb(x).
    (c) Update the residuals,ri ← ri − λ ˆfb(xi).
3. Output the boosted model, f(x) = sum(b=1 to B) λf_b(x).
```

---


## Diseño Experimental
 
El primer paso para la realizacion del proyecto es la adecuacion del dataset para poder ser utilizado correctamente por
los algoritmos de inteligencia artificial. Para esto se realizo una limpieza de los datos, eliminando aquellos que no 
sean necesarios para las predicciones de los modelos, tales como, datos de temporada, creacion de partida, modo de juego
(solo nos enfocamos en el modo de juego classic) y estadisticas de los jugadores al finalizar la partida.

Tambien se modifico la presentacion de los datos, ya que estos se encontraban almacenados en estructuras de datos, tales
como listas o diccionarios, dificultando su uso. Para ello se crearon nuevas columnas en el dataset, donde se guardaran
los datos de dichas estructruras.

---

## Análisis y discusión de resultados


---

## Conclusiones finales


---

## Bibliografía

[1] Stuart Russell and Peter Norvig. Artificial Intelligence: A Modern Approach (3rd. ed.). Pearson, 2010.

[2] Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani.An Introduction to Statistical Learning with 
Applications in R. 

