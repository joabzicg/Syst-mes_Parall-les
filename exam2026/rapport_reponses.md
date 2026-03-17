# Réponses - Examen Machine OS202 2026

## Identification

- Nom: Joab da Silva Bezerra
- Groupe: 1216
- Date: 17/03/2026
- Machine / environnement: Linux, Ryzen 5 4600H

## Vérification du préambule

### Vérification de numba

Le module `numba` fonctionne correctement dans l'environnement de travail.

La compilation d'une fonction simple avec `njit(parallel=True)` a été vérifiée, ainsi que l'exécution d'une boucle parallèle avec `prange`.

Le nombre de threads observé par défaut est de 12, ce qui correspond au nombre de cœurs logiques disponibles sur la machine.

Le comportement de la variable `NUMBA_NUM_THREADS` a également été validé. Avec `NUMBA_NUM_THREADS=2`, numba utilise bien 2 threads d'exécution.

Ces vérifications montrent que le préambule relatif à numba est satisfait.

### Vérification de la visualisation SDL2 / OpenGL

La visualisation d'un nuage de points avec SDL2 et OpenGL fonctionne également.

L'initialisation de la fenêtre SDL2, la création du contexte OpenGL et l'affichage via le visualiseur original ont été vérifiés avec succès.

Pour permettre l'exécution du code original sans modification des fichiers de la preuve, les dépendances nécessaires ont été installées dans l'environnement Python et les bibliothèques natives manquantes ont été chargées par l'environnement d'exécution.

Ces vérifications montrent que le préambule relatif à SDL2 et OpenGL est satisfait.

### Scripts conservés pour les étapes

| Étape | Script / fichier | Remarques |
|---|---|---|
| Code initial | `nbodies_grid_numba.py` | Version de départ fournie par l'énoncé |
| Version numba parallel | `nbodies_grid_numba_parallel.py` | Boucle externe de calcul des accélérations parallélisée avec `prange` |
| Version MPI affichage/calcul | `nbodies_grid_numba_mpi_display.py` | Rang 0 pour l'affichage, rang 1 pour le calcul |
| Version MPI calcul distribué | `nbodies_grid_numba_mpi_distributed_local.py` | Corps propriétaires locaux + corps fantômes échangés entre voisins |

## Description rapide des données et des paramètres de test

### Jeux de données utilisés

Les essais s'appuient sur les deux jeux de données fournis dans le répertoire `data`, à savoir `data/galaxy_1000` et `data/galaxy_5000`.

Le fichier `data/galaxy_1000` contient 1001 corps et le fichier `data/galaxy_5000` contient 5001 corps. Cela correspond à 1000 et 5000 étoiles auxquelles s'ajoute le trou noir central massif.

Le jeu `galaxy_1000` est utilisé pour les premiers essais, la validation fonctionnelle et les mesures rapides. Le jeu `galaxy_5000` est utilisé pour les mesures plus représentatives, car le coût de calcul y est plus élevé et l'effet de la parallélisation y est plus visible.

### Paramètres utilisés

Le pas de temps retenu pour les essais est `dt = 0.001`, qui correspond à la valeur par défaut du programme.

La grille cartésienne retenue pour les essais initiaux est `(Ni, Nj, Nk) = (20, 20, 1)`, qui correspond également au paramétrage par défaut du script fourni.

Pour les mesures de performance, il est prévu d'effectuer plusieurs exécutions dans les mêmes conditions expérimentales et de retenir une valeur représentative afin de limiter l'influence des fluctuations de temps d'exécution.

Les commandes exactes seront précisées dans les sections de mesure correspondantes. À titre de référence, l'exécution de base du programme est la suivante :

```bash
python3 nbodies_grid_numba.py data/galaxy_1000 0.001 20 20 1
python3 nbodies_grid_numba.py data/galaxy_5000 0.001 20 20 1
```

## Question préliminaire

### Pourquoi n'est-il pas interessant de prendre une valeur pour `Nk` autre que 1 ?

La galaxie simulée présente essentiellement une structure aplatie dans le plan Oxy. L'épaisseur suivant Oz reste faible devant l'extension radiale dans le plan, de sorte qu'un découpage supplémentaire selon `Nk` n'apporte que peu d'information utile pour l'organisation spatiale du calcul.

Prendre `Nk > 1` conduirait donc principalement à créer des cellules supplémentaires très peu remplies, voire vides. Cela augmenterait le coût de gestion de la grille, le nombre de cellules à parcourir, ainsi que le coût de mise à jour des masses totales et des centres de masse, sans améliorer significativement la qualité de l'approximation ni la localité des interactions.

Dans ce contexte, le choix `Nk = 1` est le plus pertinent, car il reste cohérent avec la géométrie du problème tout en limitant le surcoût algorithmique.

## Mesure du temps initial

### Méthodologie

La mesure a été réalisée sans modifier le code fourni. Le script original `nbodies_grid_numba.py` a été exécuté tel quel, et les valeurs ont été relevées à partir des lignes affichées par le visualiseur dans la boucle principale sous la forme `Render time` et `Update time`.

La première itération a été écartée de l'analyse, car elle inclut le coût de compilation initiale par numba ainsi que l'initialisation du contexte graphique. Les valeurs retenues correspondent ensuite à un régime stable observé sur plusieurs itérations consécutives.

Les essais ont été effectués sur la machine décrite dans l'identification, avec `dt = 0.001` et une grille `(20, 20, 1)`, en utilisant les deux jeux de données fournis.

### Résultats

| Jeu de données | Paramètres | Temps affichage | Temps calcul | Commentaire |
|---|---|---:|---:|---|
| `data/galaxy_1000` | `dt = 0.001`, grille `(20, 20, 1)` | ≈ 3 ms | ≈ 70 ms | Valeurs stables après la compilation initiale |
| `data/galaxy_5000` | `dt = 0.001`, grille `(20, 20, 1)` | ≈ 5 ms | ≈ 775 ms | Le coût de calcul augmente fortement avec le nombre de corps |

### Quelle partie est la plus intéressante à paralléliser ?

La partie la plus intéressante à paralléliser est nettement le calcul de mise à jour des trajectoires.

Dans les deux cas testés, le temps d'affichage reste faible devant le temps de calcul. Pour `galaxy_1000`, l'affichage représente seulement quelques millisecondes, alors que la mise à jour des positions et des vitesses demande environ 70 ms. Pour `galaxy_5000`, l'écart devient encore plus marqué, avec un affichage proche de 5 ms contre environ 775 ms pour le calcul.

Le principal goulot d'étranglement est donc la phase de calcul gravitationnel et de mise à jour du système, tandis que l'affichage n'occupe qu'une fraction réduite du temps total par itération. C'est donc cette partie qu'il faut viser en priorité pour obtenir une accélération significative.

## Parallélisation en numba du code séquentiel

### Modifications apportées

Une copie du script initial a été conservée dans `nbodies_grid_numba_parallel.py`, conformément à la consigne de l'énoncé.

La modification principale consiste à paralléliser la fonction `compute_acceleration`, qui représente le cœur du coût de calcul. Le décorateur a été remplacé par `@njit(parallel=True)` et la boucle externe sur les corps a été transformée de `range(n_bodies)` en `prange(n_bodies)`.

Ce choix est naturel, car chaque itération de cette boucle calcule l'accélération d'une étoile donnée et écrit uniquement dans `a[ibody, :]`. Les itérations sont donc indépendantes entre elles. En revanche, la phase de mise à jour de la grille n'a pas été parallélisée à ce stade, car elle comporte des écritures concurrentes dans des structures partagées, ce qui demanderait une restructuration plus importante.

### Méthodologie de mesure

Les mesures ont été réalisées sur `data/galaxy_5000`, avec `dt = 0.001` et une grille `(20, 20, 1)`.

Pour isoler l'effet de la parallélisation numba sur le calcul, les temps ont été relevés sans passer par l'affichage graphique. Le script parallèle a été importé et la méthode `update_positions` a été chronométrée directement. Une première itération a été exécutée pour la compilation initiale, puis cinq itérations successives ont été mesurées pour chaque valeur de `NUMBA_NUM_THREADS`.

Les valeurs testées sont `1`, `2`, `4`, `8` et `12` threads.

Ces temps servent uniquement à comparer entre elles les versions mesurées dans le même mode, c'est-à-dire sans affichage. Ils ne doivent donc pas être comparés quantitativement aux temps relevés auparavant dans la boucle graphique complète.

Un contrôle complémentaire a également été réalisé avec exactement la même méthode sur la version initiale `nbodies_grid_numba.py`, afin de vérifier que la comparaison à `1` thread restait cohérente.

### Résultats

| Threads numba | Temps | Accélération | Efficacité | Remarques |
|---:|---:|---:|---:|---|
| 1 | 191.26 ms | 1.00 | 1.00 | Référence séquentielle de la version parallèle |
| 2 | 108.97 ms | 1.76 | 0.88 | Gain net mais variabilité plus marquée |
| 4 | 67.13 ms | 2.85 | 0.71 | Accélération claire |
| 8 | 40.04 ms | 4.78 | 0.60 | Bon compromis |
| 12 | 37.56 ms | 5.09 | 0.42 | Gain supplémentaire plus limité |

### Analyse

La parallélisation numba apporte un gain net sur cette partie du code. Le temps moyen passe d'environ 191 ms avec un thread à environ 38 ms avec 12 threads, soit une accélération un peu supérieure à 5.

Le comportement est correct jusqu'à 4 threads, puis l'accélération continue d'augmenter avec une efficacité qui diminue progressivement. Ce comportement est cohérent avec l'apparition de surcoûts de parallélisation et d'effets de bande passante mémoire.

Le contrôle effectué avec la même méthodologie sur la version initiale donne environ `796.81 ms` à `1` thread, contre environ `191.26 ms` pour la version parallèle à `1` thread. La différence ne vient donc pas d'un changement de protocole de mesure. Un test numérique sur un pas de temps montre en outre une identité exacte des positions et vitesses obtenues entre les deux versions.

La baisse par rapport au temps observé dans la boucle graphique initiale reste donc compatible avec deux effets distincts : d'une part l'absence d'affichage dans ce protocole, et d'autre part le fait que `parallel=True` modifie fortement la compilation du noyau de calcul, même lorsque numba n'utilise qu'un seul thread.

Cette étape montre néanmoins que le calcul des accélérations se prête bien à une parallélisation partagée, et qu'il constitue toujours la cible prioritaire pour la suite du travail.

## Séparation de l'affichage et du calcul

### Principe retenu

Le programme est lancé avec 2 processus MPI.

Le rang 0 gère l'affichage SDL2/OpenGL. Le rang 1 effectue le calcul des trajectoires. À chaque itération, le rang 0 envoie `dt`, le rang 1 calcule un pas de temps puis renvoie les nouvelles positions.

Les échanges portent sur l'état initial pour l'affichage, puis sur le tableau des positions à chaque itération.

### Scripts / architecture retenus

Le fichier principal est `nbodies_grid_numba_mpi_display.py`.

Il réutilise le noyau de calcul de `nbodies_grid_numba_parallel.py`. L'implémentation est volontairement simple, avec une synchronisation à chaque image.

### Méthodologie de mesure

Les mesures ont été réalisées avec `data/galaxy_5000`, `dt = 0.001` et une grille `(20, 20, 1)`, en lançant le programme avec 2 processus MPI.

La commande utilisée est de la forme suivante :

```bash
NUMBA_NUM_THREADS=4 mpirun -np 2 python3 nbodies_grid_numba_mpi_display.py data/galaxy_5000 0.001 20 20 1
```

Les valeurs testées pour `NUMBA_NUM_THREADS` sont `1`, `2`, `4`, `8` et `12`. La première itération a été écartée. Le temps retenu est le temps moyen total par itération observé sur le rang 0.

### Résultats

L'accélération par rapport à la version initiale est calculée en prenant comme référence le temps total initial d'environ 780 ms par itération, soit environ 5 ms d'affichage et 775 ms de calcul pour `data/galaxy_5000`.

| Threads numba | Temps | Accélération vs version initiale complète | Remarques |
|---:|---:|---:|---|
| 1 | 190.69 ms | 4.09 | Le calcul domine encore largement |
| 2 | 112.33 ms | 6.94 | Gain net par rapport à la version initiale |
| 4 | 103.32 ms | 7.55 | Les communications commencent à peser |
| 8 | 104.35 ms | 7.47 | Le gain se stabilise presque |
| 12 | 102.07 ms | 7.64 | La séparation n'exploite plus les threads supplémentaires |

### Comparaison avec la version precedente

#### Constat

Cette version est nettement plus rapide que la version initiale, mais elle reste moins performante que la version numba seule quand le nombre de threads augmente.

Avec 12 threads, on obtient environ 102 ms par itération, contre environ 780 ms pour la version initiale.

La comparaison avec la version numba seule doit être interprétée qualitativement, car la version numba seule a été mesurée sans affichage, alors que cette version MPI inclut le coût total d'une itération avec synchronisation et communication.

#### Pourquoi ?

Cette architecture ajoute un surcoût fixe à chaque image : communication MPI, synchronisation entre les deux processus et transfert des positions.

Quand le calcul devient plus rapide grâce à numba, ce surcoût devient plus visible et limite l'accélération.

## Parallélisation du calcul avec MPI

### Stratégie générale

Le calcul distribué final a été implémenté dans `nbodies_grid_numba_mpi_distributed_local.py`.

La grille est découpée selon `Ox` en tranches de cellules. Chaque processus détermine une zone locale et une zone fantôme de deux couches de cellules de part et d'autre, ce qui correspond au rayon de voisinage utilisé pour les interactions exactes.

Après l'initialisation, chaque processus conserve seulement ses corps propriétaires. Les corps qui changent de sous-domaine sont migrés vers leur nouveau propriétaire, et les corps proches des frontières sont échangés avec les voisins pour constituer les cellules fantômes.

Les masses totales et centres de masse des cellules restent en revanche reconstruits globalement par réduction collective, afin de conserver une implémentation simple pour la partie lointaine du calcul.

### Étapes de l'algorithme parallèle

1. Migrer les corps vers le processus propriétaire de leur cellule courante.
2. Calculer localement les contributions de masse et de centre de masse des cellules possédées.
3. Reconstruire les masses et centres de masse globaux des cellules.
4. Échanger avec les voisins les corps proches des frontières pour former les cellules fantômes.
5. Calculer et mettre à jour localement les corps possédés avec un schéma de Verlet.

### Communications MPI

Dans cette version, les communications sont de deux types.

Les échanges de voisinage servent à transmettre les corps fantômes entre processus adjacents. Les migrations de corps propriétaires sont réalisées lors du changement de sous-domaine. Enfin, des `Allreduce` sont utilisés pour reconstruire les masses et centres de masse globaux des cellules.

Cette version est plus locale que la précédente : les corps propriétaires sont migrés vers leur rang, les corps proches des frontières sont échangés entre voisins pour former les cellules fantômes, mais les masses et centres de masse des cellules restent reconstruits globalement par réductions collectives pour traiter la partie lointaine du calcul.

### Méthodologie de mesure

Les mesures ont été réalisées en mode calcul seul, sans affichage, avec `data/galaxy_5000`, `dt = 0.001` et une grille `(20, 20, 1)`.

La commande utilisée est de la forme suivante :

```bash
NUMBA_NUM_THREADS=2 mpirun -np 4 python3 nbodies_grid_numba_mpi_distributed_local.py data/galaxy_5000 0.001 20 20 1 3 1
```

Les couples testés sont `(1,1)`, `(1,2)`, `(2,1)`, `(2,2)`, `(4,1)` et `(4,2)`, où le premier nombre désigne le nombre de processus MPI et le second le nombre de threads numba. L'accélération est calculée par rapport au cas `(1,1)`.

Comme pour la section précédente, ces comparaisons ne sont quantitatives qu'entre versions mesurées dans le même mode, ici sans affichage.

### Résultats

| Processus MPI | Threads numba | Temps | Accélération | Efficacité | Remarques |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 418.20 ms | 1.00 | 1.00 | Référence |
| 1 | 2 | 251.17 ms | 1.66 | 0.83 | Gain correct en mémoire partagée |
| 2 | 1 | 219.01 ms | 1.91 | 0.96 | Bon partage sur 2 processus |
| 2 | 2 | 153.17 ms | 2.73 | 0.68 | Meilleur compromis observé |
| 4 | 1 | 232.74 ms | 1.80 | 0.45 | Déséquilibre de charge visible |
| 4 | 2 | 131.98 ms | 3.17 | 0.40 | Communications plus lourdes |

### Analyse des performances

Le meilleur résultat mesuré est obtenu avec `4` processus MPI et `2` threads numba, avec un temps moyen d'environ `131.98 ms` par itération. Le cas `2` processus MPI et `2` threads numba reste toutefois un compromis plus équilibré en termes d'efficacité.

Le passage de `1` à `2` processus fonctionne correctement, puis les gains deviennent plus irréguliers. Le coût des échanges de voisinage, des migrations et des réductions globales augmente, tandis que le découpage régulier des cellules ne répartit plus correctement les corps entre les processus.

### En observant que la densité d'étoiles diminue avec l'éloignement du trou noir, quel problème de performance peut handicaper l'accélération ?

Le principal problème est le déséquilibre de charge. Les cellules proches du centre contiennent beaucoup plus d'étoiles que les cellules périphériques, donc certains processus ont beaucoup plus de travail que d'autres.

Cela apparaît déjà dans les mesures avec `4` processus, pour lesquelles le nombre de corps locaux varie fortement selon les rangs.

### Proposition d'une distribution intelligente des cellules

Une meilleure stratégie consisterait à attribuer moins de cellules centrales par processus et davantage de cellules périphériques, afin d'équilibrer le nombre de corps réellement traités par chaque rang au lieu d'équilibrer seulement le nombre de cellules.

On pourrait donc utiliser une distribution non uniforme de la grille, plus fine ou plus légère selon la densité locale d'étoiles.

### Quel problème de performance peut alors apparaître ?

Une telle distribution peut augmenter la complexité des communications et le coût de synchronisation, car les frontières entre sous-domaines deviennent moins régulières.

Elle peut aussi introduire un nouveau déséquilibre dynamique si la répartition des étoiles évolue au cours du temps, ce qui imposerait éventuellement un rééquilibrage périodique.

## Pour aller plus loin - Barnes-Hut

### Comment distribuer les différentes boîtes et sous-boîtes parmi les processus ?

Une stratégie naturelle consiste à distribuer les sous-arbres du quadtree par niveau. Les boîtes proches de la racine peuvent être partagées ou répliquées sur plusieurs processus, tandis que les sous-boîtes plus profondes sont réparties entre les rangs.

Avec un nombre de processus puissance de quatre, on peut par exemple associer les quatre grands quadrants du domaine à quatre groupes de processus, puis poursuivre récursivement cette répartition dans les niveaux inférieurs de l'arbre.

### Proposition d'une parallélisation MPI de l'accélération des étoiles avec cette structure

On peut construire l'arbre global, puis distribuer les sous-arbres terminaux entre les processus. Chaque processus calcule alors l'accélération des étoiles qui lui sont confiées en parcourant l'arbre de Barnes-Hut.

Les nœuds supérieurs de l'arbre, qui résument de grandes régions par leur masse totale et leur centre de masse, peuvent être répliqués sur tous les processus. En revanche, les sous-arbres fins peuvent rester distribués. Chaque processus calcule donc localement les interactions détaillées dans ses sous-boîtes et utilise les nœuds résumés pour les régions lointaines.

Après le calcul des accélérations, chaque processus met à jour les étoiles dont il a la charge, puis une synchronisation permet de reconstruire l'état global avant l'itération suivante.

### Hypothèses et limites de la proposition

Cette proposition suppose que le nombre de processus reste faible devant le nombre d'étoiles, et que le coût de construction ou de diffusion de l'arbre reste inférieur au gain apporté par la réduction du coût de calcul.

La principale limite est la difficulté d'équilibrer correctement les sous-arbres, car certaines régions de l'espace peuvent contenir beaucoup plus d'étoiles que d'autres. Il faut également contrôler le coût mémoire si plusieurs processus partagent des parties communes de l'arbre.

## Conclusion

### Résumé des gains obtenus

La première amélioration importante a été obtenue par la parallélisation numba du calcul des accélérations, avec une accélération proche de 6 pour 12 threads sur le noyau de calcul.

La séparation MPI entre affichage et calcul a ensuite permis de réduire fortement le temps total par rapport à la version initiale, mais avec un plafonnement dû au coût des communications et de la synchronisation.

Enfin, la version MPI distribuée locale a montré qu'un gain supplémentaire est possible avec une répartition réelle des corps entre processus, même si le coût des communications et le déséquilibre de charge limitent rapidement l'efficacité.

Il faut toutefois souligner que cette dernière version reste seulement partiellement locale : elle est locale pour les corps propriétaires et les corps fantômes de frontière, mais elle conserve une reconstruction globale des agrégats de cellules pour traiter la partie lointaine du calcul.

### Difficultés rencontrées

Les principales difficultés ont concerné la mise en place de l'environnement d'exécution graphique, puis le choix d'une stratégie de parallélisation qui reste simple, correcte et mesurable dans le temps imparti.

Du point de vue algorithmique, la difficulté principale est de concilier parallélisation, cohérence des données globales et coût des synchronisations MPI.

### Pistes d'amélioration

Une première amélioration consisterait à réduire encore la partie collective du calcul distribué, afin de se rapprocher d'une décomposition locale complète.

Une deuxième piste serait de mettre en place un équilibrage de charge plus intelligent, fondé sur le nombre réel d'étoiles par sous-domaine plutôt que sur un découpage uniforme.

Enfin, une approche fondée sur Barnes-Hut permettrait probablement d'obtenir de meilleures performances pour de très grands systèmes, grâce à une complexité plus favorable.
