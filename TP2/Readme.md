# TD n° 2 - 27 Janvier 2026

##  1. Parallélisation ensemble de Mandelbrot

L'ensensemble de Mandebrot est un ensemble fractal inventé par Benoit Mandelbrot permettant d'étudier la convergence ou la rapidité de divergence dans le plan complexe de la suite récursive suivante :

$$
\left\{
\begin{array}{l}
    c\,\,\textrm{valeurs\,\,complexe\,\,donnée}\\
    z_{0} = 0 \\
    z_{n+1} = z_{n}^{2} + c
\end{array}
\right.
$$
dépendant du paramètre $c$.

Il est facile de montrer que si il existe un $N$ tel que $\mid z_{N} \mid > 2$, alors la suite $z_{n}$ diverge. Cette propriété est très utile pour arrêter le calcul de la suite puisqu'on aura détecter que la suite a divergé. La rapidité de divergence est le plus petit $N$ trouvé pour la suite tel que $\mid z_{N} \mid > 2$.

On fixe un nombre d'itérations maximal $N_{\textrm{max}}$. Si jusqu'à cette itération, aucune valeur de $z_{N}$ ne dépasse en module 2, on considère que la suite converge.

L'ensemble de Mandelbrot sur le plan complexe est l'ensemble des valeurs de $c$ pour lesquels la suite converge.

Pour l'affichage de cette suite, on calcule une image de $W\times H$ pixels telle qu'à chaque pixel $(p_{i},p_{j})$, de l'espace image, on associe une valeur complexe  $c = x_{min} + p_{i}.\frac{x_{\textrm{max}}-x_{\textrm{min}}}{W} + i.\left(y_{\textrm{min}} + p_{j}.\frac{y_{\textrm{max}}-y_{\textrm{min}}}{H}\right)$. Pour chacune des valeurs $c$ associées à chaque pixel, on teste si la suite converge ou diverge.

- Si la suite converge, on affiche le pixel correspondant en noir
- Si la suite diverge, on affiche le pixel avec une couleur correspondant à la rapidité de divergence.

1. À partir du code séquentiel `mandelbrot.py`, faire une partition équitable par bloc suivant les lignes de l'image pour distribuer le calcul sur `nbp` processus  puis rassembler l'image sur le processus zéro pour la sauvegarder. Calculer le temps d'exécution pour différents nombre de tâches et calculer le speedup. Comment interpréter les résultats obtenus ?

    ### Réponse – item 1

    En prenant le temps **Temps total (calcul + gather)** comme $T_p$, le temps de référence séquentiel est :
    
    $$T_1 = 2.359119\ \text{s}$$

    Les métriques de performance sont définies comme suit :

    **Accélération (speedup)**
    $$
    S(p) = \frac{T_1}{T_p}
    $$

    **Efficacité**
    $$
    E(p) = \frac{S(p)}{p} = \frac{T_1}{p\,T_p}
    $$

    **Résultats expérimentaux :**

    | p | $T_p$ (s)  | $S(p)$ | $E(p)$ |
    |---:|----------:|-------:|-------:|
    | 1 | 2.359119 | 1.000 | 1.000 |
    | 2 | 1.214600 | 1.942 | 0.971 |
    | 4 | 0.734141 | 3.213 | 0.803 |
    | 8 | 0.505847 | 4.664 | 0.583 |

    **Interprétation :**
    
    Les résultats montrent une accélération sous-linéaire : avec 8 processus, on obtient $S(8) = 4.664$ au lieu de 8 (idéal). L'efficacité décroît de 100% à 58.3% lorsqu'on passe de 1 à 8 processus. Cette dégradation s'explique principalement par :
    - Le **déséquilibre de charge** : la répartition par blocs contigus fait que certains processus traitent des zones plus coûteuses (près de la frontière de l'ensemble de Mandelbrot)
    - Les **coûts de communication** lors du gather final
    - Le **surcoût de parallélisation** (création des processus, synchronisation)


2. Réfléchissez à une meilleur répartition statique des lignes au vu de l'ensemble obtenu sur notre exemple et mettez la en œuvre. Calculer le temps d'exécution pour différents nombre de tâches et calculer le speedup et comparez avec l'ancienne répartition. Quel problème pourrait se poser avec une telle stratégie ?

    ### Réponse – item 2

    Avec une répartition par blocs contigus (Q1), certains processus peuvent tomber sur des zones de l’image beaucoup plus coûteuses (près de la frontière de l’ensemble), donc le temps total est imposé par le processus le plus lent. Pour limiter ça sans passer au dynamique, j’ai utilisé une répartition statique plus régulière : les lignes sont distribuées de manière cyclique (round-robin), c’est-à-dire que le rang `rank` calcule les lignes `rank, rank+p, rank+2p, ...`. Ainsi, chaque processus récupère un mélange de lignes “faciles” et “dures”, ce qui réduit le risque de déséquilibre.

    Les temps suivants correspondent à $T_p =$ **Temps total (calcul + gather)**, avec $T_1 = 2.639128\ \text{s}$. On calcule ensuite :

    $$
    S(p) = \frac{T_1}{T_p}
    \qquad
    E(p) = \frac{S(p)}{p}
    $$

    | p | $T_p$ (s) | $S(p)$ | $E(p)$ |
    |---:|----------:|-------:|-------:|
    | 1 | 2.639128 | 1.000 | 1.000 |
    | 2 | 1.341885 | 1.967 | 0.983 |
    | 4 | 0.658673 | 4.007 | 1.002 |
    | 8 | 0.499758 | 5.281 | 0.660 |

    En comparaison avec la répartition par blocs (Q1), on obtient des performances très proches à $p=4$, un peu moins bonnes à $p=2$, et clairement meilleures à $p=8$ (le temps total passe de 0.505847 s à 0.499758 s). Ce comportement est cohérent : quand on augmente le nombre de processus, le déséquilibre de charge de la version “blocs” devient plus pénalisant, et la distribution cyclique finit par mieux répartir les lignes coûteuses.

    Le point faible de cette stratégie, c’est qu’elle peut dégrader la localité mémoire (lignes non contiguës par processus), et qu’elle complique la reconstruction de l’image : il faut rassembler puis remettre les lignes dans le bon ordre au rang 0, ce qui ajoute un surcoût. Enfin, le gain dépend aussi de la zone visualisée (zoom/fenêtre) : si la partie “difficile” se déplace, l’intérêt de la répartition choisie peut varier.



3. Mettre en œuvre une stratégie maître-esclave pour distribuer les différentes lignes de l'image à calculer. Calculer le speedup avec cette approche et comparez  avec les solutions différentes. Qu'en concluez-vous ?

    ### Réponse – item 3

    Stratégie maître-esclave dynamique : le rang 0 distribue des blocs de lignes (par paquets) aux workers, récupère les résultats et réassigne du travail jusqu’à ce que toutes les lignes soient calculées. Cette approche vise à réduire le déséquilibre de charge, au prix d’un surcoût de communication et d’un maître qui ne calcule pas (dans cette version).

    Les temps suivants correspondent à $T_p =$ **Temps total (calcul + gather)**, avec $T_1 = 2.341504\ \text{s}$. On calcule ensuite :

    $$
    S(p) = \frac{T_1}{T_p}
    \qquad
    E(p) = \frac{S(p)}{p}
    $$

    | p | $T_p$ (s) | $S(p)$ | $E(p)$ |
    |---:|----------:|-------:|-------:|
    | 1 | 2.341504 | 1.000 | 1.000 |
    | 2 | 2.426539 | 0.965 | 0.482 |
    | 4 | 0.869175 | 2.694 | 0.673 |
    | 8 | 0.583822 | 4.011 | 0.501 |

    **Comparaison et conclusion :**

    - Par rapport aux versions par blocs et statique cyclique, cette stratégie est **moins performante** pour ces mesures, surtout à $p=2$, car le maître ne calcule pas et la granularité engendre beaucoup d’échanges.
    - Le speedup inférieur à 1 à $p=2$ est cohérent : la surcharge MPI (envois/recevions) + l’inactivité du maître peuvent coûter plus que le gain de parallélisation sur seulement 2 processus.
    - L’avantage principal est la **réduction du déséquilibre** quand la charge est très irrégulière, mais ce gain est ici masqué par le coût de communication/synchronisation.
    - Une amélioration classique serait d’augmenter la taille des blocs et/ou faire calculer le maître, afin de réduire l’overhead.

## 2. Produit matrice-vecteur

On considère le produit d'une matrice carrée $A$ de dimension $N$ par un vecteur $u$ de même dimension dans $\mathbb{R}$. La matrice est constituée des cœfficients définis par $A_{ij} = (i+j) \mod N  + 1$. 

Par soucis de simplification, on supposera $N$ divisible par le nombre de tâches `nbp` exécutées.

### a - Produit parallèle matrice-vecteur par colonne

Afin de paralléliser le produit matrice–vecteur, on décide dans un premier temps de partitionner la matrice par un découpage par bloc de colonnes. Chaque tâche contiendra $N_{\textrm{loc}}$ colonnes de la matrice. 

- Calculer en fonction du nombre de tâches la valeur de Nloc
- Paralléliser le code séquentiel `matvec.py` en veillant à ce que chaque tâche n’assemble que la partie de la matrice utile à sa somme partielle du produit matrice-vecteur. On s’assurera que toutes les tâches à la fin du programme contiennent le vecteur résultat complet.
- Calculer le speed-up obtenu avec une telle approche

    ### Réponse – item 2a

    **Découpage par colonnes :**
    
    $$
    N_{loc} = \frac{N}{p}
    $$

    Chaque processus gère un bloc de colonnes $[j_{0},\, j_{0}+N_{loc}-1]$ de $A$ et le sous-vecteur $u_{loc}$ correspondant. Il calcule une contribution partielle $v^{(loc)} = A_{loc} \cdot u_{loc}$ (dimension $N$) puis on somme toutes les contributions par **Allreduce** pour obtenir $v$ complet sur tous les rangs.

    Implémentation : [TP2/matvec_mpi_col.py](TP2/matvec_mpi_col.py)

    **Mesures (N=6000, médiane sur 5 runs, temps = calcul + allreduce)**

    | p | $T_p$ (s) | $S(p)$ | $E(p)$ |
    |---:|----------:|-------:|-------:|
    | 1 | 0.012616 | 1.000 | 1.000 |
    | 2 | 0.013246 | 0.952 | 0.476 |
    | 4 | 0.094066 | 0.134 | 0.034 |
    | 8 | 0.118419 | 0.107 | 0.013 |

    **Remarque :** pour ce découpage par colonnes, la communication **Allreduce** domine rapidement. Les mesures restent sensibles à l’overhead MPI (WSL/oversubscribe), d’où une baisse de performance quand $p$ augmente.

### b - Produit parallèle matrice-vecteur par ligne

Afin de paralléliser le produit matrice–vecteur, on décide dans un deuxième temps de partitionner la matrice par un découpage par bloc de lignes. Chaque tâche contiendra $N_{\textrm{loc}}$ lignes de la matrice.

- Calculer en fonction du nombre de tâches la valeur de Nloc
- paralléliser le code séquentiel `matvec.py` en veillant à ce que chaque tâche n’assemble que la partie de la matrice utile à son produit matrice-vecteur partiel. On s’assurera que toutes les tâches à la fin du programme contiennent le vecteur résultat complet.
- Calculer le speed-up obtenu avec une telle approche

    ### Réponse – item 2b

    **Découpage par lignes :**

    $$
    N_{loc} = \frac{N}{p}
    $$

    Chaque processus calcule un bloc de lignes de $A$ (taille $N_{loc} \times N$) et obtient un sous‑vecteur $v_{loc}$ de taille $N_{loc}$. Ensuite, on reconstruit $v$ complet par **Allgather** pour que tous les rangs possèdent le vecteur résultat.

    Implémentation : [TP2/matvec_mpi_row.py](TP2/matvec_mpi_row.py)

    **Mesures (N=2000, temps = calcul + allgather)**

    | p | $T_p$ (s) | $S(p)$ | $E(p)$ |
    |---:|----------:|-------:|-------:|
    | 1 | 0.001835 | 1.000 | 1.000 |
    | 2 | 0.001565 | 1.173 | 0.586 |
    | 4 | 0.096352 | 0.019 | 0.005 |
    | 8 | 0.119855 | 0.015 | 0.002 |

    **Remarque :** pour $N$ relativement petit et en oversubscribe, le coût de communication (Allgather) et la contention dominent rapidement, ce qui explique la forte dégradation dès $p\ge 4$.

## 3. Entraînement pour l'examen écrit

Alice a parallélisé en partie un code sur machine à mémoire distribuée. Pour un jeu de données spécifiques, elle remarque que la partie qu’elle exécute en parallèle représente en temps de traitement 90% du temps d’exécution du programme en séquentiel.

En utilisant la loi d’Amdhal, pouvez-vous prédire l’accélération maximale que pourra obtenir Alice avec son code (en considérant n ≫ 1) ?

À votre avis, pour ce jeu de donné spécifique, quel nombre de nœuds de calcul semble-t-il raisonnable de prendre pour ne pas trop gaspiller de ressources CPU ?

En effectuant son cacul sur son calculateur, Alice s’aperçoit qu’elle obtient une accélération maximale de quatre en augmentant le nombre de nœuds de calcul pour son jeu spécifique de données.

En doublant la quantité de donnée à traiter, et en supposant la complexité de l’algorithme parallèle linéaire, quelle accélération maximale peut espérer Alice en utilisant la loi de Gustafson ?

### Réponse – Exercice 3

On a une fraction parallélisable $f = 0.9$ (donc fraction séquentielle $\alpha = 0.1$).

**Amdahl ($n \to \infty$)**
$$
S_{max} = \frac{1}{1- f} = \frac{1}{0.1} = 10
$$

Or, si l’accélération **maximale observée** est 4 quand on augmente $p$, cela signifie qu’en pratique l’effet séquentiel (ou l’overhead) est plus grand. Un modèle équivalent serait :
$$
S_{max} = \frac{1}{\alpha} = 4 \Rightarrow \alpha = 0.25 \; (f = 0.75)
$$

**Nombre de nœuds raisonnable**

Théoriquement, le plafond est 10 (si $\alpha=0.1$). Mais comme l’observation montre une saturation à 4, un choix raisonnable est plutôt de l’ordre de 4 nœuds pour ce jeu de données (au‑delà, le gain marginal est négligeable).

**Gustafson avec données ×2**

Si les données doublent et la partie parallèle est linéaire, en prenant l’**effet séquentiel effectif** issu de l’observation ($\alpha=0.25$), la fraction séquentielle devient :
$$
\alpha' = \frac{0.25}{0.25 + 2\times 0.75} = \frac{1}{7} \approx 0.1429
$$
Avec $p=4$ nœuds (puisque le speedup observé est 4), la loi de Gustafson donne :
$$
S_G = p - \alpha'(p-1) = 4 - 0.1429\times 3 \approx 3.57
$$

