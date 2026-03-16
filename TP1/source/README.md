
# TD1

`pandoc -s --toc README.md --css=./github-pandoc.css -o README.html`

## lscpu

*lscpu donne des infos utiles sur le processeur : nb core, taille de cache :*

```
Model name:                           AMD Ryzen 5 4600H with Radeon Graphics
CPU(s):                               12
Thread(s) per core:                   2
Core(s) per socket:                   6
Socket(s):                            1
L1d cache:                            192 KiB (6 instances)
L1i cache:                            192 KiB (6 instances)
L2 cache:                             3 MiB (6 instances)
L3 cache:                             4 MiB (1 instance)
```


## Produit matrice-matrice

### Effet de la taille de la matrice

  n            | MFlops
---------------|--------
512            | 4123.43
1024           | 2981.30
2048           | 3500.30
4096           | 3127.40

*Expliquer les résultats.*

Les performances varient avec $n$ à cause des effets de **hiérarchie mémoire** (cache) et de la taille des données. Pour $n=512$, tout tient mieux dans les caches → MFlops élevés. Pour $n=1024$, on observe un creux (plus de conflits/cache misses). Pour $n=2048$ et $n=4096$, le coût est dominé par la bande passante mémoire ; les MFlops se stabilisent autour de 3–3.5 GFlops.


### Permutation des boucles

*Expliquer comment est compilé le code (ligne de make ou de gcc) : on aura besoin de savoir l'optim, les paramètres, etc. Par exemple :*

`make TestProductMatrix.exe && LOOP_ORDER=jki BLOCK_SIZE=1024 OMP_NUM_THREADS=1 ./TestProductMatrix.exe 1024`


  ordre           | time    | MFlops  | MFlops(n=2048)
------------------|---------|---------|----------------
i,j,k (origine)   | 11.9614 | 179.534 | 166.66
j,i,k             | 9.01144 | 238.306 | 203.344
i,k,j             | 29.4565 | 72.9035 | 83.0811
k,i,j             | 30.9227 | 69.4469 | 77.3369
j,k,i             | 0.785247| 2734.79 | 2516.81
k,j,i             | 1.04216 | 2060.62 | 1998.18


*Discuter les résultats.*

L’ordre **j–k–i** est très largement le meilleur car il suit le **stockage colonne‑major** et parcourt des segments contigus en mémoire pour $A$ et $C$, avec un accès régulier à $B$. Les ordres **i–k–j** et **k–i–j** sont les pires (strides défavorables), ce qui explose les cache misses. L’écart de performance (×10–40) illustre l’importance de la localité mémoire.



### OMP sur la meilleure boucle

`make TestProduct.exe && OMP_NUM_THREADS=8 ./TestProduct.exe 1024`

  OMP_NUM         | MFlops  | MFlops(n=2048) | MFlops(n=512)  | MFlops(n=4096)
------------------|---------|----------------|----------------|---------------
1                 | 2729.81 | 2103.42        | 4137.06        | 3266.14
2                 | 5244.00 | 5219.76        | 7565.50        | 5231.18
3                 | 7966.42 | 6568.33        | 11944.8        | 7580.88
4                 | 9929.83 | 8809.26        | 13477.2        | 7808.03
5                 | 12454.1 | 9948.87        | 16301.1        | 8660.36
6                 | 12060.6 | 9540.08        | 13967.7        | 8806.67
7                 | 13741.1 | 10117.6        | 15373.2        | 7299.71
8                 | 11889.5 | 11520.5        | 16624.9        | 8701.69

*Tracer les courbes de speedup (pour chaque valeur de n), discuter les résultats.*

Le speedup est correct jusqu’à 4–5 threads, puis la courbe **sature** (et parfois baisse) à cause de la **bande passante mémoire** et des overheads OpenMP. Pour $n=512$, les gains sont meilleurs car le problème tient mieux en cache. Pour $n=4096$, la saturation est plus marquée : l’algorithme est **memory‑bound**, donc l’augmentation du nombre de threads apporte un bénéfice limité.



### Produit par blocs

`make TestProduct.exe && ./TestProduct.exe 1024`

  szBlock         | MFlops  | MFlops(n=2048) | MFlops(n=512)  | MFlops(n=4096)
------------------|---------|----------------|----------------|---------------
origine (=max)    | 2921.92 | 3271.11        | 3958.22        | 3297.41
32                | 2387.57 | 2013.97        | 2979.87        | 2544.66
64                | 2669.56 | 3654.04        | 3824.81        | 3247.55
128               | 3087.80 | 3857.40        | 3709.39        | 2549.61
256               | 3411.92 | 3537.05        | 4121.79        | 2142.32
512               | 2770.80 | 2874.11        | 3958.22        | 2584.36
1024              | 2921.92 | 1854.97        | 4277.65        | 2481.05

*Discuter les résultats.*

Le blocage améliore la **réutilisation en cache** quand la taille de bloc est adaptée. On observe un optimum autour de **128–256** pour $n=1024$ et $n=2048$. Des blocs trop petits augmentent l’overhead de boucles, et des blocs trop grands se comportent comme le cas “origine”, perdant l’avantage du cache.



### Bloc + OMP


  szBlock      | OMP_NUM | MFlops  | MFlops(n=2048) | MFlops(n=512)  | MFlops(n=4096)|
---------------|---------|---------|----------------|----------------|---------------|
1024           |  1      | 3094.00 | 2834.36        | 4177.43        | 2863.65       |
1024           |  8      | 14615.7 | 12422.5        | 14482.5        | 10687.6       |
512            |  1      | 2735.71 | 2632.95        | 4026.10        | 2655.69       |
512            |  8      | 13641.2 | 14173.2        | 16784.6        | 10848.3       |

*Discuter les résultats.*

La combinaison **bloc + OMP** donne les meilleurs résultats : la localité mémoire est améliorée et le parallélisme est bien exploité. À 8 threads, les performances montent jusqu’à ~14–17 GFlops selon $n$. Le gain dépend du compromis entre **taille de bloc** (localité) et **nombre de threads** (parallelisme).


### Comparaison avec BLAS, Eigen et numpy

*Comparer les performances avec un calcul similaire utilisant les bibliothèques d'algèbre linéaire BLAS, Eigen et/ou numpy.*

Mesures BLAS (DGEMM) :

| n | MFlops |
|---:|---:|
| 512 | 3407.58 |
| 1024 | 3292.33 |
| 2048 | 3293.70 |
| 4096 | 2883.82 |

**Comparaison :** sur cette machine, BLAS est compétitif mais pas systématiquement supérieur à la meilleure version maison (bloc + OMP), qui atteint ~14–17 GFlops à 8 threads. Ici, la BLAS liée est probablement une implémentation générique (non optimisée type OpenBLAS/MKL), d’où un débit plus faible.


# Tips

```
	env
	OMP_NUM_THREADS=4 ./produitMatriceMatrice.exe
```

```
    $ for i in $(seq 1 4); do elap=$(OMP_NUM_THREADS=$i ./TestProductOmp.exe|grep "Temps CPU"|cut -d " " -f 7); echo -e "$i\t$elap"; done > timers.out
```
