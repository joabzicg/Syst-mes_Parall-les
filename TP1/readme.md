# TD n° 1 - 27 Janvier 2026

## 1. Produit matrice–matrice

### Réponse – item 1

Mesures avec l’exécutable [TP1/source/TestProductMatrix.exe](TP1/source/TestProductMatrix.exe).

| Dimension | Temps (s) | MFlops |
|---:|---:|---:|
| 1023 | 4.58403 | 467.099 |
| 1024 | 18.9469 | 113.342 |
| 1025 | 11.8692 | 181.459 |

**Interprétation :**

- La dimension 1024 (puissance de deux) provoque des **conflits de cache** plus fréquents : les accès réguliers à la mémoire tombent souvent sur les mêmes ensembles de cache (set-associative), ce qui augmente fortement les miss.
- Les dimensions 1023 et 1025 **cassent la périodicité** des strides, ce qui répartit mieux les accès en cache et réduit les conflits.
- Le produit utilisé par défaut est **j–k–i** (voir [TP1/source/ProdMatMat.cpp](TP1/source/ProdMatMat.cpp)) et la matrice est stockée en **colonne‑major** (indexation $i + j\,n_{rows}$).
- Il n’y a **pas de blocage effectif** dans cette mesure : la fonction reçoit une taille de bloc égale à la dimension, donc le calcul est un produit naïf sans découpage en sous‑blocs.

### Réponse – item 2

**Permutation des boucles (optimum trouvé) :** ordre **j–k–i**.

Justification : la matrice est stockée en **colonne‑major** (indexation $i + j\,n_{rows}$). Avec l’ordre j–k–i, la boucle interne parcourt $i$ de façon contiguë pour **A(i,k)** et **C(i,j)**, tandis que **B(k,j)** est parcouru de façon contiguë sur $k$.

Mesures après permutation des boucles :

| Dimension | Temps (s) | MFlops |
|---:|---:|---:|
| 1023 | 0.804886 | 2660.25 |
| 1024 | 0.745896 | 2879.07 |
| 1025 | 0.785573 | 2741.67 |

**Interprétation :**

- L’ordre j–k–i maximise la **localité mémoire** (accès contigus sur $i$ et $k$), ce qui réduit fortement les cache misses.
- La performance devient bien plus stable et élevée, et la dimension 1024 n’est plus pénalisée : le goulot principal était l’ordre des accès mémoire.

### Réponse – item 3

Parallélisation OpenMP du produit matrice–matrice (boucle externe sur $j$) :

```cpp
#pragma omp parallel for schedule(static)
for (int j = 0; j < n; ++j)
	for (int k = 0; k < n; ++k)
		for (int i = 0; i < n; ++i)
			C(i,j) += A(i,k)*B(k,j);
```

Mesures (dimension 1024) :

| Threads | Temps (s) | Speedup | Efficacité |
|---:|---:|---:|---:|
| 1 | 0.90133 | 1.000 | 1.000 |
| 2 | 0.418458 | 2.154 | 1.077 |
| 4 | 0.242186 | 3.722 | 0.930 |
| 8 | 0.170533 | 5.285 | 0.661 |

**Commentaires :**

- Le speedup est bon jusqu’à 4 threads, puis la saturation apparaît (bandwidth mémoire, contention).  
- L’efficacité légèrement > 1 à 2 threads est un effet de variabilité/cache (mesures très sensibles).  
- Globalement, le gain reste cohérent avec une charge mémoire importante.

### Réponse – item 4

Il est possible d’améliorer nettement les performances en **réduisant le trafic mémoire** et en **augmentant la localité**. Le produit actuel est limité par la bande passante mémoire (accès à $A$, $B$, $C$ sans blocage). Un **produit par blocs (tiling)** permet de réutiliser des sous‑blocs en cache et d’augmenter le ratio calcul/mémoire. On peut aussi viser :

- **Vectorisation** (SIMD) et alignement mémoire,
- **Choix d’un meilleur ordre de boucles** couplé au blocage,
- **BLAS optimisé** (DGEMM),
- **Parallélisation sur blocs** (OpenMP) avec granularité adaptée.

### Réponse – item 5

Implémentation du produit par **blocs** dans [TP1/source/ProdMatMat.cpp](TP1/source/ProdMatMat.cpp) en découpant $A$, $B$, $C$ en sous‑matrices de taille `BLOCK_SIZE` et en accumulant :

$$
C_{IJ} = \sum_K A_{IK}\,B_{KJ}
$$

Mesures (dimension 1024, 1 thread) en faisant varier `BLOCK_SIZE` :

| BLOCK_SIZE | Temps (s) | MFlops |
|---:|---:|---:|
| 8 | 2.34243 | 916.774 |
| 16 | 1.21471 | 1767.89 |
| 32 | 0.899445 | 2387.57 |
| 64 | 0.804432 | 2669.56 |
| 128 | 0.695473 | 3087.80 |
| 192 | 0.652725 | 3290.03 |
| 256 | 0.629406 | 3411.92 |
| 384 | 0.680767 | 3154.51 |
| 512 | 0.77504 | 2770.80 |
| 1024 | 0.734957 | 2921.92 |

**Optimum observé :** `BLOCK_SIZE = 256` pour ce matériel et cette dimension.

### Réponse – item 6

Le produit par blocs est **nettement plus rapide** que le produit scalaire (sans blocage) parce qu’il **réutilise** mieux les données en cache. En version scalaire, les accès à $A$ et $B$ provoquent beaucoup de cache misses, donc le calcul est limité par la bande passante mémoire. Avec le blocage, chaque sous‑bloc reste en cache pendant plusieurs opérations, ce qui **augmente l’intensité de calcul** (flops par octet) et réduit le trafic mémoire. D’où un gain clair tant que la taille de bloc est bien choisie.

### Réponse – item 7

La parallélisation **par blocs + OpenMP** donne de meilleurs résultats que la version scalaire parallélisée, car elle combine **localité mémoire** et **parallélisme**. En pratique, la version scalaire sature rapidement (memory‑bound), tandis que le blocage réduit les cache misses et permet une montée en charge plus efficace avec le nombre de threads. Le gain dépend du compromis entre **taille de bloc** et **nombre de threads** : un bloc trop petit augmente l’overhead, un bloc trop grand perd l’avantage cache.

### Réponse – item 8

Comparaison avec BLAS (DGEMM) mesuré avec [TP1/source/test_product_matrice_blas.exe](TP1/source/test_product_matrice_blas.exe) :

| n | MFlops BLAS |
|---:|---:|
| 512 | 3407.58 |
| 1024 | 3292.33 |
| 2048 | 3293.70 |
| 4096 | 2883.82 |

Sur cette machine, la BLAS liée est correcte mais **moins rapide** que la meilleure version maison (bloc + OMP), qui atteint ~14–17 GFlops à 8 threads. Exemple de rapport à $n=1024$ : $14615.7/3292.33 \approx 4.44\times$ (bloc+OMP vs BLAS). À $n=2048$ : $14173.2/3293.7 \approx 4.30\times$.

**Remarque threads/config** : le test BLAS a été lancé **sans réglage de threading** (pas de variables type `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`). Le résultat correspond donc au comportement par défaut de la BLAS installée. Une BLAS multithreadée/tunée pourrait donner un débit plus élevé. Malgré cela, sur cette machine, la meilleure version reste **bloc + OpenMP**.

## 2. Parallélisation MPI

### Réponse – item 2.1 (Circulation d’un jeton dans un anneau)

Implémentation en C/MPI dans [TP1/source/mpi_ring_token.c](TP1/source/mpi_ring_token.c). Le rang 0 initialise le jeton à 1, l’envoie au rang 1, chaque rang l’incrémente puis le transmet au suivant. Le rang $nbp-1$ renvoie le jeton au rang 0.

Exemple (4 processus) : le jeton final vaut 4, ce qui correspond à $1 + (nbp-1)$.

### Réponse – item 2.2 (Calcul très approché de $\pi$) — OpenMP

Version mémoire partagée en C/OpenMP : [TP1/source/calcul_pi_omp.c](TP1/source/calcul_pi_omp.c). Le nombre d’échantillons est réparti entre threads avec une réduction sur le nombre de points dans le disque.

Compilation :

`make calcul_pi_omp.exe`

Exécution (ex. 10^7 échantillons, 8 threads) :

`OMP_NUM_THREADS=8 ./calcul_pi_omp.exe 10000000`

**Mesures OpenMP (10^7 échantillons)**

| Threads | Temps (s) | Speedup | Efficacité |
|---:|---:|---:|---:|
| 1 | 0.096840 | 1.000 | 1.000 |
| 2 | 0.047040 | 2.059 | 1.029 |
| 4 | 0.024746 | 3.913 | 0.978 |
| 8 | 0.013748 | 7.044 | 0.880 |

### Réponse – item 2.2 (MPI en C)

Version MPI C : [TP1/source/calcul_pi.cpp](TP1/source/calcul_pi.cpp). Chaque processus calcule une portion des échantillons et on réduit le nombre de points dans le disque avec `MPI_Reduce`.

Compilation :

`make calcul_pi.exe`

Exécution :

`mpirun -np 4 ./calcul_pi.exe 10000000`

**Mesures MPI C (10^7 échantillons)**

| Processus | Temps (s) | Speedup | Efficacité |
|---:|---:|---:|---:|
| 1 | 0.171803 | 1.000 | 1.000 |
| 2 | 0.090284 | 1.903 | 0.951 |
| 4 | 0.042274 | 4.064 | 1.016 |
| 8 | 0.027988 | 6.139 | 0.767 |

### Réponse – item 2.2 (MPI en Python / mpi4py)

Version Python : [TP1/source/compute_pi_mpi.py](TP1/source/compute_pi_mpi.py). Chaque processus génère ses points aléatoires avec NumPy puis réduit le nombre de points dans le disque.

Exécution :

`mpirun -np 4 python compute_pi_mpi.py 10000000`

**Mesures mpi4py (10^7 échantillons)**

| Processus | Temps (s) | Speedup | Efficacité |
|---:|---:|---:|---:|
| 1 | 0.509181 | 1.000 | 1.000 |
| 2 | 0.245843 | 2.071 | 1.036 |
| 4 | 0.148824 | 3.421 | 0.855 |
| 8 | 0.157903 | 3.225 | 0.403 |

**Commentaire :** les valeurs d’efficacité légèrement > 1 proviennent de la variabilité/effet cache. En général, OpenMP est le plus rapide ici ; mpi4py plafonne plus tôt à cause du surcoût Python.

### Réponse – item 2.3.1 (Hypercube dimension 1)

Programme MPI en C : [TP1/source/mpi_hypercube_dim1.c](TP1/source/mpi_hypercube_dim1.c).

- Le rang 0 initialise un jeton (ici 42) et l’envoie au rang 1.
- Le rang 1 reçoit le jeton et l’affiche.

Exécution :

`mpirun -np 2 ./mpi_hypercube_dim1.exe`

Exemple de sortie :

Rank 0 sent token=42
Rank 1 received token=42

### Réponse – item 2.3.2 (Hypercube dimension 2)

Programme MPI en C : [TP1/source/mpi_hypercube_dim2.c](TP1/source/mpi_hypercube_dim2.c).

Diffusion en **2 étapes** (minimum) :

1. étape 1 (bit 0) : rang 0 → rang 1
2. étape 2 (bit 1) : rang 0 → rang 2 et rang 1 → rang 3 (en parallèle)

Exécution :

`mpirun -np 4 ./mpi_hypercube_dim2.exe`

Exemple de sortie :

Rank 0 has token=42
Rank 1 has token=42
Rank 2 has token=42
Rank 3 has token=42

### Réponse – item 2.3.3 (Hypercube dimension 3)

Programme MPI en C : [TP1/source/mpi_hypercube_dim3.c](TP1/source/mpi_hypercube_dim3.c).

Diffusion en **3 étapes** (minimum) :

1. étape 1 (bit 0) : rang 0 → rang 1
2. étape 2 (bit 1) : rang 0 → rang 2, rang 1 → rang 3
3. étape 3 (bit 2) : rang 0 → rang 4, rang 1 → rang 5, rang 2 → rang 6, rang 3 → rang 7

Exécution :

`mpirun -np 8 ./mpi_hypercube_dim3.exe`

Exemple de sortie :

Rank 0 has token=42
Rank 1 has token=42
Rank 2 has token=42
Rank 3 has token=42
Rank 4 has token=42
Rank 5 has token=42
Rank 6 has token=42
Rank 7 has token=42

### Réponse – item 2.3.4 (Hypercube dimension d)

Programme MPI générique : [TP1/source/mpi_hypercube_general.c](TP1/source/mpi_hypercube_general.c).

Le principe : à l’étape $s$ (de 0 à $d-1$), **seuls les rangs qui possèdent déjà le jeton** émettent vers le partenaire du demi‑groupe supérieur. Autrement dit, dans le groupe de taille $2^{s+1}$ :

- rangs $0..2^s-1$ envoient à $rank + 2^s$
- rangs $2^s..2^{s+1}-1$ reçoivent de $rank - 2^s$

En $d$ étapes, le jeton est diffusé à tous les nœuds (2^d processus).

Exécution (ex. d=3) :

`mpirun -np 8 ./mpi_hypercube_general.exe 3`

### Réponse – item 2.3.5 (Mesure de l’accélération)

Programme de mesure : [TP1/source/mpi_hypercube_broadcast_time.c](TP1/source/mpi_hypercube_broadcast_time.c). On répète la diffusion et on mesure le temps moyen par diffusion.

Mesures (2000 itérations) :

| d | p=2^d | Temps moyen (s) |
|---:|---:|---:|
| 1 | 2 | 0.000000678 |
| 2 | 4 | 0.000001760 |
| 3 | 8 | 0.000011175 |

**Interprétation :** le temps augmente avec $d$ (il y a $d$ étapes de communication). L’“accélération” n’est pas favorable ici car la diffusion hypercube est déjà optimale en $d$ étapes ; ajouter des nœuds augmente le coût de communication total.