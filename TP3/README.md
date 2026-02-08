# TD n°3 - parallélisation du Bucket Sort

*Ce TD peut être réalisé au choix, en C++ ou en Python*

Implémenter l'algorithme "bucket sort" tel que décrit sur les deux dernières planches du cours n°3 :

- le process 0 génère un tableau de nombres arbitraires,
- il les dispatch aux autres process,
- tous les process participent au tri en parallèle,
- le tableau trié est rassemblé sur le process 0.

## Réponse

### 1) Rappel de l’algorithme Bucket Sort

- Initialiser un tableau de *buckets* vides.
- **Scatter** : placer chaque valeur dans le bucket correspondant à son intervalle.
- Trier localement chaque bucket.
- **Gather** : concaténer les buckets dans l’ordre pour produire le tableau trié.

### 2) Parallélisation 

1. **Génération et distribution**
	- Le process 0 génère $N$ valeurs (par exemple uniformes sur $[0,1[$).
	- Distribution équitable des données : `MPI_Scatter` (hypothèse $N$ multiple de `nbp`).

2. **Tri local**
	- Chaque process trie ses $N/nbp$ valeurs localement (tri standard).

3. **Choix des intervalles de buckets (splitters)**
	- Chaque process prélève `nbp+1` valeurs à intervalles réguliers dans sa liste triée.
	- Rassemblement des échantillons : `MPI_Gather` ou `MPI_Allgather`.
	- Le process 0 trie le tableau d’échantillons et extrait `nbp+1` valeurs régulières (indices $i\,(nbp+1)$).
	- Diffusion des bornes d’intervalles : `MPI_Bcast`.

4. **Distribution dans les buckets**
	- Chaque process partitionne ses valeurs selon les intervalles reçus.
	- Échanges de segments vers le process cible : `MPI_Alltoallv` (ou `MPI_Send/MPI_Recv`).

5. **Tri final et rassemblement**
	- Chaque process effectue une fusion (*merge*) des segments reçus (chaque segment est déjà trié).
	- Rassemblement sur le process 0 : `MPI_Gatherv` (ou `MPI_Gather` si tailles égales).

### 3) Complexité

- Tri local : $O(\frac{N}{nbp}\log\frac{N}{nbp})$
- Calcul des intervalles : tri de l’échantillon $O(nbp\log nbp)$
- Coût communication (schéma du cours) :
  - collecte des échantillons : $nbp\,(t_s + (nbp+1)t_d)$
  - redistribution des données : $(nbp-1)t_s + \frac{N^2}{nbp}t_d$

## Rapport 

Approche « 1 process = 1 bucket » conforme au cours. Les intervalles des buckets sont estimés par échantillonnage régulier des données triées localement, puis tri global des échantillons et extraction de *splitters* aux indices $i\,(nbp+1)$ avant diffusion. L’hypothèse $N$ multiple de `nbp` permet d’utiliser `MPI_Scatter`. La redistribution se fait via `MPI_Alltoallv` pour gérer des tailles variables de buckets. Après réception, les segments sont déjà triés et sont fusionnés par *merge* (fusion itérative) pour obtenir le bucket local final, puis `MPI_Gatherv` rassemble le tableau trié sur le process 0.

