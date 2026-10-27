# Data

Datasets handed out in the course. The notebooks load them through
`comphy.DATA_DIR`, so they work from any folder.

| File | Contents | Used in |
|---|---|---|
| `lj10clusters.txt` | 38 relaxed 10-atom Lennard-Jones clusters, one per row (20 coordinates) | Exercises 18–21 |
| `pos_e1_e2_e3_1000.txt` | 1000 clusters of 12 atoms in 2D: 24 coordinates followed by three energy labels per row | Exercise 54 |
| `labeled_clusters_1000.txt` | 1000 clusters of 12 atoms: 24 coordinates, an energy and a class label (0–9) per row | Exercise 57 |
| `clusters_with_atomic_labels.npz` | `positions` (1000, 12, 2) and a boolean label for every atom, `labels` (1000, 12) | Exercise 58 |
| `exercise_62/*.npz` | 44 six-atom clusters (`positions`, `energies`) for three energy models: nearest/next-nearest neighbour, tight binding and modified tight binding | Exercise 62 |
| `exercise_63/pyroxene.npz` | Pyroxene structures: `positions`, atomic numbers `z_vals` and `energies` | Exercise 63 |
