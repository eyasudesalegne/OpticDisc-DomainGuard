# Notebooks

## Provenance note

The manuscript's final experiments were assembled from staged Colab runs. The recovered run reports identify a May 16 cross-domain notebook/run and subsequent G1020 annotation-aware phases, but the Drive connector did not expose a uniquely identifiable original `.ipynb` under the DomainGuard/OpticDisc names.

For scientific integrity, this repository does **not** upload a newly generated notebook and call it the original.

A clean reproduction notebook should orchestrate the modular code in `src/` and `scripts/`, while the exact historical notebook—if later recovered by filename or exported from Colab history—should be stored separately under an explicit archival name such as:

```text
notebooks/archive/original_colab_<date>.ipynb
```

and compared against the cleaned implementation before being cited as the computational record.
