# test/ — bac à sable d'exécution

**Toute** exécution de test/éval se fait ici, jamais sur le repo. L'éval écrit ses
artefacts dans `test/<label>/<tâche>/` (ex. `test/oracle/csv_summary/`). Persistant,
donc inspectable après coup ; nettoyé au début de chaque run.

Le contenu est **gitignoré** (voir `.gitignore`) : les artefacts produits par un
modèle n'apparaissent jamais comme des changements du repo.

Invariant dur : `agent/eval/harness.py` refuse via `_assert_sandboxed()` tout chemin
hors de ce dossier. Un solver (petit modèle + outils `write_file`/`apply_patch`) ne
peut donc pas écrire dans le repo c3po — il ne reçoit qu'un workdir sous `test/`.

Pour de futurs tests pytest : les diriger ici aussi, ex. `pytest --basetemp=test/pytest`.
