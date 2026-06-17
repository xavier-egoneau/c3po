# Jeu d'éval c3po

Mesure si la couche **agentic** rapproche un **petit modèle local** d'un **gros
modèle frontier** sur des tâches variées. C'est le garde-fou anti-overfit : on
définit la mesure *avant* de (re)coder l'agentic, pour ne pas optimiser autour
d'un seul prompt comme l'a fait l'ancien scaffold.

## Principe

- une **tâche** = un `prompt` + un **checker exécutable et déterministe** ;
- un **solver** (ce qu'on teste) reçoit `(prompt, workdir)` et écrit ses
  artefacts dans `workdir` ;
- le checker **exécute** les artefacts (import, subprocess, parse) et renvoie une
  liste de `(nom, passé, détail)`. **Jamais** de checklist de domaine ni de juge LLM.

Score par tâche = fraction de checks passés (0..1).

## Lancer

```bash
python -m agent.eval --list          # liste les tâches
python -m agent.eval --oracle        # solver de référence : doit être 100% partout
python -m agent.eval --only email_validator csv_summary
```

`--oracle` copie les `solution_ref/` dans le workdir : il valide **les checkers
eux-mêmes**. S'il n'est pas à 100 %, c'est un checker (pas un modèle) qui est faux.

## Bac à sable : jamais sur le repo

Toute exécution se fait sous `test/<label>/<tâche>/` (gitignoré, persistant donc
inspectable, nettoyé à chaque run). `harness.py` refuse via `_assert_sandboxed()`
tout chemin hors de `test/`. Un solver ne reçoit qu'un `workdir` sous `test/` : un
petit modèle avec des outils d'écriture **ne peut pas toucher au repo c3po**. Voir
`test/README.md`.

## Métrique : taux de réussite absolu + delta brut → agentic

Les checks sont **exécutables**, donc le score est interprétable tel quel : 70 % = 70 %
des vérifications passent. On suit deux labels :

- `small` : petit modèle **brut** (qwen via le runtime, un seul appel, sans agentic) ;
- `small_agent` : petit modèle **+ couche agentic**.

Ce qui dit si l'agentic vaut son coût = le **delta `small_agent − small`** (de combien il
fait monter le petit modèle) et l'**absolu** (à quel point on approche 100 %). Pas de
baseline gros modèle dans le harnais : on compare au gros **à la main, à côté**, au besoin.

### Lancer (déjà câblé)

```bash
c3po load <repo-gguf>                                       # re-télécharger le modèle si besoin
python -m agent.eval --model ~/.c3po/models/<x.gguf>           # baseline brute -> label "small"
python -m agent.eval --model ~/.c3po/models/<x.gguf> --agent   # couche agentic -> label "small_agent"
```

### En code

```python
from agent.eval.harness import load_tasks, run_eval, render_comparison
from agent.eval.solvers import engine_chat, make_oneshot_solver, make_agent_solver

tasks = load_tasks()
chat = engine_chat("~/.c3po/models/<x.gguf>")
reports = {
    "small":       run_eval(lambda t: make_oneshot_solver(chat), "small",       tasks),
    "small_agent": run_eval(lambda t: make_agent_solver(chat),   "small_agent", tasks),
}
print(render_comparison(reports))   # table + moyennes
```

Les deux solvers (`agent/eval/solvers.py`) :

- `make_oneshot_solver` — le **plancher** : un appel modèle, zéro outil, zéro boucle.
- `make_agent_solver` — la **couche agentic générique** : émet, puis **vérifie en exécutant**
  (compile, pytest, smoke-run) et **renvoie l'erreur concrète au modèle** pour qu'il corrige,
  en boucle bornée. Aucune logique propre à une tâche. Sa vérif est indépendante des `check.py`
  de l'éval (ne pas tricher). C'est lui qui doit battre le plancher — et le delta le prouve.

## Ajouter une tâche

Crée `tasks/<nom>/` avec :

- `task.json` : `{ "name", "prompt", "requires": [] }` ;
- `check.py` : expose `check(workdir) -> list[(nom, passé, détail)]` ;
- `solution_ref/` : artefacts corrects (pour `--oracle`) ;
- `fixture/` *(optionnel)* : état de départ copié dans le workdir avant le solver.

Vise des checks **exécutables** (lance le code, compare une sortie) plutôt que
des heuristiques textuelles.

## Tâches actuelles

| tâche | déficit du petit modèle qu'elle sonde |
|---|---|
| `email_validator` | requirements sous-spécifiés, cas limites (rigueur) |
| `fix_failing_test` | lire du code existant, correction minimale, ne pas tricher |
| `csv_summary` | sortie structurée exacte |
| `wordcount_cli` | livrable réellement exécutable (CLI) |
| `html_counter` | artefact web câblé (le domaine de l'ancien overfit, réduit à **1** tâche) |

## Limites assumées

- `html_counter` vérifie la **structure + le câblage** du JS, pas le rendu (pas de
  moteur JS embarqué). Checker **local à la tâche**, jamais un linter réutilisé.
- Les checkers **exécutent du code produit par un modèle** : à lancer localement
  uniquement.
- À étendre : une tâche **multimodale** (`requires: ["vision"]`) pour mesurer
  l'égaliseur sidecar — non implémentée tant que le runtime vision n'est pas branché.
