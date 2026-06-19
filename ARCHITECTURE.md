# Architecture c3po

Vue de l'état **actuel** du système (pas l'historique des décisions — celui-ci vit dans
[CONTEXT.md](CONTEXT.md), tenu comme journal chronologique).

## Vue d'ensemble

c3po est un orchestrateur local autour de [llama.cpp](https://github.com/ggerganov/llama.cpp)
(via `llama-cpp-python`). Il détecte le hardware (Apple Metal / Nvidia CUDA / CPU), calcule
des paramètres d'inférence adaptés, et expose les leviers plutôt que de les cacher.

```
[utilisateur] → CLI c3po / API HTTP
                    ↓
              llm_runtime (c3po-core)
   détection hardware · calcul params · chargement GGUF · génération · stats/doctor
                    ↓
             llama-cpp-python
                    ↓
              Metal | CUDA | CPU
                    ↓
               fichier GGUF
```

Deux couches coexistent dans le dépôt :

- **`llm_runtime/` (c3po-core)** : la lib stable. Inférence locale, routing hardware,
  API, stats, doctor.
- **`agent/` (c3po-agent)** : couche **expérimentale**, hors API stable. L'« égaliseur » :
  rapprocher un petit modèle local d'un modèle frontier par scaffolding agentic **générique**
  (pas d'opinion par tâche → anti-overfit), **mesuré** par un harnais d'éval. Voir la section
  dédiée plus bas et [agent/eval/README.md](agent/eval/README.md).

## Modules `llm_runtime/`

| Module | Rôle |
|---|---|
| [hardware.py](llm_runtime/hardware.py) | Détecte Metal / CUDA / CPU → `HardwareProfile` (VRAM totale − réserve fixe, déterministe). |
| [params.py](llm_runtime/params.py) | `compute_params()` → `InferenceParams` (n_gpu_layers, threads, n_ctx, flash, kv_type…). KV cache calculé exactement (gère la GQA). `apply_overrides()` applique les leviers explicites. |
| [gguf.py](llm_runtime/gguf.py) | Lit l'en-tête GGUF en stdlib (`struct`), sans charger le modèle : couches, contexte d'entraînement, embedding, têtes, têtes KV. |
| [engine.py](llm_runtime/engine.py) | `Engine` : charge un GGUF, expose `.generate()`, `.chat()`, comptage de tokens. Applique params + overrides, force flash-attn si KV quantifié, enregistre l'instance. |
| [models.py](llm_runtime/models.py) | Indexe les GGUF (`~/.c3po/models` + `./models` + Ollama). `find_model()`, `best_model()`, `FIT_MARGIN`. Exclut les `mmproj`. |
| [instances.py](llm_runtime/instances.py) | Suivi des process c3po vivants (locks dans tmpdir). Politique **mono-instance** : `terminate_other_instances()` avant chaque chargement. |
| [server.py](llm_runtime/server.py) | API HTTP compatible OpenAI (FastAPI). **Mono-modèle, mono-user**, générations sérialisées. Swap de modèle à la requête (façon Ollama). |
| [batch.py](llm_runtime/batch.py) | Traitement batch **mono-instance** (file séquentielle derrière une seule instance modèle). |
| [stats.py](llm_runtime/stats.py) | Métadonnées GGUF + benchmark réel (chargement, TTFT, tok/s) sur profils `general` et `code`. |
| [download.py](llm_runtime/download.py) | `load` / `search` Hugging Face en **HTTP brut (urllib stdlib, zéro dépendance, sans token requis)**. Quant auto selon la VRAM, reprise, vérif SHA256. |
| [doctor.py](llm_runtime/doctor.py) | Diagnostic non destructif : Python, hardware, `llama-cpp-python` + offload GPU, en-tête GGUF, `mmproj`, commandes de rebuild prescriptives. |
| [cli.py](llm_runtime/cli.py) | Point d'entrée `c3po` : list / info / doctor / search / load / run / stats / serve / batch. Imports paresseux. Compaction de l'historique en chat interactif. |

## Couche `agent/` (expérimentale, hors API stable)

But : **réduire l'écart** entre un petit modèle local et un modèle frontier, par scaffolding
agentic **générique** (jamais d'opinion par tâche). Tout est **mesuré** par un jeu d'éval à
checkers exécutables, pas supposé.

| Module | Rôle |
|---|---|
| [agent/eval/](agent/eval/) | Harnais d'éval : tâches = prompt + **checker exécutable** + oracle déterministe. Mesure l'effet de la couche agentic. Voir [agent/eval/README.md](agent/eval/README.md). |
| [agent/eval/solvers.py](agent/eval/solvers.py) | Les solvers + briques : `make_context_solver` (**sélection de contexte** — le levier prouvé), `make_oneshot_solver` (baseline), `make_agent_solver` (boucle émettre→vérifier→corriger). Sélection = nom + pertinence-contenu + clôture d'imports, budget dérivé du `n_ctx`. Vérifs déterministes (syntaxe, pytest, navigateur Playwright) + signal vision optionnel. `make_subprocess_chat` (un modèle/process). |
| [agent/run.py](agent/run.py) | Lance un solver sur une tâche réelle (`python -m agent.run`) ; mode `--chat` (itération humaine, recours fiable pour le visuel/sémantique). |
| [agent/vision/gemma4.py](agent/vision/gemma4.py) | Sidecar vision (gemma-4 + `mmproj`) : image → observation structurée. `observe_image_subprocess()` pour un usage répété sûr sur CUDA. |
| [agent/eval/chat_worker.py](agent/eval/chat_worker.py) | Worker subprocess (un modèle/process) : orchestrer petit solver + gros juge sans swap in-process (qui crashe sur CUDA). |

**Findings mesurés** : le levier dominant est la **sélection/optimisation de contexte** (sans
elle, un petit modèle **crashe** — dépasse sa fenêtre — sur un vrai repo) ; la boucle de vérif
aide sur les erreurs **déterministes** (crash/test/erreur navigateur) ; le signal sémantique
vision fonctionne avec un **juge capable** (drive-then-look + reasoning ~14B) mais reste
probabiliste ; le sémantique fin reste à l'humain via `--chat`.

## Principes directeurs

- **Leviers exposés, pas cachés.** `--ctx`, `--n-gpu-layers`, `--threads`, `--flash-attn`,
  `--kv-type`, `--speculative`, `--repeat-penalty` sur run/stats/serve/batch. L'auto est
  calculé, l'override s'applique par-dessus (None = auto).
- **Mono-instance.** Un seul gros modèle vivant à la fois : moins de copies en VRAM/RAM,
  mémoire prévisible, moins d'OOM. Les commandes qui chargent un modèle terminent d'abord
  les autres instances c3po.
- **Pas de magie silencieuse.** Pas de dégradation de qualité cachée (jamais de KV Q4 en
  auto). Le serveur OpenAI ne compacte pas l'historique du client.
- **Zéro dépendance lourde hors runtime.** download/gguf/instances sont en stdlib pure.

## Tests & CI

- **Dette connue** : la suite `tests/` de logique pure du runtime (anciennement ~87 tests) a été
  **supprimée** lors d'un ménage et **reste à réécrire**. La CI GitHub Actions est à réaligner en
  conséquence.
- En attendant, la seule vérif automatique déterministe est l'**oracle du harnais d'éval** :
  `python -m agent.eval --oracle` — les solutions de référence doivent passer tous les checkers
  (~14 tâches), sans GPU.
- Chemins GPU / inférence / réseau / multimodal : validés **manuellement** (voir CONTEXT.md).

## Hardware de référence

- **Mac M4**, 16 Go unifiés (~12 Go utilisables), backend Metal.
- **PC RTX 4070**, 12 Go VRAM (~11 Go après réserve), backend CUDA.

## Limites connues

- **Swap de modèle sur CUDA** : historiquement, charger deux modèles successifs dans le même
  process crashait (`ggml_cuda_error`, cf. CONTEXT.md Phase 11). `Engine.close()` (libération
  explicite avant rechargement dans `server.get_engine()`) rend la libération déterministe.
  **Validé sur RTX 4070** : swap Engine-level et triple swap via `get_engine()` OK. Le crash
  d'origine n'a pas pu être reproduit dans la config actuelle (llama-cpp-python 0.3.29,
  driver 580) — possiblement déjà résolu en amont. Fallback robuste si le crash réapparaît :
  isolation par subprocess (un modèle = un process worker, déchargement = arrêt du process).
- **Couverture GPU/inférence/réseau** : manuelle uniquement (pas de CI dotée d'un GPU).
- **Multimodal** : non branché dans l'**API stable** (le runtime sert le texte). Sidecar vision
  **fonctionnel** dans `agent/vision/` (gemma-4 + `mmproj`), hors API stable. Note CUDA :
  recharger un modèle multimodal in-process crashe → le sidecar tourne en **subprocess**
  (`observe_image_subprocess`).

## Installation & usage

Voir [README.md](README.md).
