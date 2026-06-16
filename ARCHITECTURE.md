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
- **`agent/` + `experiments/` (c3po-agent)** : couche **expérimentale**, hors API stable.
  Échafaudage agentique pour petits modèles locaux (vision sidecar pour l'instant).
  Voir [agent/roadmap.md](agent/roadmap.md).

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

- 87 tests de **logique pure** (`pytest tests/`, ~5 s), sans GPU ni GGUF.
- CI GitHub Actions sur Python 3.9 / 3.11 / 3.12, installe le package `--no-deps`
  (`llama_cpp` n'est importé que paresseusement, au chargement réel d'un modèle).
- Les chemins GPU / inférence / réseau restent validés manuellement (voir CONTEXT.md).

## Hardware de référence

- **Mac M4**, 16 Go unifiés (~12 Go utilisables), backend Metal.
- **PC RTX 4070**, 12 Go VRAM (~11 Go après réserve), backend CUDA.

## Limites connues

- **Swap de modèle sur CUDA** : historiquement, charger deux modèles successifs dans le même
  process crashait (`ggml_cuda_error`). Atténué par une libération explicite
  (`Engine.close()`) appelée avant rechargement dans `server.get_engine()`. **À valider sur
  RTX 4070** ; si le crash persiste, le fallback robuste est l'isolation par subprocess
  (un modèle = un process worker, déchargement = arrêt du process).
- **Couverture GPU/inférence/réseau** : manuelle uniquement (pas de CI dotée d'un GPU).
- **Multimodal** : non branché dans l'API stable ; spike vision dans `agent/` + `experiments/`.

## Installation & usage

Voir [README.md](README.md).
