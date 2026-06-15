# llm-runtime — Contexte projet

## Objectif
Construire une lib Python (`llm_runtime`) qui orchestre llama.cpp avec routing hardware automatique : Apple Silicon (Metal) ou Nvidia (CUDA). Similaire à Ollama mais avec les leviers exposés, pas cachés.

## Stack
- **Runtime** : llama-cpp-python (wrapper autour de llama.cpp)
- **Format modèles** : GGUF (un seul fichier, compatible Metal et CUDA)
- **Interface cible** : lib Python + API HTTP compatible OpenAI

## Hardware dispo
- **Machine 1** : MacBook Apple M4, 16 Go RAM unifiée (12 Go utilisables pour les modèles), 10 CPU cores — backend Metal
- **Machine 2** : PC Nvidia RTX 4070, 12 Go VRAM — backend CUDA (non encore testée)

## Structure du projet
```
/Users/xavieregoneau/projets/runtime/
├── pyproject.toml
├── models/
│   └── Qwen2.5-7B-Instruct-Q4_K_M.gguf   # modèle de dev (4.4 Go)
└── llm_runtime/
    ├── __init__.py
    ├── hardware.py   # détecte Metal / CUDA / CPU → HardwareProfile
    ├── params.py     # calcule n_gpu_layers, n_threads, n_ctx → InferenceParams
    ├── engine.py     # charge un GGUF, génère du texte (.generate) et du chat (.chat)
    ├── models.py     # indexe les GGUFs disponibles (blobs Ollama + fichiers locaux)
    ├── server.py     # API HTTP compatible OpenAI (FastAPI)
    ├── batch.py      # traitement batch parallèle multi-process (workers indépendants)
    └── cli.py        # CLI `c3po` (list, info, run, serve, batch)
```

## Phases de dev

- [x] **Phase 1** — Détection hardware (`hardware.py`) ✅
- [x] **Phase 2** — Moteur d'inférence (`engine.py`, `params.py`) ✅
- [x] **Phase 3** — Indexation modèles (`models.py`) ✅
- [x] **Phase 4** — API HTTP compatible OpenAI (`server.py`) ✅
  - `GET /v1/models`, `POST /v1/chat/completions` (streaming SSE + non-streaming), `GET /health`
  - Lancement : `LLM_RUNTIME_MODEL=models/Qwen2.5-7B-Instruct-Q4_K_M.gguf python3 -m uvicorn llm_runtime.server:app --port 8000`
  - Sans `LLM_RUNTIME_MODEL`, utilise `best_model()` (scanne `./models` + Ollama)
  - Testé avec succès (non-streaming + streaming) sur Metal/M4

- [x] **Phase 5** — CLI `c3po` + batch parallèle (`cli.py`, `batch.py`) ✅
  - `c3po list` / `c3po info` / `c3po run [<modèle>]` (chat interactif) / `c3po serve [<modèle>] [--port]`
  - `c3po batch <modèle> --input <fichiers> --prompt "...{content}..." --output results.json`
    - Traite chaque fichier dans un worker séparé (`multiprocessing.Pool`), modèle chargé
      une fois par worker (`_worker_init`)
    - `optimal_jobs()` calcule le nb de workers selon mémoire dispo / taille du modèle
    - `_validate_model()` charge le modèle une fois en `n_ctx=512` avant de lancer le pool,
      pour échouer proprement plutôt que de boucler sur des workers cassés
  - Testé avec succès : `c3po batch qwen --input doc1.txt doc2.txt doc3.txt --prompt "Résume en une phrase : {content}" --output results.json` → 3/3 succès, ~13s, 2 workers
  - Note : sur macOS, le mode `spawn` de `multiprocessing` réimporte le module `cli` dans
    chaque worker (warning `RuntimeWarning: 'llm_runtime.cli' found in sys.modules...`),
    sans impact — c'est juste du bruit au démarrage de chaque worker.

## Phase 6 — Robustesse & reproductibilité ✅

Suite à une revue critique du projet (8 points), correctifs appliqués :

- **`requires-python = ">=3.9"`** dans `pyproject.toml`, alignée sur l'interpréteur
  réel (`/usr/bin/python3`, Python 3.9.6, CommandLineTools). Audit des `X | None` :
  seul `server.py` était concerné et utilisait déjà `Optional[str]` pour les modèles
  pydantic (eager evaluation).
- **Dépendances PEP 621 corrigées** : `[project.dependencies]` était une table TOML
  invalide (silencieusement ignorée par hatchling) → remplacée par
  `dependencies = [...]` (liste) directement dans `[project]`. C'est pour ça que
  `fastapi`/`uvicorn` n'étaient pas installés automatiquement avant.
- **Dev deps** : `[project.optional-dependencies] dev = ["pytest>=8.0"]`.
  Installation : `python3 -m pip install --user --upgrade pip` (pip ≥ 21.3 requis
  pour les installs editable PEP 660 avec hatchling), puis
  `python3 -m pip install --user -e ".[dev]"`.
- **`requirements-lock.txt`** : snapshot des versions validées via
  `python3 -m pip freeze --user > requirements-lock.txt` (à régénérer après tout
  changement de dépendances).
- **Suite de tests** (`tests/`, 20 tests, ~7s, aucun GPU/GGUF requis) :
  - `test_params.py` — `compute_params()` pour METAL/CUDA/CPU
  - `test_hardware.py` — `_detect_nvidia()` avec `nvidia-smi` mocké (couvre la
    logique CUDA sans accès à la RTX 4070)
  - `test_models.py` — `list_models()`, dédup, tri, `best_model()`
  - `test_batch.py` — `tasks_from_files`/`tasks_from_prompts`, `optimal_jobs()`
  - `test_server.py` — modèles pydantic `ChatMessage`/`ChatCompletionRequest`
  - Lancer : `python3 -m pytest tests/ -v`
- **`optimal_jobs()` tient compte du KV cache** : nouveau paramètre `n_ctx`,
  overhead heuristique `kv_overhead_gb = (n_ctx / 4096) * 1.0`, mémoire par
  instance = `model_size_gb * 1.15 + kv_overhead_gb`. `cli.py::cmd_batch` passe
  `args.ctx`. Vérifié : avec `--ctx 4096` (vs 2048 par défaut), le batch passe de
  2 à 1 worker sur la config M4/12 Go.
- **`tokens_per_second`** : `BatchResult.tokens` rempli depuis
  `usage.completion_tokens`, `BatchSummary.tokens_per_second` calculé dans
  `run_batch()` et inclus dans `results.json` (`summary.tokens_per_second`).
  Vérifié : `12.11` tokens/s sur un run de 3 documents.
- **`server.py`** :
  - `threading.Lock` global autour de `engine.chat()` (streaming + non-streaming) —
    `llama_cpp.Llama` n'est pas thread-safe pour des générations concurrentes.
    Vérifié : 2 requêtes `/v1/chat/completions` concurrentes → 200/200, sérialisées.
  - Champ `model` de la requête validé contre le modèle réellement chargé
    (`engine.model_path.stem`/`.name`) → `HTTPException(400, ...)` si différent.
    Vérifié : `model: "wrong-model"` → 400.
  - `get_engine()` : `try/except` élargi à `RuntimeError`, `FileNotFoundError`,
    `ValueError` → 503 au lieu d'une 500 générique.
- **Point d'entrée `c3po`** : `[project.scripts] c3po = "llm_runtime.cli:main"`
  (actif via `pip install -e .`, installé dans `~/Library/Python/3.9/bin/c3po`)
  est désormais la source canonique. Le wrapper bash `./c3po` à la racine est
  conservé temporairement car `/usr/local/bin/c3po` (symlink, nécessite `sudo`
  pour être repointé) pointe encore vers lui — à finaliser manuellement :
  ```bash
  sudo ln -sf /Users/xavieregoneau/Library/Python/3.9/bin/c3po /usr/local/bin/c3po
  rm /Users/xavieregoneau/projets/runtime/c3po
  ```

## Phase 7 — Swap de modèle façon Ollama, pre-commit, garde-fou multi-instances ✅

- **`find_model(query, local_dirs) -> ModelInfo`** (`models.py`) : factorise la
  résolution d'un nom de modèle (chemin direct, ou sous-chaîne du nom parmi
  `list_models()`). Lève `ValueError` si introuvable ou ambigu (plusieurs
  correspondances). `cli._resolve_model` est maintenant un fin wrapper autour
  de cette fonction (comportement CLI inchangé : auto-sélection via
  `best_model()` si aucun modèle n'est précisé).

- **Swap de modèle à la volée dans `server.py`** (façon `ollama run <modèle>`) :
  - `get_engine(model_query: str | None = None)` :
    - `model_query` fourni → résolu via `find_model()`. Si différent du
      modèle actuellement chargé, l'ancien `Engine` est libéré (`del` +
      `gc.collect()`) et le nouveau est chargé.
    - Si c'est le même modèle → cache hit, pas de rechargement.
    - `model_query is None` → comportement historique (`LLM_RUNTIME_MODEL` ou
      `best_model()`), sans swap implicite si un moteur est déjà chargé.
  - `chat_completions()` : `ValueError` (modèle introuvable/ambigu) → 400,
    `RuntimeError`/`FileNotFoundError` → 503. Le swap + la génération restent
    sérialisés par `_engine_lock`.
  - `GET /health` retourne désormais `"model": <nom du modèle chargé ou null>`.
  - Vérifié en conditions réelles : `LLM_RUNTIME_MODEL=models/Qwen2.5-7B-...gguf
    c3po serve` puis requêtes `/v1/chat/completions` avec `model: "qwen..."`
    (chargement puis cache hit) et `model: "inconnu"` (→ 400).

- **Garde-fou multi-instances** (`llm_runtime/instances.py`, nouveau) :
  - `register_instance(model_name, size_gb)` écrit
    `<tmpdir>/c3po-instances/<pid>.json` (`pid`, `model`, `size_gb`,
    `started_at`), avec nettoyage `atexit`.
  - `active_instances(exclude_pid=None)` liste les instances dont le PID est
    encore vivant, en nettoyant les fichiers orphelins (process mort sans
    nettoyage propre).
  - `check_memory_pressure(new_size_gb, available_gb)` : si
    `(new_size_gb + somme des instances actives) * 1.15 > available_gb`,
    retourne un message d'avertissement listant les instances concurrentes —
    **avertissement seul, jamais bloquant**.
  - `Engine.__init__` calcule `self.size_gb`, affiche l'avertissement sur
    `stderr` si pression mémoire détectée, puis `register_instance(...)` après
    chargement réussi.
  - `c3po list` affiche une section "Instances actives" (PID, modèle, taille)
    si des instances tournent. Vérifié : `c3po serve` dans un terminal +
    `c3po list` dans un autre → instance visible avec PID/taille corrects ;
    le fichier de lock disparaît à l'arrêt du serveur.

- **Hook pre-commit pour `requirements-lock.txt`** (anti-drift, suite Phase 6) :
  - `pre-commit>=3.0` ajouté à `[project.optional-dependencies] dev`.
  - `scripts/regen_lock.sh` (exécutable) régénère
    `requirements-lock.txt` via `pip freeze --user` et le `git add`.
  - `.pre-commit-config.yaml` : hook local `regen-lock`, déclenché uniquement
    si `pyproject.toml` change.
  - `.gitignore` (nouveau) : `__pycache__/`, `.pytest_cache/`, `models/*.gguf`,
    `tmp/`, `results*.json`.
  - Installation : `python3 -m pip install --user -e ".[dev]"` puis
    `pre-commit install`. Comportement standard : si le hook modifie
    `requirements-lock.txt`, le premier `git commit` est rejeté — il faut
    `git add requirements-lock.txt` et recommit.

- **Tests** (33 au total, `python3 -m pytest tests/ -v`) :
  - `test_models.py` — `find_model()` (chemin direct, sous-chaîne, ambigu,
    introuvable)
  - `test_instances.py` (nouveau) — `register_instance`/`active_instances`
    (PID mort nettoyé, PID vivant conservé), `check_memory_pressure` (OK et
    dépassement)
  - `test_server.py` — `get_engine()` avec `Engine` mocké (premier chargement,
    cache hit, swap, `ValueError` sur modèle inconnu)

### Validation Machine 2 (CUDA) — ✅ faite (15/06/2026, RTX 4070, Ubuntu 24.04)

Environnement de la Machine 2 :
- Ubuntu 24.04, Python 3.11.11 (pyenv), driver Nvidia 580 / CUDA 13.0, RTX 4070 12 Go.
- **Pas de wheel CUDA précompilée utilisable** : l'index `abetlen.github.io/.../whl/cuXXX`
  plafonne à `llama-cpp-python` 0.2.66 (avril 2024, Linux/py311), trop ancien pour
  Qwen2.5. → **compilation depuis les sources obligatoire** :
  ```bash
  sudo apt install -y nvidia-cuda-toolkit        # nvcc 12.0 (multiverse), suffit
  python3 -m pip install cmake                   # cmake absent par défaut
  CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
      python3 -m pip install --no-cache-dir llama-cpp-python
  ```
  Build OK en ~4 min → `llama-cpp-python 0.3.29` (même version que le Mac),
  `llama_supports_gpu_offload() == True`, GPU détecté à l'import.

Checklist :
- [x] `c3po info` → `Backend: cuda`, `NVIDIA GeForce RTX 4070`, GPU memory 11.0 Go
      (VRAM libre), `n_gpu_layers: toutes couches` (-1), flash_attn True. Conforme à
      `_params_cuda`.
- [x] `c3po list` → modèle local listé, `FIT ✓`. (chat interactif `c3po run` non
      déroulé en TTY, mais `serve`/`batch` exercent le même chemin `Engine.chat`.)
- [x] `c3po batch qwen --ctx 4096` → 3/3 succès, 2.3s, **24.3 tok/s** (~2× le M4),
      1 worker (cohérent : `model*1.15 + kv ≈ 6 Go` vs 11 Go libres).
- [x] `c3po serve` + `/v1/chat/completions` streaming **et** non-streaming → OK,
      `/health` reflète le modèle chargé. VRAM observée à 5.5 Go pendant le service
      (modèle + KV + contexte CUDA bien offloadés sur GPU), libérée à l'arrêt.

**Bug CUDA trouvé et corrigé — `batch.py` : `fork` → `spawn`.**
Sur Linux, `multiprocessing` utilise `fork` par défaut. `run_batch` appelait
`_validate_model` (qui initialise le backend CUDA dans le process parent) **avant** de
créer le `mp.Pool`. Les workers forkés héritaient alors d'un contexte CUDA invalide
→ segfault des workers (observé : worker en état `t`, gdb attaché dumpant la backtrace,
process principal figé indéfiniment). Sur macOS le défaut est `spawn`, d'où l'absence
du bug là-bas (et le `RuntimeWarning` de réimport déjà noté en Phase 5).
Correctif : `ctx = mp.get_context("spawn"); ctx.Pool(...)` dans `run_batch`.
Comportement désormais identique sur les deux plateformes. Après fix : batch 3/3 OK.

Note opérationnelle : `c3po serve` lance `uvicorn llm_runtime.server:app` comme
**process séparé** (pas le même PID que `c3po`) — un `pkill -f "c3po serve"` ne le
trouve pas ; viser `pkill -f uvicorn` ou Ctrl-C en avant-plan.

## Phase 8 — `c3po stats`, `c3po load` (téléchargement HF), dossier modèles utilisateur ✅

- **`c3po stats <modèle> [--ctx N] [--json]`** (`stats.py`, nouveau) : métadonnées GGUF
  (architecture, quantization, nb de params si présent, contexte d'entraînement, couches,
  embedding, vocab) + paramètres d'inférence appliqués + **benchmark réel** (temps de
  chargement, time-to-first-token, tokens/s en génération, VRAM consommée). Le benchmark
  charge le modèle et génère ~128 tokens à temp 0. Sur RTX 4070 / Qwen2.5-7B Q4_K_M :
  chargement 1.6 s, TTFT 79 ms, **91.7 tok/s**, 4857 Mo de VRAM. (`tok/s` mesuré ici est
  la génération pure, ≠ le débit wall-clock du batch qui inclut chargement + spawn.)
  Mesure annexe : la génération décroît doucement avec le contexte (90→78 tok/s de ~vide
  à 6k tokens, ~14 %), alors que le TTFT, lui, croît ~linéairement avec la taille du
  prompt (82 ms → 1.7 s à 6k tokens, c'est le prefill).

- **`c3po load <user>/<repo>[:QUANT]`** (`download.py`, nouveau) : téléchargement de GGUF
  depuis Hugging Face en **HTTP brut (urllib stdlib, zéro dépendance, sans CLI ni token
  requis)** — choix délibéré pour rester installable tel quel par n'importe quel
  utilisateur de la lib (on n'a pas utilisé le CLI HF de la machine).
  - Référence **explicite obligatoire** (`<user>/<repo>`), pas d'alias opaque : on sait ce
    qu'on télécharge. `:QUANT` ou `--quant` pour forcer.
  - Résolution via l'API publique `…/api/models/<repo>/tree/main?recursive=1` (tailles +
    `lfs.oid` = SHA256). Regroupement par quant, **gestion des shards** (`-00001-of-…`).
  - **Quant auto selon la VRAM** : plus grosse quant dont `taille × 1.15 ≤ gpu_memory_gb`
    (= `best_model()` au moment du download) ; sinon la plus petite + avertissement.
  - Téléchargement robuste : **reprise** (`Range:` + `.part` renommé à la fin),
    **progression** (réécrite sur TTY, sparse tous les 10 % si capturée), **vérif SHA256**
    contre l'oid LFS. `401` → message gated (honore `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`
    s'il existe, sans jamais l'exiger).
  - Validé en réel : listing des 24 quants du repo bartowski Qwen2.5-7B (choix auto = Q8_0
    pour 10.8 Go), download + reprise + intégrité OK sur le 0.5B.

- **`c3po search <requête> [--limit N] [--all]`** (`download.py`) : découverte de modèles
  GGUF sur HF éligibles au hardware courant. Recherche via `…/api/models?filter=gguf&
  sort=downloads`, puis un appel `tree` par repo (parallélisés par un `ThreadPoolExecutor`,
  ~1 s pour 15 repos) pour récupérer les tailles et calculer, via `best_fitting_quant()`,
  la meilleure quant qui tient dans `gpu_memory_gb`. Filtre sur les éligibles par défaut
  (`--all` pour tout voir, colonne `FIT`), tri éligibles d'abord puis par téléchargements.

- **Correction du regroupement par quant** (`group_by_quant`, bénéficie à `load` ET
  `search`) : on regroupe désormais par *base* (nom sans suffixe de shard) avant de mapper
  vers une quant. Évite deux bugs vus en conditions réelles : (1) un repo offrant une même
  quant en fichier unique **et** en shards (ex. Qwen2.5-Coder-7B) voyait ses tailles
  additionnées (8.7 Go au lieu de 7.5) ; (2) sur les repos multimodaux, un `mmproj`
  partageant un tag (`BF16`) était fusionné au modèle. Désormais : une représentation par
  base (shards complets sinon fichier unique), et en cas de collision de quant entre deux
  bases on garde la plus grosse (le vrai modèle, pas le mmproj).

- **Dossier modèles utilisateur** (`models.py`) : `models_dir()` = `$C3PO_MODELS_DIR` ou
  `~/.c3po/models` (le repo est une lib, les modèles sont du contenu utilisateur → hors du
  repo). `list_models()`/`find_model()`/`best_model()` scannent par défaut `~/.c3po/models`
  **+** `./models` (compat) **+** Ollama. `find_model` tolère désormais un suffixe `.gguf`
  dans la requête.

- **Tests** (47 au total) : `test_download.py` (parse_ref ; regroupement par quant : extraction,
  fusion des shards, non-addition unique+shards, collision de quant ; choix de quant
  auto/explicite/inconnue/fallback ; `best_fitting_quant`), `test_models.py` (suffixe `.gguf`).
  Aucune dépendance ajoutée (urllib + concurrent.futures stdlib) → `requirements-lock.txt` inchangé.

## Phase 9 — Corrections post-revue ✅

Suite à une revue critique du projet, correction des bugs concrets identifiés :

- **Flash attention jamais activée** (`engine.py`) : on passait `use_flash_attn=…` à
  `llama_cpp.Llama`, mais le vrai paramètre est `flash_attn`. Comme `Llama` accepte
  `**kwargs`, l'argument était silencieusement ignoré → toute la logique
  `params.use_flash_attn` était morte. Corrigé en `flash_attn=…`. Vérifié :
  `llama_context: flash_attn = enabled` au chargement.
- **`server.py` ignorait `~/.c3po/models`** : les appels utilisaient encore
  `local_dirs=["models"]` (régression Phase 8). Remplacés par les défauts
  (`find_model`/`list_models`/`best_model` sans `local_dirs`). Vérifié : `/v1/models`
  liste bien les modèles de `~/.c3po/models`.
- **VRAM codée en dur dans le serveur** : `best_model(available_memory_gb=12.0)` →
  `best_model(detect_hardware().gpu_memory_gb)`.
- **Serveur exposé sur `0.0.0.0` sans auth** : défaut passé à `127.0.0.1`, avec un
  flag `--host` pour ouvrir explicitement sur le réseau (avertissement affiché si
  `0.0.0.0`).

**`_params_cuda` étoffé (point #5 de la revue) + lecteur d'en-tête GGUF** (`gguf.py`,
nouveau). `gguf.py::model_shape()` lit `n_layers`, `n_ctx_train`, `n_embd` directement
dans l'en-tête GGUF (stdlib `struct`, sans charger le modèle ni llama.cpp ; les gros
tableaux comme le vocabulaire tokenizer sont sautés par `seek`). `compute_params` accepte
désormais `model_size_gb` / `n_layers` / `n_ctx_train`, et `Engine` les fournit. Nouveau
comportement CUDA :
- `n_ctx` plafonné au contexte d'entraînement du modèle (jamais plus) ;
- tout sur GPU si poids (×1.05) + KV estimé tiennent dans la VRAM (− 0.8 Go de réserve) ;
- sinon **réduction du contexte** (3072→512) jusqu'à ce que ça tienne ;
- en dernier recours, **offload partiel chiffré** `n_gpu_layers = n_layers × (budget/poids)`
  au lieu de tenter `-1` et faire OOM (llama.cpp ne fait pas de spill automatique).
Vérifié : 7B sur 4070 → -1/4096 ; 7B simulé sur 6 Go → -1/2048 ; 14B (8.5 Go) sur 4070 →
-1/2048 (tient au lieu d'OOM à 4096). Le chemin sans info modèle reste inchangé
(rétrocompat des tests). KV cache encore approximé (×0.20 × taille × ctx/4096) faute de
config d'attention exacte (GQA).

**Batch limité à 1 worker sur CUDA (point #8 de la revue)** : `optimal_jobs()` retourne
désormais `1` quand le backend est CUDA. Sur une carte Nvidia la VRAM est dédiée et le GPU
sérialise les kernels → plusieurs workers ne font que dupliquer le modèle en VRAM (risque
d'OOM) sans gain de débit. Metal (mémoire unifiée, grande RAM) et CPU gardent le calcul
multi-worker existant ; `--jobs` permet toujours de forcer. Vérifié : `c3po batch` sur la
4070 annonce « 1 worker(s) » sans `--jobs` (contre 2 auparavant).

**VRAM déterministe (point #6) + marge mémoire centralisée (point #7).**
- `_detect_nvidia` part désormais de la VRAM *totale* moins une réserve fixe
  (`_NVIDIA_RESERVE_GB = 1.0`) au lieu de la VRAM *libre*. La VRAM libre fluctuait avec
  l'usage du bureau → `c3po list`/`search`/`info` donnaient des résultats qui bougeaient
  d'un appel à l'autre. Sur la 4070 : 11.0 Go stable (12.0 − 1.0), au lieu de 10.8/11.0
  variable. Query nvidia-smi simplifiée (`name,memory.total`).
- Marge mémoire unique `models.FIT_MARGIN = 1.15` (anciennement ×1.1 dans `fits_in` et
  ×1.15 dispersés dans `optimal_jobs`, `check_memory_pressure`, `download`). Une seule
  source de vérité pour « est-ce que ce modèle tient ».

Points de la revue restant ouverts (non bugs, décisions de conception) : `search`/`load`
trompeurs sur les modèles multimodaux (mmproj non téléchargé) — dernier point notable.

## Problème résolu — Gemma 3n (gemma4) non chargeable

Le modèle **gemma4:e4b** via Ollama (blob `sha256-4c27e0f5...`, ~8.9 Go) est un GGUF valide
(magic `GGUF`, version 3) mais correspond à **Gemma 3n**, une architecture multimodale
(texte + audio + vision, 2131 tensors). `llama-cpp-python==0.3.29` (dernière version PyPI)
ne construit que les 720 tensors du décodeur texte et échoue sur un contrôle strict :

```
llama_model_load: error loading model: done_getting_tensors: wrong number of tensors; expected 2131, got 720
ValueError: Failed to load model from file: ...sha256-4c27...
```

**Conclusion** : support incomplet de l'architecture `gemma4` dans cette version de
llama.cpp — pas un problème de fichier. Pour débloquer le dev, on utilise à la place
**Qwen2.5-7B-Instruct-Q4_K_M** (`models/`, téléchargé depuis
`bartowski/Qwen2.5-7B-Instruct-GGUF`), qui charge et génère correctement.

Si Gemma 3n est nécessaire plus tard : builder `llama-cpp-python` depuis les sources
contre le `main` de llama.cpp (le support audio/vision de Gemma 3n a continué d'évoluer
après la release 0.3.29).

## Architecture cible complète
```
[utilisateur]
     ↓
[llm_runtime] ← ce qu'on construit
  - détecte le hardware
  - choisit le modèle (quantization adaptée à la VRAM/RAM)
  - calcule les paramètres optimaux (n_gpu_layers, threads, ctx)
  - expose generate() et une API HTTP /v1/chat/completions
     ↓
[llama-cpp-python] ← runtime, on l'utilise tel quel
     ↓
[Metal ou CUDA] ← backend GPU
     ↓
[fichier GGUF] ← le modèle
```

## Concepts clés
- **GGUF** : format de modèle portable, un seul fichier fonctionne sur Metal et CUDA
- **n_gpu_layers** : nb de couches envoyées sur GPU (-1 = toutes, utile sur Apple car RAM unifiée)
- **quantization** : Q4_K_M = léger/rapide, Q8_0 = qualité proche de l'original mais 2x plus lourd
- **RAM unifiée Apple** : CPU et GPU partagent la même mémoire → avantage énorme vs VRAM dédiée
- **Ollama blobs** : les modèles Ollama sont des GGUFs stockés sans extension `.gguf`, nommés par hash SHA256
