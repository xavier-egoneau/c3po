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

### Validation Machine 2 (CUDA) — à faire

Le chemin CUDA (`_detect_nvidia`, `_params_cuda`) n'est couvert que par des tests
avec `nvidia-smi` mocké. À valider sur la RTX 4070 :

- [ ] `c3po info` → vérifier `Backend: cuda`, VRAM détectée correcte (~12 Go),
      `n_gpu_layers` cohérent avec `_params_cuda` (-1 si VRAM ≥ 8 Go)
- [ ] `c3po run <modèle>` → chat interactif fonctionne, pas d'erreur CUDA
- [ ] `c3po batch <modèle> --input ... --ctx 4096 --output results.json` →
      `optimal_jobs()` retourne un nombre de workers cohérent avec la VRAM dispo
- [ ] `c3po serve` + requêtes `/v1/chat/completions` (streaming + non-streaming)

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
