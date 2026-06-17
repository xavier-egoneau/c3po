# c3po — llm-runtime

Orchestrateur Python autour de [llama.cpp](https://github.com/ggerganov/llama.cpp), avec
détection automatique du hardware (Apple Silicon / Metal ou Nvidia / CUDA) et calcul des
paramètres d'inférence optimaux. Un peu comme Ollama, mais avec les leviers exposés plutôt
que cachés.

## Installation

### Windows + Nvidia

Le chemin recommande sous Windows est un environnement virtuel Python 3.11/3.12 avec une wheel
CUDA precompilee. C'est le plus fiable : Python 3.13 peut compiler `llama-cpp-python` en CPU-only
si le CUDA toolkit (`nvcc`) n'est pas installe.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows_cuda.ps1
.\.venv\Scripts\Activate.ps1
c3po doctor
```

Verification attendue :

```text
Offload GPU       : oui
```

Si `c3po` pointe encore vers un Python global, lance directement :

```powershell
.\.venv\Scripts\c3po.exe doctor
.\.venv\Scripts\c3po.exe stats <modèle>
```

### Installation standard

```bash
python3 -m pip install --user -e ".[dev]"
```

Sous Windows, si `c3po` n'est pas reconnu apres l'installation standard, ajoutez le dossier Scripts
utilisateur au `PATH` :

```powershell
$scripts = python -c "import sysconfig; print(sysconfig.get_path('scripts', scheme='nt_user'))"
$env:Path = "$scripts;$env:Path"
```

**Apple Silicon (Metal)** : le backend Metal est activé par défaut, rien de plus à faire.

**Linux + Nvidia (CUDA)** : `llama-cpp-python` doit être compilé avec le backend CUDA (nécessite le
CUDA toolkit `nvcc` + `cmake`) — les wheels précompilées sont trop anciennes pour les modèles
récents :

```bash
sudo apt install -y nvidia-cuda-toolkit          # fournit nvcc (ou repo NVIDIA)
python3 -m pip install cmake
CMAKE_ARGS="-DGGML_CUDA=on" python3 -m pip install --no-cache-dir llama-cpp-python
python3 -m pip install --user -e ".[dev]"
```

## Utilisation

```bash
# Liste les modèles disponibles (~/.c3po/models + ./models + Ollama) et le hardware détecté
c3po list

# Affiche le profil hardware et les paramètres d'inférence calculés
c3po info

# Diagnostique l'installation (llama-cpp-python, backend GPU) et, optionnellement, un modèle
c3po doctor
c3po doctor <modèle>

# Bench réel ; vérifiez surtout "Offload GPU : oui" et la VRAM modèle
c3po stats <modèle>

# Cherche sur Hugging Face les modèles GGUF qui tiennent dans ta VRAM
c3po search qwen2.5            # éligibles seulement
c3po search "llama 3" --all   # tout, avec colonne FIT
c3po search mistral --limit 30  # inspecter plus de repos (défaut : 20)

# Télécharge un modèle GGUF depuis Hugging Face (quant auto selon la VRAM)
c3po load bartowski/Qwen2.5-7B-Instruct-GGUF          # quant choisie selon la VRAM
c3po load bartowski/Qwen2.5-7B-Instruct-GGUF:Q5_K_M   # quant forcée
c3po load unsloth/gemma-4-E2B-it-qat-GGUF:Q4_K_XL --mmproj  # inclut le projecteur multimodal

# Chat interactif
c3po run <modèle>
# L'historique est compacté automatiquement quand il approche 95% du contexte du modèle.
# Chaque réponse est aussi bornée au budget restant pour éviter d'atteindre la fin du contexte.

# Leviers d'inférence exposés (sinon auto-calculés selon le hardware) — sur run/stats/serve/batch
c3po run <modèle> --ctx 8192 --n-gpu-layers 20 --threads 6 --no-flash-attn
# Quantization du KV cache (réduit la VRAM sur long contexte). Auto : F16, ou Q8 si besoin
# pour faire tenir le contexte ; --kv-type q4 force le mode le plus compact (qualité moindre).
c3po run <modèle> --ctx 32768 --kv-type q8

# Speculative decoding (prompt-lookup) : accélère les sorties qui recopient l'entrée
# (code, RAG, édition) — sortie identique, juste plus rapide. À éviter sur du texte créatif.
c3po run <modèle> --speculative

# Pénalité de répétition (défaut 1.1 ; monter si le modèle boucle ; 1.0 = aucune)
c3po run <modèle> --repeat-penalty 1.3

# Serveur HTTP compatible OpenAI (GET /v1/models, POST /v1/chat/completions, GET /health)
# Écoute sur 127.0.0.1 par défaut ; --host 0.0.0.0 pour exposer sur le réseau (sans auth !)
c3po serve [<modèle>] [--port 8000] [--host 127.0.0.1]

# Traitement batch (file séquentielle sur une seule instance modèle)
c3po batch <modèle> --input fichier1.txt fichier2.txt --prompt "Résume : {content}" \
    --output results.json [--ctx 2048] [--max-tokens N]
```

Par défaut, `run`/`serve`/`batch` génèrent jusqu'à la fin de la réponse (ou la limite de
contexte) — `--max-tokens` ne sert qu'à borner volontairement.

Les modèles téléchargés via `c3po load` vont dans `~/.c3po/models` (surchargeable via
`C3PO_MODELS_DIR`). Vous pouvez aussi déposer des `.gguf` dans `./models/` (non versionnés,
voir `.gitignore`) — les deux dossiers sont scannés.

## Compatibilité

Les GGUF exportés par Unsloth ou convertis depuis un fine-tune sont utilisables directement si
leur architecture est supportée par la version de `llama.cpp` embarquée dans
`llama-cpp-python`. En cas de doute : `c3po doctor <modèle>` inspecte l'environnement, le
backend GPU, l'en-tête GGUF, signale les projecteurs multimodaux `mmproj` et propose les
commandes de rebuild adaptées si `llama-cpp-python` est absent ou compilé sans offload GPU.
Il affiche aussi le Python exact et le binaire `c3po` utilisés, utile pour repérer un décalage
entre un Python système et un environnement pyenv/venv.

Pour les modèles très récents, le point faible est souvent la version de `llama.cpp` incluse
dans `llama-cpp-python`. `c3po doctor <modèle>` aide à distinguer un problème d'installation,
un GGUF illisible, un modèle trop gros, un `mmproj` multimodal, ou une architecture qui demande
une version plus récente de llama.cpp.

Les commandes qui chargent un modèle (`run`, `serve`, `stats`, `batch`) arrêtent d'abord les
autres instances c3po actives. Le projet privilégie une seule instance modèle vivante à la fois :
moins de copies en VRAM/RAM, calculs mémoire plus prévisibles, moins de risques d'OOM.

Aujourd'hui c3po sert d'abord les modèles texte GGUF. Le multimodal n'est pas exclu du projet :
il demande une extension dédiée (téléchargement/association des `mmproj`, messages image/audio,
budget contexte multimodal, backend `libmtmd` ou délégation à `llama-server`).
`c3po load --mmproj` permet déjà de récupérer le projecteur multimodal quand un repo en fournit
un, sans encore brancher le runtime image/audio.

## Développement

```bash
python3 -m pytest tests/ -v
pre-commit install

# Bench perf local reproductible (hors CI)
scripts/bench_perf.sh
```

Voir [ARCHITECTURE.md](ARCHITECTURE.md) pour l'état actuel du système, et
[CONTEXT.md](CONTEXT.md) pour le journal chronologique des décisions techniques.
