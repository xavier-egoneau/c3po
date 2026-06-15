# Plan — à faire sur la Machine 1 (MacBook M4, Metal)

Tâches à exécuter / vérifier quand je serai sur le Mac, après la session de dev faite sur
le PC CUDA (commits jusqu'à `963e88b`). Cocher au fur et à mesure.

## 1. Récupérer et installer
- [ ] `git pull` sur `main`
- [ ] `python3 -m pip install --user -e ".[dev]"` (pas de nouvelle dépendance ajoutée,
      mais resync le lock / les nouveaux modules `download.py`, `gguf.py`, `stats.py`)
- [ ] `python3 -m pytest tests/ -v` → doit afficher **59 passed** (logique pure, sans GPU)

## 2. Vérifier que les fixes de la session ne régressent pas sur Metal
- [ ] **Flash attention** : `c3po stats <modèle>` puis vérifier au chargement
      `flash_attn = enabled` (le fix `use_flash_attn`→`flash_attn` doit aussi valoir sur Metal)
- [ ] **Batch `spawn`** : `c3po batch <modèle> --input a.txt b.txt c.txt --prompt "..."`
      → doit tourner sans warning bloquant (macOS était déjà en `spawn`, le changement est
      un no-op fonctionnel, à confirmer)
- [ ] **Batch multi-worker préservé sur Metal** : sur la mémoire unifiée, `optimal_jobs`
      doit toujours renvoyer **>1** worker si le modèle est petit (le forçage à 1 ne
      concerne que CUDA). Vérifier dans la sortie « N worker(s) ».
- [ ] **`_params_cuda` n'impacte pas Metal** : `c3po info` doit toujours montrer
      `n_gpu_layers = toutes` (-1) sur Metal ; vérifier juste que `n_ctx` est bien plafonné
      au contexte d'entraînement du modèle (nouveau comportement commun).
- [ ] **Lecteur GGUF** (`gguf.py`, stdlib, cross-plateforme) : `c3po stats <modèle>` doit
      afficher couches / contexte max / embedding corrects.

## 3. Tester les nouvelles commandes sur Metal
- [ ] `c3po search qwen2.5` → liste les modèles éligibles à la RAM unifiée (≈12 Go utiles),
      repos multimodaux marqués `*`
- [ ] `c3po load bartowski/Qwen2.5-7B-Instruct-GGUF` → download dans `~/.c3po/models`
      (quant auto selon mémoire), progression + reprise + vérif SHA256
- [ ] `c3po stats <modèle>` → benchmark Metal : **comparer les tok/s au CUDA**
      (réf. PC : ~91 tok/s gen, TTFT ~4 ms après warmup sur la 4070)
- [ ] **Leviers exposés** : `c3po run <modèle> --ctx 8192 --no-flash-attn --threads 6`
      → vérifier que les overrides sont bien appliqués (visible via `c3po stats … <flags>`)
- [ ] **KV cache quantifié** : `c3po stats <modèle> --kv-type q8` → doit charger sur Metal,
      afficher `kv_type q8_0`, flash activée. (Vérifier que `type_k`/`type_v` sont honorés sur
      le backend Metal comme sur CUDA — sinon ce serait un no-op silencieux.)
- [ ] `c3po serve` + requêtes `/v1/chat/completions` (streaming + non-streaming) — non régressé

## 4. Dossier modèles
- [ ] Le modèle de dev actuel est dans `./models/` (toujours scanné). Optionnel : le
      déplacer vers `~/.c3po/models` pour homogénéiser avec le PC.
- [ ] `C3PO_MODELS_DIR` disponible si on veut un autre emplacement.

## 5. Reliquat pré-session (note Phase 6 du CONTEXT.md) — à finaliser sur Mac
- [ ] Repointer le symlink système vers l'entry point pip et supprimer le wrapper bash racine :
      ```bash
      sudo ln -sf /Users/xavieregoneau/Library/Python/3.9/bin/c3po /usr/local/bin/c3po
      rm /Users/xavieregoneau/projets/runtime/c3po
      ```
      (à adapter selon le chemin réel d'install pip sur le Mac)

## 6. Points connus (pas des bugs — pour mémoire)
- Vision/audio non supportée (mmproj non chargé) — modèles multimodaux marqués/avertis.
- Gemma 3n non chargeable (limite `llama-cpp-python` 0.3.29).
- #12 de la revue : pas de CI GPU — la validation Metal reste manuelle (cette liste).
