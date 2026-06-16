# c3po-agent — Roadmap

## Positionnement

`c3po-agent` est une couche expérimentale au-dessus de `c3po-core`.

Son objectif n'est pas de remplacer l'outil agentique final, mais de rendre les modèles
locaux plus substituables à des providers agentiques comme Codex ou Claude.

```
[outil agentique final]
  objectif produit, UX, permissions, mémoire métier, workflows haut niveau
       ↓
[c3po-agent]
  échafaudage agentique pour modèles locaux
       ↓
[c3po-core]
  inférence locale GGUF, hardware, stats, doctor, streaming/API
```

Boussole :

> `c3po-agent` augmente la substituabilité des modèles locaux dans un outil agentique,
> sans capturer la logique produit de cet outil.

## Responsabilités

### L'outil agentique final décide

- objectif utilisateur et stratégie produit ;
- UX et interactions ;
- permissions globales ;
- mémoire métier ;
- choix du provider : Codex, Claude, Ollama, c3po-agent ;
- workflows haut niveau.

### c3po-agent décide localement

- comment obtenir une réponse structurée fiable d'un petit modèle ;
- quand retry ;
- comment valider une sortie ;
- comment transformer une intention outil en JSON utilisable ;
- comment découper une sous-tâche locale ;
- comment évaluer une réponse avant de la remonter.

### c3po-core ne devient pas agentique

- charge et sert les modèles locaux ;
- applique les paramètres d'inférence ;
- expose une API stable ;
- stream les réponses ;
- fournit stats, doctor, benchmarks ;
- protège le contexte et la VRAM.

## Axes d'exploration

### 1. Capability Registry

Décrire honnêtement les capacités d'un backend local :

```json
{
  "chat": true,
  "streaming": true,
  "tools": "native|prompted|none",
  "structured_outputs": "native|prompted|validated|none",
  "vision": false,
  "context_window": 32768,
  "local": true
}
```

But : permettre à l'outil final de dégrader proprement selon le provider.

### 2. Structured Outputs Validés

Ajouter un mode où c3po-agent demande au modèle une sortie JSON, valide le schéma,
et retry avec l'erreur de validation si nécessaire.

Objectifs :

- réduire les sorties non parseables ;
- normaliser les décisions de type `run_tool`, `answer`, `ask_clarification` ;
- éviter les regex fragiles.

### 3. Tool Calling Émulé

Supporter les modèles sans tool calling natif en leur faisant produire une action structurée :

```json
{
  "action": "tool_call",
  "tool": "read_file",
  "args": {"path": "README.md"}
}
```

c3po-agent ne devrait pas exécuter les outils arbitrairement. Il prépare et valide la demande ;
l'outil agentique final garde les permissions et l'exécution réelle.

### 4. State Machine Locale

Explorer une boucle explicite pour sous-tâches cadrées :

```
PLAN -> ACT -> OBSERVE -> CHECK -> ANSWER
```

Usage visé : petits modèles locaux qui ont besoin d'un cadre plus strict qu'un provider
frontier.

Garde-fou : cette state machine reste locale à une sous-tâche. Elle ne remplace pas le
workflow haut niveau de l'outil agentique final.

### 5. Evaluator Léger

Ajouter une étape de vérification avant de remonter une réponse :

- JSON valide ?
- champs requis présents ?
- réponse conforme à la consigne ?
- outil demandé autorisé par le contrat ?
- réponse trop longue ou trop vague ?

À utiliser de façon ciblée : cela coûte tokens et latence.

### 6. Prompt Evaluation Harness

Créer un harnais de prompts pour mesurer le comportement :

- smoke tests ;
- instruction following ;
- JSON extraction ;
- tool call emulation ;
- compaction retention ;
- code/rewrite ;
- long context.

But : mesurer si l'échafaudage améliore réellement la substituabilité des modèles locaux.

### 7. Mémoire Structurée

Réutiliser l'idée de compaction structurée :

- objectif courant ;
- décisions ;
- contraintes ;
- fichiers/modèles ;
- état des tâches ;
- prochains pas.

Important : cette mémoire est une aide locale. La mémoire métier reste dans l'outil final.

### 8. Vision Sidecar

Rendre c3po-agent multimodal côté interface, même si le modèle principal est texte-only :

```
image + question
  -> modèle vision c3po-core
  -> observation structurée
  -> modèle texte c3po-core
  -> réponse finale
```

Premier candidat validé : `gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` avec `mmproj-BF16.gguf`,
téléchargés depuis `unsloth/gemma-4-E2B-it-qat-GGUF`.

Spike existant : `experiments/vision_gemma4.py` produit déjà une observation JSON à partir
d'une image via `llama-cpp-python` / `Gemma4ChatHandler`.

Garde-fou : ne pas dépendre d'Ollama dans c3po-agent. Si le modèle vision n'est pas servi par
c3po-core / llama.cpp, la capacité reste indisponible côté c3po.

## Non-objectifs initiaux

- Ne pas transformer `c3po-core` en agent.
- Ne pas exécuter des outils dangereux sans validation externe.
- Ne pas promettre qu'un petit modèle local devient équivalent à Codex/Claude.
- Ne pas imposer une mémoire serveur cachée à l'API OpenAI-compatible.
- Ne pas stabiliser trop tôt une API agentique avant évaluation.

## Première Milestone

Construire un prototype minimal hors API stable :

1. `agent/capabilities.py` : modèle de capacités backend.
2. `agent/structured.py` : validation JSON + retry. Base créée :
   - extraction d'objet JSON depuis une réponse brute ou fenced ;
   - validation de champs/types ;
   - retry générique avec retour d'erreurs ;
   - premier usage par `agent/vision/gemma4.py`.
3. `agent/prompts/` : prompts système pour petits modèles.
4. `agent/evals/` : suite smoke + JSON + tool-call emulation.
5. `agent/vision/` : encapsuler le spike `experiments/vision_gemma4.py` en primitive
   image -> observation structurée. Base créée avec Gemma 4 E2B + `mmproj`.
6. Un script manuel pour comparer :
   - modèle local brut ;
   - modèle local + c3po-agent scaffold ;
   - provider frontier si disponible.

Critère de succès : montrer au moins un cas où un petit modèle local devient plus fiable
avec l'échafaudage, sans déplacer la logique produit dans c3po.
