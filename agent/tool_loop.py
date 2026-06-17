"""Boucle tool-calling émulée pour petits modèles sans tool calling natif."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal

from agent.structured import FieldSpec, parse_validated
from agent.tools import AgentToolRegistry, ToolResult, ToolSpec


Message = dict[str, str]
ChatModel = Callable[[list[Message]], str]


@dataclass
class ToolLoopStep:
    index: int
    action: str
    raw: str
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    observation: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    plan: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolLoopResult:
    answer: str
    memory_after: str
    steps: list[ToolLoopStep]
    stopped_reason: Literal["answer", "max_steps", "invalid_action"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_tool_loop(
    prompt: str,
    model: ChatModel,
    tools: AgentToolRegistry,
    max_steps: int = 6,
    memory: str = "",
    max_context_chars: int = 12_000,
    require_artifact_audit: bool = False,
) -> ToolLoopResult:
    """
    Laisse le modèle demander des outils via JSON, puis renvoie les observations.

    Le modèle ne peut pas exécuter directement : chaque action est validée,
    bornée par `max_steps`, puis passée à la registry.
    """
    steps: list[ToolLoopStep] = []
    messages = _initial_messages(prompt, tools.specs(), memory)
    messages = _compact_messages_if_needed(messages, steps, max_context_chars)

    for index in range(1, max_steps + 1):
        raw = model(messages)
        parsed, errors = parse_tool_action(
            raw,
            known_tools=[spec.name for spec in tools.specs()],
        )
        if errors:
            step = ToolLoopStep(
                index=index,
                action="invalid",
                raw=raw,
                error="; ".join(errors),
            )
            steps.append(step)
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": _retry_observation(errors)},
                ]
            )
            messages = _compact_messages_if_needed(messages, steps, max_context_chars)
            continue

        action = parsed["action"]
        if action == "plan":
            plan = _normalize_plan(parsed)
            step = ToolLoopStep(
                index=index,
                action="plan",
                raw=raw,
                reason=str(parsed.get("reason", "")),
                plan=plan,
            )
            steps.append(step)
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Plan accepté. Tu peux maintenant utiliser les outils. "
                            "Si tu écris un fichier, relis-le ou lance une validation "
                            "avant la réponse finale."
                        ),
                    },
                ]
            )
            messages = _compact_messages_if_needed(messages, steps, max_context_chars)
            continue

        if action == "answer":
            unfinished = _invalid_final_answer(
                parsed,
                steps,
                require_artifact_audit=require_artifact_audit,
            )
            if unfinished:
                step = ToolLoopStep(
                    index=index,
                    action="invalid",
                    raw=raw,
                    error=unfinished,
                )
                steps.append(step)
                messages.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": _retry_observation([unfinished])},
                    ]
                )
                messages = _compact_messages_if_needed(messages, steps, max_context_chars)
                continue
            steps.append(
                ToolLoopStep(
                    index=index,
                    action="answer",
                    raw=raw,
                    reason=str(parsed.get("reason", "")),
                )
            )
            return ToolLoopResult(
                answer=str(parsed.get("content", "")),
                memory_after=str(parsed.get("memory_after", "")),
                steps=steps,
                stopped_reason="answer",
            )

        if action != "tool_call":
            step = ToolLoopStep(
                index=index,
                action=action,
                raw=raw,
                error=f"unsupported action: {action}",
            )
            steps.append(step)
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": _retry_observation([step.error]),
                    },
                ]
            )
            messages = _compact_messages_if_needed(messages, steps, max_context_chars)
            continue

        tool = str(parsed.get("tool", ""))
        args = parsed.get("args", {})
        if not isinstance(args, dict):
            args = {}
        if tool == "artifact_audit" and not str(args.get("prompt", "")).strip():
            args = {**args, "prompt": prompt}
        tool_error = _validate_tool_call(tool, args, steps)
        if tool_error:
            step = ToolLoopStep(
                index=index,
                action="invalid",
                raw=raw,
                tool=tool,
                args=args,
                reason=str(parsed.get("reason", "")),
                error=tool_error,
            )
            steps.append(step)
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": _retry_observation([tool_error])},
                ]
            )
            messages = _compact_messages_if_needed(messages, steps, max_context_chars)
            continue
        result = tools.call(tool, args)
        observation = _tool_observation(result, tools.specs())
        steps.append(
            ToolLoopStep(
                index=index,
                action="tool_call",
                raw=raw,
                tool=tool,
                args=args,
                reason=str(parsed.get("reason", "")),
                observation=observation,
                error=result.error or "",
            )
        )
        messages.extend(
            [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Observation outil. Utilise cette observation pour décider "
                        "la prochaine action. Réponds encore uniquement en JSON.\n"
                        f"{json.dumps(observation, ensure_ascii=False)}"
                    ),
                },
            ]
        )
        messages = _compact_messages_if_needed(messages, steps, max_context_chars)

    return ToolLoopResult(
        answer="",
        memory_after=memory,
        steps=steps,
        stopped_reason="max_steps",
    )


def parse_tool_action(
    raw: str,
    known_tools: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    result = parse_validated(
        raw,
        {
            "action": str,
            "tool": FieldSpec(str, required=False, default=""),
            "args": FieldSpec(dict, required=False, default={}),
            "reason": FieldSpec(str, required=False, default=""),
            "content": FieldSpec(str, required=False, default=""),
            "memory_after": FieldSpec(str, required=False, default=""),
            "files": FieldSpec(list, required=False, default=[]),
            "acceptance_criteria": FieldSpec(list, required=False, default=[]),
            "steps": FieldSpec(list, required=False, default=[]),
        },
    )
    if not result.ok or result.parsed is None:
        return {}, result.errors

    action = str(result.parsed.get("action", ""))
    if known_tools and action in set(known_tools):
        result.parsed["tool"] = action
        result.parsed["action"] = "tool_call"
        action = "tool_call"
    errors: list[str] = []
    if action not in {"plan", "tool_call", "answer"}:
        errors.append("action must be plan, tool_call or answer")
    if action == "tool_call" and not result.parsed.get("tool"):
        errors.append("tool_call requires tool")
    if action == "answer" and not result.parsed.get("content"):
        errors.append("answer requires content")
    if action == "plan":
        if not result.parsed.get("acceptance_criteria"):
            errors.append("plan requires acceptance_criteria")
        if not result.parsed.get("steps"):
            errors.append("plan requires steps")
    return result.parsed, errors


def _invalid_final_answer(
    parsed: dict[str, Any],
    steps: list[ToolLoopStep],
    require_artifact_audit: bool,
) -> str:
    content = str(parsed.get("content", "")).lower()
    if require_artifact_audit and not _has_artifact_audit(steps):
        return (
            "answer is premature. This task appears to produce HTML/CSS/JS or an "
            "interactive artifact, but artifact_audit has not been called. Call "
            "artifact_audit with the user intent before final answer."
        )
    failed_audit = _last_artifact_audit_failed(steps)
    if failed_audit and not _is_blocked_answer(content):
        return (
            "answer is premature. artifact_audit still reports blocking issues. "
            "Fix the files and call artifact_audit again, or answer explicitly with "
            "a blocked/non validated status if no useful tool action remains."
        )
    if _looks_like_todo_answer(content) and not _is_blocked_answer(content):
        return (
            "answer is premature. The answer lists remaining work instead of doing it. "
            "Continue with tool_call actions while budget remains, or answer explicitly "
            "with a blocked/non validated status."
        )
    if steps and any(step.error for step in steps):
        unfinished_markers = [
            "je vais",
            "je réessaie",
            "réessayer",
            "reessayer",
            "corriger cela",
            "try again",
            "will retry",
            "going to",
        ]
        if any(marker in content for marker in unfinished_markers):
            return (
                "answer must be final. A tool failed and the answer only announces "
                "a future retry; call a tool again with corrected args, or answer "
                "with a concrete blocked status and what remains undone."
            )
    if _claims_patch_applied_after_dry_run(content, steps):
        return (
            "answer claims files were applied/created, but apply_patch only ran "
            "with dry_run=true. Call apply_patch again with dry_run=false before "
            "claiming the files exist."
        )
    if _has_unverified_write(steps):
        return (
            "answer is premature. Files were written after the last validation. "
            "Call read_file on changed files or run_tests before final answer."
        )
    return ""


def _validate_tool_call(tool: str, args: dict[str, Any], steps: list[ToolLoopStep]) -> str:
    if tool in {"write_file", "apply_patch", "delete_file"} and not _has_plan(steps):
        return (
            "write/delete tools require an accepted plan first. Reply with action=plan "
            "including files, steps and acceptance_criteria before changing files."
        )
    repeated = _repeated_tool_call_error(tool, args, steps)
    if repeated:
        return repeated
    return ""


def _has_plan(steps: list[ToolLoopStep]) -> bool:
    return any(step.action == "plan" for step in steps)


def _has_artifact_audit(steps: list[ToolLoopStep]) -> bool:
    return any(step.action == "tool_call" and step.tool == "artifact_audit" for step in steps)


def _last_artifact_audit_failed(steps: list[ToolLoopStep]) -> bool:
    for step in reversed(steps):
        if step.action == "tool_call" and step.tool == "artifact_audit":
            return not bool(step.observation.get("ok"))
    return False


def _is_blocked_answer(content: str) -> bool:
    blocked_markers = [
        "bloqué",
        "bloque",
        "blocked",
        "non valid",
        "invalide",
        "impossible",
        "budget",
        "audit signale",
        "audit artefacts signale",
    ]
    return any(marker in content for marker in blocked_markers)


def _looks_like_todo_answer(content: str) -> bool:
    todo_markers = [
        "prochaines étapes",
        "prochaines etapes",
        "étapes restantes",
        "etapes restantes",
        "tâches restantes",
        "taches restantes",
        "reste à",
        "reste a",
        "il reste",
        "il manque",
        "doit être ajouté",
        "doit etre ajoute",
        "à faire",
        "a faire",
        "todo",
        "peut être amélioré",
        "peut etre ameliore",
    ]
    return any(marker in content for marker in todo_markers)


def _repeated_tool_call_error(
    tool: str,
    args: dict[str, Any],
    steps: list[ToolLoopStep],
    max_repeats: int = 3,
) -> str:
    args_key = json.dumps(args, ensure_ascii=False, sort_keys=True)
    repeats = 0
    for step in reversed(steps):
        if step.action == "plan":
            break
        step_args_key = json.dumps(step.args, ensure_ascii=False, sort_keys=True)
        if step.action == "tool_call" and step.tool == tool and step_args_key == args_key:
            repeats += 1
        if repeats >= max_repeats:
            return (
                f"tool {tool} was already called {repeats} times in this task. "
                "Use the observations already available and move to a different "
                "tool/action instead of repeating it."
            )
    return ""


def _has_unverified_write(steps: list[ToolLoopStep]) -> bool:
    last_write = -1
    last_validation = -1
    for index, step in enumerate(steps):
        if step.action != "tool_call":
            continue
        if step.tool == "artifact_audit":
            last_validation = index
            continue
        if not step.observation.get("ok"):
            continue
        if step.tool in {"write_file", "apply_patch", "delete_file"}:
            data = step.observation.get("data", {})
            if step.tool == "apply_patch" and data.get("dry_run") is True:
                continue
            last_write = index
        elif step.tool in {"repo_graph", "read_file", "read_selected_context", "run_tests"}:
            last_validation = index
    return last_write >= 0 and last_validation < last_write


def _normalize_plan(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "files": [str(item) for item in parsed.get("files", [])],
        "steps": [str(item) for item in parsed.get("steps", [])],
        "acceptance_criteria": [
            str(item) for item in parsed.get("acceptance_criteria", [])
        ],
    }


def _claims_patch_applied_after_dry_run(content: str, steps: list[ToolLoopStep]) -> bool:
    applied_markers = [
        "appliqué",
        "applique",
        "créé",
        "cree",
        "créés",
        "created",
        "applied",
        "files were created",
        "fichiers créés",
    ]
    if not any(marker in content for marker in applied_markers):
        return False

    saw_dry_run_patch = False
    saw_real_patch = False
    for step in steps:
        if step.tool != "apply_patch" or not step.observation.get("ok"):
            continue
        data = step.observation.get("data", {})
        if data.get("dry_run") is True:
            saw_dry_run_patch = True
        elif data.get("dry_run") is False:
            saw_real_patch = True
    return saw_dry_run_patch and not saw_real_patch


def _initial_messages(
    prompt: str,
    specs: list[ToolSpec],
    memory: str,
) -> list[Message]:
    return [
        {
            "role": "system",
            "content": (
                "Tu pilotes une boucle d'outils locale. Réponds uniquement en JSON.\n"
                '{"action":"plan","files":["..."],"steps":["..."],'
                '"acceptance_criteria":["..."]}\n'
                '{"action":"tool_call","tool":"nom_outil","args":{},"reason":"..."}\n'
                '{"action":"answer","content":"...","memory_after":"..."}\n'
                "Règles: plan avant écriture/suppression; après changement de fichier, "
                "relis ou valide avant answer; "
                "après erreur outil, corrige avec un autre tool_call. Si tu dois écrire, "
                "ta première action doit être plan. Pour HTML/CSS/JS complets, préfère "
                "write_file à apply_patch. Utilise delete_file pour nettoyer un fichier "
                "généré mais orphelin. Pour une page ou application HTML/CSS/JS, "
                "appelle artifact_audit avant answer. N'appelle jamais un fichier ou une fonction "
                "comme si c'était un outil. run_tests lance pytest : ne l'utilise pas "
                "sur des fichiers .html/.css/.js."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Objectif utilisateur :\n{prompt}\n\n"
                f"Mémoire compacte :\n{memory or '[vide]'}\n\n"
                f"Outils disponibles :\n{_render_tool_specs(specs)}"
            ),
        },
    ]


def _render_tool_specs(specs: list[ToolSpec]) -> str:
    rows = []
    for spec in specs:
        args = ", ".join(spec.args_schema.keys())
        rows.append(f"- {spec.name}({args})")
    return "\n".join(rows)


def _compact_messages_if_needed(
    messages: list[Message],
    steps: list[ToolLoopStep],
    max_context_chars: int,
) -> list[Message]:
    if _messages_chars(messages) <= max_context_chars:
        return messages
    if len(messages) < 2:
        return messages
    compact_user = (
        _truncate_middle(messages[1]["content"], max_chars=max_context_chars // 2)
        + "\n\nHistorique compact des actions déjà réalisées :\n"
        + _render_compact_step_history(steps)
        + "\n\nContinue depuis cet état. Réponds uniquement en JSON avec la prochaine action."
    )
    return [messages[0], {"role": "user", "content": compact_user}]


def _messages_chars(messages: list[Message]) -> int:
    return sum(len(message.get("content", "")) for message in messages)


def _render_compact_step_history(steps: list[ToolLoopStep], max_steps: int = 18) -> str:
    selected = steps[-max_steps:]
    lines = []
    for step in selected:
        if step.action == "plan":
            files = ", ".join(step.plan.get("files", []))
            lines.append(f"- {step.index}: plan files=[{files}]")
        elif step.action == "tool_call":
            status = "ok" if step.observation.get("ok") else "error"
            summary = step.observation.get("summary") or step.error or step.observation.get("error")
            lines.append(
                f"- {step.index}: tool {step.tool} {status} "
                f"{_truncate_text(str(summary or ''), 220)}"
            )
        elif step.action == "invalid":
            lines.append(f"- {step.index}: invalid {_truncate_text(step.error, 220)}")
        elif step.action == "answer":
            lines.append(f"- {step.index}: answer")
        else:
            lines.append(f"- {step.index}: {step.action}")
    return "\n".join(lines) if lines else "- aucune action précédente"


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head - 30
    return text[:head].rstrip() + "\n[...contexte compacté...]\n" + text[-tail:].lstrip()


def _truncate_text(text: str, max_chars: int) -> str:
    single_line = " ".join(text.split())
    if len(single_line) <= max_chars:
        return single_line
    return single_line[: max_chars - 3].rstrip() + "..."


def _retry_observation(errors: list[str]) -> str:
    return (
        "Action JSON invalide. Corrige au prochain tour.\n"
        f"Erreurs : {json.dumps(errors, ensure_ascii=False)}\n"
        "Ne réponds pas que tu vas réessayer : réessaie réellement avec tool_call "
        "si une action outil est nécessaire.\n"
        "Format attendu : "
        '{"action":"plan","files":["..."],"steps":["..."],'
        '"acceptance_criteria":["..."]} '
        '{"action":"tool_call","tool":"...","args":{},"reason":"..."} '
        'ou {"action":"answer","content":"...","memory_after":"..."}'
    )


def _tool_observation(result: ToolResult, specs: list[ToolSpec]) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "tool": result.tool,
        "ok": result.ok,
        "summary": result.summary,
        "error": result.error,
    }
    if not result.ok and result.error and result.error.startswith("unknown tool:"):
        observation["hint"] = "Utilise uniquement un nom présent dans available_tools."
        observation["available_tools"] = [spec.name for spec in specs]
        observation["examples"] = [
            {"tool": "write_file", "args": {"path": "index.html", "content": "...", "overwrite": True}},
            {"tool": "read_file", "args": {"path": "index.html"}},
            {"tool": "artifact_audit", "args": {"prompt": "intention utilisateur"}},
        ]
    if result.tool == "repo_graph":
        observation["data"] = {
            "root": result.data.get("root"),
            "prompt_json": result.data.get("prompt_json"),
            "total_files": result.data.get("total_files"),
            "total_directories": result.data.get("total_directories"),
        }
    elif result.tool == "read_selected_context":
        observation["data"] = {
            "root": result.data.get("root"),
            "files": result.data.get("files", []),
            "rendered": result.data.get("rendered", ""),
        }
    elif result.tool == "apply_patch":
        observation["data"] = result.data
        if not result.ok:
            observation["hint"] = (
                "Le patch doit utiliser ce format, pas un unified diff : "
                "*** Begin Patch\\n*** Add File: chemin/relatif.ext\\n+ligne\\n"
                "*** End Patch ou *** Update File avec @@ puis lignes préfixées "
                "par espace, + ou -. Il ne doit y avoir qu'un seul *** End Patch, "
                "tout à la fin."
            )
        elif result.data.get("dry_run") is True:
            observation["hint"] = (
                "Dry-run validé seulement : aucun fichier n'a été écrit. "
                "Pour appliquer réellement, rappelle apply_patch avec le même patch "
                "et dry_run=false."
            )
    elif result.tool == "write_file":
        observation["data"] = result.data
    else:
        observation["data"] = result.data
    return observation
