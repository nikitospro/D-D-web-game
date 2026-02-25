import json
import os
import random
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlencode

import requests
from flask import Flask, redirect, render_template, request, session, url_for


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

SAVE_DIR = Path("saves")
SAVE_DIR.mkdir(exist_ok=True)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_TIMEOUT_SECONDS = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "240"))
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_AUTOSTART = os.environ.get("OLLAMA_AUTOSTART", "1") == "1"

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
COMFYUI_ENABLED = os.environ.get("COMFYUI_ENABLED", "1") == "1"
COMFYUI_TIMEOUT_SECONDS = int(os.environ.get("COMFYUI_TIMEOUT_SECONDS", "180"))
COMFYUI_MAX_WAIT_SECONDS = int(os.environ.get("COMFYUI_MAX_WAIT_SECONDS", "120"))
COMFYUI_POLL_INTERVAL_SECONDS = float(os.environ.get("COMFYUI_POLL_INTERVAL_SECONDS", "1.2"))
COMFYUI_WORKFLOW_PATH = os.environ.get("COMFYUI_WORKFLOW_PATH", "comfyui_workflow_api.json")
COMFYUI_CHECKPOINT = os.environ.get("COMFYUI_CHECKPOINT", "").strip()
COMFYUI_FALLBACK_URLS = os.environ.get("COMFYUI_FALLBACK_URLS", "").strip()
COMFYUI_NEGATIVE_PROMPT = os.environ.get(
    "COMFYUI_NEGATIVE_PROMPT",
    "low quality, blurry, watermark, text, signature, logo, distorted anatomy, extra limbs",
)

# Local Ollama calls must bypass system proxies, otherwise localhost requests
# may be routed through a corporate proxy and return 503.
OLLAMA_HTTP = requests.Session()
OLLAMA_HTTP.trust_env = False

COMFYUI_HTTP = requests.Session()
COMFYUI_HTTP.trust_env = False

_OLLAMA_AUTOSTART_LOCK = Lock()
_OLLAMA_LAST_AUTOSTART_TS = 0.0
_COMFYUI_RESOLVE_LOCK = Lock()
_COMFYUI_LAST_RESOLVE_TS = 0.0
_COMFYUI_RESOLVED_URL = COMFYUI_URL

DEFAULT_STARTER_ITEMS = [
    "Нож",
    "Зелье здоровья",
    "Меч",
    "Короткий меч",
    "Кожаная броня",
    "Лук",
    "Факел",
    "Веревка",
]

STARTER_ITEMS_SESSION_KEY = "starter_items"
GAME_DEBUG_DEFAULT = os.environ.get("GAME_DEBUG_DEFAULT", "0") == "1"
GAME_DEBUG_MAX_LOGS = int(os.environ.get("GAME_DEBUG_MAX_LOGS", "160"))
GAME_DEBUG_EASTER_CHANCE = float(os.environ.get("GAME_DEBUG_EASTER_CHANCE", "0.12"))

DEBUG_EASTER_EGGS = [
    "гавно какое то",
    "^q^",
    "пупупу",
    "чем гуще лез if else if else",
    "https://open.spotify.com/track/0Hnik88ryug02UaEOTiXrE?si=102bb4929fe84ff4",
    "https://open.spotify.com/track/36TB6pTzZq0qnFOlFbxbZH?si=a5cd176585a641ec",
]


@dataclass
class Character:
    name: str
    race: str
    character_class: str
    appearance: str
    age: str
    gender: str
    hp: int = 20
    inventory: list[str] = field(default_factory=list)


@dataclass
class Scene:
    description: str
    options: list[str]


@dataclass
class GameState:
    save_name: str
    world_scenario: str
    main_goal: str
    character: Character
    history: list[dict[str, Any]] = field(default_factory=list)
    debug_enabled: bool = GAME_DEBUG_DEFAULT
    debug_logs: list[dict[str, Any]] = field(default_factory=list)
    current_scene: Scene | None = None
    current_scene_image_url: str | None = None
    game_over: bool = False


def save_path(save_name: str) -> Path:
    # Keep unicode letters/digits to avoid collisions for Cyrillic names.
    safe = re.sub(r"[^\w-]+", "_", save_name, flags=re.UNICODE).strip("_")
    safe = safe or "save"
    return SAVE_DIR / f"{safe}.json"


def save_game(state: GameState) -> None:
    raw = asdict(state)
    with save_path(state.save_name).open("w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)


def find_save_file(save_name: str) -> Path | None:
    direct_path = save_path(save_name)
    if direct_path.exists():
        return direct_path
    for path in SAVE_DIR.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if str(raw.get("save_name", "")).strip() == save_name:
                return path
        except Exception:
            continue
    return None


def load_game(save_name: str) -> GameState | None:
    path = find_save_file(save_name)
    if not path:
        return None
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    character = Character(**raw["character"])
    scene = None
    if raw.get("current_scene"):
        scene = Scene(**raw["current_scene"])
    return GameState(
        save_name=raw["save_name"],
        world_scenario=raw["world_scenario"],
        main_goal=raw["main_goal"],
        character=character,
        history=raw.get("history", []),
        debug_enabled=raw.get("debug_enabled", GAME_DEBUG_DEFAULT),
        debug_logs=raw.get("debug_logs", []),
        current_scene=scene,
        current_scene_image_url=raw.get("current_scene_image_url"),
        game_over=raw.get("game_over", False),
    )


def list_available_saves() -> list[dict[str, Any]]:
    saves = []
    for path in SAVE_DIR.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            character = raw.get("character", {})
            stat = path.stat()
            saves.append(
                {
                    "save_name": raw.get("save_name", path.stem),
                    "character_name": character.get("name") or "Персонаж не создан",
                    "hp": character.get("hp", 20),
                    "updated_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                }
            )
        except Exception:
            continue
    saves.sort(key=lambda s: s["updated_at"], reverse=True)
    return saves


def _debug_ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def add_debug_log(
    state: GameState | None,
    message: str,
    *,
    level: str = "info",
    add_easter: bool = False,
) -> None:
    if state is None or not state.debug_enabled:
        return

    entry = {
        "ts": _debug_ts(),
        "level": level,
        "message": str(message).strip()[:320],
    }
    state.debug_logs.append(entry)

    if add_easter and random.random() < GAME_DEBUG_EASTER_CHANCE:
        state.debug_logs.append(
            {
                "ts": _debug_ts(),
                "level": "easter",
                "message": random.choice(DEBUG_EASTER_EGGS),
            }
        )

    if len(state.debug_logs) > GAME_DEBUG_MAX_LOGS:
        state.debug_logs = state.debug_logs[-GAME_DEBUG_MAX_LOGS:]


def clean_ollama_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    return cleaned


def extract_json_payload(text: str) -> Any | None:
    if not text:
        return None
    decoder = json.JSONDecoder()
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except Exception:
        pass

    for idx, char in enumerate(stripped):
        if char not in "{[":
            continue
        try:
            payload, _ = decoder.raw_decode(stripped[idx:])
            return payload
        except Exception:
            continue
    return None


def _is_connection_refused(exc: requests.RequestException) -> bool:
    text = str(exc)
    return "WinError 10061" in text or "Connection refused" in text or "Failed to establish a new connection" in text


def _ollama_healthcheck(timeout_seconds: int = 2) -> bool:
    try:
        response = OLLAMA_HTTP.get(f"{OLLAMA_URL}/api/tags", timeout=timeout_seconds)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _ensure_ollama_running() -> bool:
    global _OLLAMA_LAST_AUTOSTART_TS

    if not OLLAMA_AUTOSTART:
        return False

    if _ollama_healthcheck():
        return True

    with _OLLAMA_AUTOSTART_LOCK:
        # Another request may have started Ollama already.
        if _ollama_healthcheck():
            return True

        now = time.time()
        if now - _OLLAMA_LAST_AUTOSTART_TS < 12:
            return False
        _OLLAMA_LAST_AUTOSTART_TS = now

        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            app.logger.warning("Ollama was not reachable. Started `ollama serve` automatically.")
        except FileNotFoundError:
            app.logger.warning("Cannot autostart Ollama: `ollama` executable not found in PATH.")
            return False
        except Exception as exc:
            app.logger.warning("Cannot autostart Ollama: %s", exc)
            return False

    # Give the server a short warm-up window.
    for _ in range(12):
        if _ollama_healthcheck():
            app.logger.warning("Ollama API is reachable after autostart.")
            return True
        time.sleep(1)
    return False


def call_ollama(prompt: str, *, json_mode: bool = False, timeout_seconds: int | None = None) -> str:
    timeout = timeout_seconds or OLLAMA_TIMEOUT_SECONDS
    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    if json_mode:
        payload["format"] = "json"

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            response = OLLAMA_HTTP.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            text = clean_ollama_text(data.get("response", ""))
            if not text:
                app.logger.warning("Ollama returned an empty response.")
            return text
        except requests.Timeout:
            app.logger.warning(
                "Ollama request timed out after %s seconds (attempt %s/%s).",
                timeout,
                attempt,
                max_attempts,
            )
        except requests.RequestException as exc:
            status = exc.response.status_code if exc.response is not None else "n/a"
            details = ""
            if exc.response is not None:
                details = (exc.response.text or "").strip().replace("\n", " ")
                if len(details) > 220:
                    details = details[:220] + "..."

            should_retry_after_autostart = exc.response is None and _is_connection_refused(exc)
            app.logger.warning(
                "Ollama request failed (status=%s, attempt %s/%s): %s %s",
                status,
                attempt,
                max_attempts,
                exc,
                details,
            )
            if should_retry_after_autostart and _ensure_ollama_running():
                # API became reachable, retry immediately on next loop step.
                continue
        except Exception as exc:
            app.logger.warning(
                "Unexpected Ollama error (attempt %s/%s): %s",
                attempt,
                max_attempts,
                exc,
            )

        if attempt < max_attempts:
            time.sleep(attempt)

    return ""


def comfyui_candidate_urls() -> list[str]:
    urls: list[str] = [COMFYUI_URL]
    if COMFYUI_FALLBACK_URLS:
        for raw in COMFYUI_FALLBACK_URLS.split(","):
            candidate = raw.strip().rstrip("/")
            if candidate:
                urls.append(candidate)

    # Common ComfyUI ports: standard API (8188), desktop app defaults often use 8000.
    urls.extend(
        [
            "http://127.0.0.1:8188",
            "http://127.0.0.1:8000",
            "http://localhost:8188",
            "http://localhost:8000",
        ]
    )

    result: list[str] = []
    seen: set[str] = set()
    for url in urls:
        key = url.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(url)
    return result


def comfyui_healthcheck(base_url: str | None = None, timeout_seconds: int = 3) -> bool:
    if not COMFYUI_ENABLED:
        return False

    target = (base_url or _COMFYUI_RESOLVED_URL).rstrip("/")
    for endpoint in ("/system_stats", "/queue", "/object_info"):
        try:
            response = COMFYUI_HTTP.get(f"{target}{endpoint}", timeout=timeout_seconds)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            continue
    return False


def resolve_comfyui_url(force_refresh: bool = False) -> str | None:
    global _COMFYUI_LAST_RESOLVE_TS, _COMFYUI_RESOLVED_URL

    if not COMFYUI_ENABLED:
        return None

    now = time.time()
    if not force_refresh and now - _COMFYUI_LAST_RESOLVE_TS < 10 and comfyui_healthcheck(_COMFYUI_RESOLVED_URL):
        return _COMFYUI_RESOLVED_URL

    with _COMFYUI_RESOLVE_LOCK:
        now = time.time()
        if not force_refresh and now - _COMFYUI_LAST_RESOLVE_TS < 10 and comfyui_healthcheck(_COMFYUI_RESOLVED_URL):
            return _COMFYUI_RESOLVED_URL

        for candidate in comfyui_candidate_urls():
            if comfyui_healthcheck(candidate):
                if candidate != _COMFYUI_RESOLVED_URL:
                    app.logger.warning(
                        "ComfyUI URL switched from %s to %s",
                        _COMFYUI_RESOLVED_URL,
                        candidate,
                    )
                _COMFYUI_RESOLVED_URL = candidate
                _COMFYUI_LAST_RESOLVE_TS = time.time()
                return candidate

        _COMFYUI_LAST_RESOLVE_TS = time.time()
        return None


def comfyui_get_checkpoint_candidates() -> list[str]:
    base_url = resolve_comfyui_url()
    if not base_url:
        return []

    candidates: list[str] = []

    for endpoint in ("/models/checkpoints", "/api/models/checkpoints"):
        try:
            response = COMFYUI_HTTP.get(f"{base_url}{endpoint}", timeout=4)
            response.raise_for_status()
            payload = response.json()
            values = payload if isinstance(payload, list) else payload.get("checkpoints", [])
            for item in values:
                if isinstance(item, str):
                    name = item.strip()
                elif isinstance(item, dict):
                    name = str(item.get("name", item.get("filename", ""))).strip()
                else:
                    name = ""
                if name:
                    candidates.append(name)
            if candidates:
                break
        except Exception:
            continue

    if not candidates:
        try:
            response = COMFYUI_HTTP.get(f"{base_url}/object_info/CheckpointLoaderSimple", timeout=4)
            response.raise_for_status()
            payload = response.json()
            options = (
                payload.get("CheckpointLoaderSimple", {})
                .get("input", {})
                .get("required", {})
                .get("ckpt_name", [[]])[0]
            )
            if isinstance(options, list):
                for option in options:
                    name = str(option).strip()
                    if name:
                        candidates.append(name)
        except Exception:
            pass

    uniq: list[str] = []
    seen: set[str] = set()
    for name in candidates:
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(name)
    return uniq


def comfyui_pick_checkpoint() -> str:
    if COMFYUI_CHECKPOINT:
        return COMFYUI_CHECKPOINT

    candidates = comfyui_get_checkpoint_candidates()
    if candidates:
        return candidates[0]

    # Fallback default for standard SD1.5 setups.
    return "v1-5-pruned-emaonly.safetensors"


def load_comfyui_workflow_template() -> dict[str, Any] | None:
    path = Path(COMFYUI_WORKFLOW_PATH)
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            if "prompt" in payload and isinstance(payload["prompt"], dict):
                return payload["prompt"]
            if payload and all(isinstance(v, dict) for v in payload.values()):
                return payload
    except Exception as exc:
        app.logger.warning("Cannot load ComfyUI workflow from %s: %s", path, exc)
    return None


def comfyui_default_workflow() -> dict[str, Any]:
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 1,
                "steps": 24,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": comfyui_pick_checkpoint(),
            },
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 896,
                "height": 512,
                "batch_size": 1,
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "",
                "clip": ["4", 1],
            },
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": COMFYUI_NEGATIVE_PROMPT,
                "clip": ["4", 1],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2],
            },
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "dnd_scene",
                "images": ["8", 0],
            },
        },
    }


def apply_comfyui_runtime_values(
    workflow: dict[str, Any],
    *,
    positive_prompt: str,
    negative_prompt: str,
    seed: int,
    checkpoint_name: str,
    filename_prefix: str,
) -> dict[str, Any]:
    graph = json.loads(json.dumps(workflow))
    clip_nodes: list[tuple[str, dict[str, Any]]] = []

    for node_id, node in graph.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type", ""))
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            inputs = {}
            node["inputs"] = inputs

        if class_type == "CLIPTextEncode":
            clip_nodes.append((str(node_id), inputs))
        elif class_type == "KSampler":
            inputs["seed"] = int(seed)
        elif class_type == "CheckpointLoaderSimple" and checkpoint_name:
            inputs["ckpt_name"] = checkpoint_name
        elif class_type == "SaveImage":
            inputs["filename_prefix"] = filename_prefix

    def _node_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
        node_name = item[0]
        if node_name.isdigit():
            return (0, f"{int(node_name):08d}")
        return (1, node_name)

    clip_nodes.sort(key=_node_sort_key)
    if clip_nodes:
        clip_nodes[0][1]["text"] = positive_prompt
    if len(clip_nodes) > 1:
        clip_nodes[1][1]["text"] = negative_prompt

    return graph


def extract_image_url_from_history(history_payload: Any, prompt_id: str, base_url: str) -> str | None:
    if not isinstance(history_payload, dict):
        return None

    entry = None
    if prompt_id in history_payload and isinstance(history_payload[prompt_id], dict):
        entry = history_payload[prompt_id]
    elif "outputs" in history_payload:
        entry = history_payload

    if not isinstance(entry, dict):
        return None
    outputs = entry.get("outputs", {})
    if not isinstance(outputs, dict):
        return None

    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        images = node_output.get("images", [])
        if not isinstance(images, list):
            continue
        for image in images:
            if not isinstance(image, dict):
                continue
            filename = str(image.get("filename", "")).strip()
            if not filename:
                continue
            query = urlencode(
                {
                    "filename": filename,
                    "subfolder": str(image.get("subfolder", "")),
                    "type": str(image.get("type", "output")),
                }
            )
            return f"{base_url}/view?{query}"
    return None


def build_scene_image_prompt(state: GameState) -> str:
    if not state.current_scene:
        return ""
    char_line = (
        f"{state.character.name}, {state.character.race}, {state.character.character_class}, "
        f"{state.character.appearance}"
    )
    return (
        "dark fantasy role-playing game scene, cinematic concept art, dramatic lighting, "
        "detailed environment, atmospheric composition, no text. "
        f"World context: {state.world_scenario}. "
        f"Main goal context: {state.main_goal}. "
        f"Character: {char_line}. "
        f"Scene details: {state.current_scene.description}"
    )[:1200]


def generate_scene_image(state: GameState) -> str | None:
    if not COMFYUI_ENABLED or not state.current_scene:
        return None

    add_debug_log(state, "Генерация изображения сцены: старт", level="system")
    base_url = resolve_comfyui_url(force_refresh=True)
    if not base_url:
        app.logger.warning(
            "ComfyUI is not reachable. Tried URLs: %s. Scene image skipped.",
            ", ".join(comfyui_candidate_urls()),
        )
        add_debug_log(state, "ComfyUI недоступен: изображение пропущено", level="warn")
        return None

    positive_prompt = build_scene_image_prompt(state)
    if not positive_prompt:
        add_debug_log(state, "Пустой prompt для изображения, пропуск", level="warn")
        return None

    checkpoint_name = comfyui_pick_checkpoint()
    safe_save = re.sub(r"[^\w-]+", "_", state.save_name, flags=re.UNICODE).strip("_") or "save"
    filename_prefix = f"dnd_scene_{safe_save}"
    seed = random.randint(1, 2**63 - 1)

    template = load_comfyui_workflow_template() or comfyui_default_workflow()
    workflow = apply_comfyui_runtime_values(
        template,
        positive_prompt=positive_prompt,
        negative_prompt=COMFYUI_NEGATIVE_PROMPT,
        seed=seed,
        checkpoint_name=checkpoint_name,
        filename_prefix=filename_prefix,
    )

    client_id = f"dnd-{random.randint(100000, 999999)}"
    try:
        response = COMFYUI_HTTP.post(
            f"{base_url}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=COMFYUI_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        prompt_id = str(payload.get("prompt_id", "")).strip()
        if not prompt_id:
            app.logger.warning("ComfyUI returned no prompt_id.")
            add_debug_log(state, "ComfyUI не вернул prompt_id", level="warn")
            return None
    except Exception as exc:
        app.logger.warning("ComfyUI prompt request failed: %s", exc)
        add_debug_log(state, f"Ошибка запроса к ComfyUI: {exc}", level="error")
        return None

    deadline = time.time() + COMFYUI_MAX_WAIT_SECONDS
    while time.time() < deadline:
        try:
            history_response = COMFYUI_HTTP.get(
                f"{base_url}/history/{prompt_id}",
                timeout=min(15, COMFYUI_TIMEOUT_SECONDS),
            )
            if history_response.status_code == 200:
                image_url = extract_image_url_from_history(history_response.json(), prompt_id, base_url)
                if image_url:
                    add_debug_log(state, "Изображение сцены успешно сгенерировано", level="ok", add_easter=True)
                    return image_url
        except Exception:
            pass
        time.sleep(COMFYUI_POLL_INTERVAL_SECONDS)

    app.logger.warning("ComfyUI image wait timed out for prompt_id=%s", prompt_id)
    add_debug_log(state, "Таймаут ожидания изображения ComfyUI", level="warn")
    return None


def scene_signature(text: str) -> str:
    normalized = re.sub(r"[^\w\s]+", " ", text.lower(), flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized[:220]


def recent_scene_signatures(state: GameState, limit: int = 12) -> set[str]:
    signatures: list[str] = []
    if state.current_scene and state.current_scene.description:
        signatures.append(scene_signature(state.current_scene.description))

    for turn in state.history[-limit:]:
        sig = turn.get("scene_signature")
        if isinstance(sig, str) and sig.strip():
            signatures.append(sig.strip())
    return set(signatures)


def fallback_scene(state: GameState, used_signatures: set[str] | None = None) -> Scene:
    used_signatures = used_signatures or set()

    locations = [
        "на заброшенном тракте у разрушенного моста",
        "в сыром тоннеле под старым кварталом",
        "на площади вымершей деревни",
        "в склепе под обвалившейся часовней",
        "у пристани, где гниют пустые барки",
        "на лестнице к башне наблюдателей",
        "в разбитом лагере на краю болота",
        "в узком проходе между каменными статуями",
    ]
    weather = [
        "туман стелется по земле",
        "морось бьет в лицо и гасит факелы",
        "ветер свистит в щелях стен",
        "пепел кружится в воздухе",
        "далекая гроза подсвечивает небо",
        "морозный воздух режет горло",
    ]
    sounds = [
        "глухой стук металла",
        "торопливые шаги за спиной",
        "шорох ткани за колонной",
        "скрип цепей где-то впереди",
        "тихий шепот на незнакомом языке",
        "плеск воды в темноте",
    ]
    clues = [
        "на камне видна свежая кровь",
        "в пыли остаются новые следы сапог",
        "на двери выжжен странный знак",
        "под ногами хрустит сломанный амулет",
        "у стены лежит оброненная карта",
        "в трещине мерцает тусклый кристалл",
    ]
    threats = [
        "рядом прячется что-то крупное и голодное",
        "враг может устроить засаду в любой момент",
        "ловушки в этом месте почти незаметны",
        "кто-то уже наблюдает за тобой из тени",
        "один неверный шаг может обрушить проход",
        "магия вокруг нестабильна и опасна",
    ]
    option_templates = [
        "Проверить местность и {clue}",
        "Тихо обойти опасный участок справа",
        "Подготовить оружие и идти к источнику звука",
        "Осмотреть укрытие рядом и найти следы",
        "Использовать предмет из инвентаря для разведки",
        "Попробовать приманить врага на открытое место",
        "Отступить на шаг и оценить обстановку",
        "Идти вперед, прикрываясь тенями",
    ]

    for _ in range(18):
        place = random.choice(locations)
        meteo = random.choice(weather)
        sound = random.choice(sounds)
        clue = random.choice(clues)
        threat = random.choice(threats)
        world_hint = state.world_scenario.strip()[:80]
        goal_hint = state.main_goal.strip()[:80]

        description = (
            f"Ты оказываешься {place}, и {meteo}. "
            f"В этой части мира ощущается влияние твоего пути: {world_hint}. "
            f"В тишине резко выделяется {sound}, заставляя тебя замереть и прислушаться. "
            f"Рядом {clue}, и это явно связано с тем, что происходит вокруг. "
            f"Твоя главная цель все так же впереди: {goal_hint}. "
            f"Но {threat}, поэтому действовать нужно осторожно и быстро."
        )

        options = []
        seen_options: set[str] = set()
        shuffled_templates = option_templates[:]
        random.shuffle(shuffled_templates)
        for template in shuffled_templates:
            option = template.format(clue=clue)
            key = option.casefold()
            if key in seen_options:
                continue
            seen_options.add(key)
            options.append(option)
            if len(options) == 3:
                break

        signature = scene_signature(description)
        if signature not in used_signatures:
            return Scene(description=description, options=options)

    # Last resort: return one more generated scene even if duplicate.
    return Scene(
        description=(
            "Перед тобой еще один опасный участок пути. Воздух тяжелый, вокруг слишком тихо, "
            "и каждый звук кажется предупреждением. Следы недавнего движения указывают, что ты не один. "
            "Цель близка, но любая ошибка может стоить слишком дорого."
        ),
        options=[
            "Осторожно осмотреть окружение",
            "Продвинуться вперед короткими рывками",
            "Подготовить защиту и ждать движения врага",
        ],
    )


def parse_scene_response(text: str) -> Scene | None:
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    description_lines = []
    options = []
    for ln in lines:
        if re.match(r"^\d+[\).\s-]+", ln):
            option = re.sub(r"^\d+[\).\s-]+", "", ln).strip()
            if option:
                options.append(option)
        elif ln.lower().startswith(("вариант", "option")):
            part = ln.split(":", 1)
            if len(part) == 2 and part[1].strip():
                options.append(part[1].strip())
        else:
            description_lines.append(ln)
    if len(options) < 3:
        return None
    return Scene(description=" ".join(description_lines), options=options[:3])


def parse_starter_items_response(text: str) -> list[str]:
    if not text:
        return []

    cleaned_text = text.strip()
    cleaned_text = re.sub(r"```(?:json)?", "", cleaned_text, flags=re.IGNORECASE).strip()

    # Try JSON array first for strict responses.
    json_match = re.search(r"\[[\s\S]*\]", cleaned_text)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if isinstance(data, list):
                normalized = [str(item).strip() for item in data if str(item).strip()]
                if normalized:
                    cleaned_text = "\n".join(normalized)
        except Exception:
            pass

    lines = cleaned_text.splitlines()
    if len(lines) <= 2 and ("," in cleaned_text or ";" in cleaned_text):
        lines = re.split(r"[,\n;]+", cleaned_text)

    def normalize_item(raw_item: str) -> str:
        value = raw_item.strip()
        value = re.sub(r"^\d+\s*[\).\:\-]\s*", "", value)
        value = re.sub(r"^[-*•]+\s*", "", value)
        value = value.replace("**", "").replace("__", "").replace("`", "")
        for sep in (" — ", " - ", ": ", " ("):
            if sep in value:
                head = value.split(sep, 1)[0].strip()
                if head:
                    value = head
                    break
        value = value.strip(" \"'.,;:!?-")
        value = re.sub(r"\s+", " ", value)
        return value

    items: list[str] = []
    seen: set[str] = set()
    for raw_line in lines:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        parts = [raw_line]
        if "," in raw_line and len(raw_line) > 40:
            parts = [p.strip() for p in raw_line.split(",") if p.strip()]

        for part in parts:
            item = normalize_item(part)
            if not item:
                continue
            if len(item) < 2 or len(item) > 42:
                continue

            low = item.casefold()
            if low.startswith(("вот ", "список", "конечно", "ниже", "ответ")):
                continue
            if "предмет" in low and len(item.split()) > 3:
                continue

            if low in seen:
                continue
            seen.add(low)
            items.append(item)

    return items


def generate_starter_items(state: GameState) -> list[str]:
    add_debug_log(state, "Генерация стартовых предметов", level="system")
    prompt_json = f"""
Generate exactly 8 starting items for a fantasy RPG character.
Return valid JSON only in this shape:
{{"items":["item1","item2","item3","item4","item5","item6","item7","item8"]}}
All item names must be in Russian, short (1-3 words), and unique.
No explanations.

World: {state.world_scenario}
Main goal: {state.main_goal}
"""
    prompt_lines = f"""
Generate exactly 8 unique starting items for a fantasy RPG character.
Output must be in Russian. One item per line. No numbering. No explanations.
Each item should be short (1-3 words).

World: {state.world_scenario}
Main goal: {state.main_goal}
"""

    best_items: list[str] = []

    json_text = call_ollama(prompt_json, json_mode=True)
    json_payload = extract_json_payload(json_text)
    if isinstance(json_payload, dict) and isinstance(json_payload.get("items"), list):
        parsed_json_items = [str(x).strip() for x in json_payload["items"] if str(x).strip()]
        best_items = parse_starter_items_response("\n".join(parsed_json_items))
    elif isinstance(json_payload, list):
        parsed_json_items = [str(x).strip() for x in json_payload if str(x).strip()]
        best_items = parse_starter_items_response("\n".join(parsed_json_items))

    if len(best_items) < 6:
        line_text = call_ollama(prompt_lines)
        line_items = parse_starter_items_response(line_text)
        if len(line_items) > len(best_items):
            best_items = line_items

    if not best_items:
        app.logger.warning("Starter item generation failed. Using default item set.")
        add_debug_log(state, "Не удалось сгенерировать предметы, использован fallback", level="warn")
        return DEFAULT_STARTER_ITEMS.copy()

    # Keep first 8 generated items and backfill from defaults if needed.
    items = best_items[:8]
    existing = {item.casefold() for item in items}
    for default_item in DEFAULT_STARTER_ITEMS:
        if len(items) >= 8:
            break
        if default_item.casefold() in existing:
            continue
        items.append(default_item)
        existing.add(default_item.casefold())
    add_debug_log(state, f"Стартовые предметы готовы: {len(items)}", level="ok", add_easter=True)
    return items


def generate_scene(state: GameState) -> Scene:
    add_debug_log(state, "Генерация новой сцены", level="system")
    used_signatures = recent_scene_signatures(state, limit=14)
    recent_turns = state.history[-4:]
    history_text = "Нет предыдущих ходов."
    if recent_turns:
        history_lines = []
        for idx, turn in enumerate(recent_turns, start=1):
            result = "успех" if turn.get("success") else "провал"
            history_lines.append(f"{idx}) {turn.get('action', 'действие')} -> {result}")
        history_text = "\n".join(history_lines)

    recent_signatures_text = "\n".join(f"- {sig}" for sig in list(used_signatures)[:6]) or "- нет"

    def _valid_scene(candidate: Scene | None) -> Scene | None:
        if candidate is None:
            return None
        if len(candidate.options) < 3:
            return None
        sig = scene_signature(candidate.description)
        if sig in used_signatures:
            return None
        return Scene(description=candidate.description.strip(), options=[opt.strip() for opt in candidate.options[:3]])

    # First try: strict JSON format from model, multiple attempts with different seeds.
    for _ in range(4):
        variation_seed = random.randint(1000, 999999)
        prompt_json = f"""
You are an RPG narrator. Output language must be Russian.
Generate a new scene that is clearly different from prior scenes.
Return valid JSON only in this shape:
{{
  "description": "6-9 Russian sentences, rich atmosphere",
  "options": ["option 1", "option 2", "option 3"]
}}
Rules:
- Do not reuse previous scene wording.
- Each option is a short action in Russian, max 12 words.
- No extra keys, no markdown.

World: {state.world_scenario}
Main goal: {state.main_goal}
Character: {state.character.name}, {state.character.race}, {state.character.character_class}
Appearance: {state.character.appearance}
Inventory: {", ".join(state.character.inventory) if state.character.inventory else "пусто"}
Recent turns:
{history_text}
Previous scene signatures (avoid repeating):
{recent_signatures_text}
Variation seed: {variation_seed}
"""
        json_text = call_ollama(prompt_json, json_mode=True)
        payload = extract_json_payload(json_text)

        scene_candidate = None
        if isinstance(payload, dict):
            description = str(payload.get("description", "")).strip()
            raw_options = payload.get("options", [])
            if isinstance(raw_options, list):
                options = [str(x).strip() for x in raw_options if str(x).strip()]
                if description and len(options) >= 3:
                    scene_candidate = Scene(description=description, options=options[:3])

        validated = _valid_scene(scene_candidate)
        if validated is not None:
            add_debug_log(state, "Сцена сгенерирована (JSON режим)", level="ok", add_easter=True)
            return validated

    # Second try: plain text format from model, with anti-duplicate check.
    for _ in range(2):
        variation_seed = random.randint(1000, 999999)
        prompt_text = f"""
Ты — ведущий текстовой RPG. Пиши только на русском языке.
Сгенерируй НОВУЮ сцену, заметно отличающуюся от прошлых.
Формат строго:
Описание сцены (6-9 предложений)
1. Вариант 1
2. Вариант 2
3. Вариант 3

Мир: {state.world_scenario}
Цель: {state.main_goal}
Персонаж: {state.character.name}, {state.character.race}, {state.character.character_class}
Внешность: {state.character.appearance}
Инвентарь: {", ".join(state.character.inventory) if state.character.inventory else "пусто"}
Предыдущие ходы:
{history_text}
Не повторяй сцены с такими сигнатурами:
{recent_signatures_text}
Сид вариативности: {variation_seed}
"""
        model_text = call_ollama(prompt_text)
        validated = _valid_scene(parse_scene_response(model_text))
        if validated is not None:
            add_debug_log(state, "Сцена сгенерирована (текстовый режим)", level="ok", add_easter=True)
            return validated

    app.logger.warning("Scene generation failed or repeated prior scene. Using procedural fallback.")
    add_debug_log(state, "Сцена повторилась/невалидна, используется процедурный fallback", level="warn")
    return fallback_scene(state, used_signatures=used_signatures)


def get_state_from_session() -> GameState | None:
    save_name = session.get("save_name")
    if not save_name:
        return None
    return load_game(save_name)


def roll_d20() -> int:
    return random.randint(1, 20)


def calculate_difficulty(action_text: str) -> int:
    length = len(action_text.strip())
    if length <= 20:
        return 10
    if length <= 80:
        return 12
    return 14


@app.route("/", methods=["GET", "POST"])
def index():
    saves = list_available_saves()
    if request.method == "POST":
        save_name = request.form.get("save_name", "").strip()
        world_scenario = request.form.get("world_scenario", "").strip()
        main_goal = request.form.get("main_goal", "").strip()
        if not save_name or not world_scenario or not main_goal:
            return render_template(
                "index.html",
                saves=saves,
                error="Заполни все поля.",
            )
        if find_save_file(save_name):
            return render_template(
                "index.html",
                saves=saves,
                error="Сохранение с таким именем уже существует. Загрузи его из списка или выбери другое имя.",
            )
        session["save_name"] = save_name
        session.pop(STARTER_ITEMS_SESSION_KEY, None)
        state = GameState(
            save_name=save_name,
            world_scenario=world_scenario,
            main_goal=main_goal,
            character=Character(
                name="",
                race="",
                character_class="",
                appearance="",
                age="",
                gender="",
                inventory=[],
            ),
        )
        save_game(state)
        return redirect(url_for("character_create"))
    return render_template("index.html", saves=saves)


@app.route("/load", methods=["POST"])
def load_save():
    save_name = request.form.get("save_name", "").strip()
    state = load_game(save_name)
    if not state:
        return redirect(url_for("index"))
    session["save_name"] = state.save_name
    session.pop(STARTER_ITEMS_SESSION_KEY, None)
    if state.character.name:
        return redirect(url_for("game"))
    return redirect(url_for("character_create"))


@app.route("/delete", methods=["POST"])
def delete_save():
    save_name = request.form.get("save_name", "").strip()
    if not save_name:
        return redirect(url_for("index"))
    path = find_save_file(save_name)
    if path and path.exists():
        path.unlink()
    if session.get("save_name") == save_name:
        session.pop("save_name", None)
    return redirect(url_for("index"))


@app.route("/character", methods=["GET", "POST"])
def character_create():
    state = get_state_from_session()
    if not state:
        return redirect(url_for("index"))

    available_items = session.get(STARTER_ITEMS_SESSION_KEY)
    need_regenerate = (
        not isinstance(available_items, list)
        or len(available_items) < 3
        or (request.method == "GET" and available_items == DEFAULT_STARTER_ITEMS)
    )
    if need_regenerate:
        available_items = generate_starter_items(state)
        session[STARTER_ITEMS_SESSION_KEY] = available_items

    if request.method == "POST":
        if request.form.get("refresh_items") == "1":
            available_items = generate_starter_items(state)
            session[STARTER_ITEMS_SESSION_KEY] = available_items
            return render_template("character.html", starter_items=available_items)

        selected_items = request.form.getlist("items")
        selected_unique: list[str] = []
        allowed_items = set(available_items)
        for item in selected_items:
            if item in allowed_items and item not in selected_unique:
                selected_unique.append(item)

        if len(selected_unique) != 3:
            return render_template(
                "character.html",
                starter_items=available_items,
                error="Нужно выбрать ровно 3 стартовых предмета.",
            )
        state.character = Character(
            name=request.form.get("name", "").strip(),
            race=request.form.get("race", "").strip(),
            character_class=request.form.get("character_class", "").strip(),
            appearance=request.form.get("appearance", "").strip(),
            age=request.form.get("age", "").strip(),
            gender=request.form.get("gender", "").strip(),
            hp=20,
            inventory=selected_unique,
        )
        if not all(
            [
                state.character.name,
                state.character.race,
                state.character.character_class,
                state.character.appearance,
                state.character.age,
                state.character.gender,
            ]
        ):
            return render_template(
                "character.html",
                starter_items=available_items,
                error="Заполни все поля персонажа.",
            )
        session.pop(STARTER_ITEMS_SESSION_KEY, None)
        state.current_scene = generate_scene(state)
        state.current_scene_image_url = generate_scene_image(state)
        save_game(state)
        return redirect(url_for("game"))
    return render_template("character.html", starter_items=available_items)


@app.route("/game", methods=["GET", "POST"])
def game():
    state = get_state_from_session()
    if not state:
        return redirect(url_for("index"))
    if state.game_over:
        return render_template("game_over.html", state=state)
    if not state.current_scene:
        state.current_scene = generate_scene(state)
        state.current_scene_image_url = generate_scene_image(state)
        add_debug_log(state, "Инициализирована первая сцена", level="system")
        save_game(state)

    turn_result = None
    if request.method == "POST":
        action = request.form.get("action", "").strip()
        custom_action = request.form.get("custom_action", "").strip()
        chosen_action = custom_action if custom_action else action
        if not chosen_action:
            chosen_action = "Остаться на месте и оценить ситуацию"
        add_debug_log(state, f"Ход игрока: {chosen_action}", level="info")

        current_scene_sig = scene_signature(state.current_scene.description) if state.current_scene else ""
        dc = calculate_difficulty(chosen_action)
        roll = roll_d20()
        success = roll >= dc
        hp_change = 0
        enemy_damage = 0

        if not success:
            # On failure: either standard penalty (-1 HP) or enemy hit (2-4 HP).
            if random.random() < 0.5:
                hp_change -= 1
            else:
                enemy_damage = random.randint(2, 4)
                hp_change -= enemy_damage

        state.character.hp += hp_change
        add_debug_log(
            state,
            f"d20={roll}, DC={dc}, {'успех' if success else 'провал'}, HP change={hp_change}, HP now={state.character.hp}",
            level="combat" if not success else "roll",
            add_easter=True,
        )
        if state.character.hp <= 0:
            state.character.hp = 0
            state.game_over = True
            add_debug_log(state, "HP <= 0. Игра окончена", level="error")

        turn_result = {
            "action": chosen_action,
            "roll": roll,
            "dc": dc,
            "success": success,
            "hp_change": hp_change,
            "enemy_damage": enemy_damage,
            "scene_signature": current_scene_sig,
        }
        state.history.append(turn_result)

        if not state.game_over:
            state.current_scene = generate_scene(state)
            state.current_scene_image_url = generate_scene_image(state)
            add_debug_log(state, "Сцена после хода обновлена", level="system")
        save_game(state)

    return render_template("game.html", state=state, turn_result=turn_result)


@app.route("/debug/toggle", methods=["POST"])
def toggle_debug():
    state = get_state_from_session()
    if not state:
        return redirect(url_for("index"))
    state.debug_enabled = not state.debug_enabled
    if state.debug_enabled:
        add_debug_log(state, "DEBUG режим включен", level="system", add_easter=False)
    save_game(state)
    return redirect(url_for("game"))


@app.route("/exit", methods=["POST"])
def exit_to_menu():
    session.pop("save_name", None)
    session.pop(STARTER_ITEMS_SESSION_KEY, None)
    return redirect(url_for("index"))


@app.route("/restart", methods=["POST"])
def restart():
    save_name = session.get("save_name")
    if save_name:
        path = save_path(save_name)
        if path.exists():
            path.unlink()
    session.pop("save_name", None)
    session.pop(STARTER_ITEMS_SESSION_KEY, None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
