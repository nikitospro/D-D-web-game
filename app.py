import json
import os
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from flask import Flask, redirect, render_template, request, session, url_for


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

SAVE_DIR = Path("saves")
SAVE_DIR.mkdir(exist_ok=True)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")

STARTER_ITEMS = [
    "Нож",
    "Зелье здоровья",
    "Меч",
    "Короткий меч",
    "Кожаная броня",
    "Лук",
    "Факел",
    "Веревка",
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
    current_scene: Scene | None = None
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
        current_scene=scene,
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


def call_ollama(prompt: str) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        text = data.get("response", "").strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        return text
    except Exception:
        return ""


def fallback_scene(state: GameState) -> Scene:
    templates = [
        (
            (
                "Ты стоишь у старых ворот заброшенной крепости, где ветер гонит пепел по треснувшим плитам. "
                "Над башнями кружат вороны, а из глубины двора доносятся тяжелые шаги и скрип цепей. "
                "У входа ты замечаешь сорванную печать и свежие следы, будто кто-то совсем недавно прошел внутрь. "
                "Пахнет сыростью, железом и чем-то гнилым, что явно не предвещает ничего хорошего."
            ),
            ["Осмотреть ворота", "Прокрасться внутрь", "Позвать тех, кто внутри"],
        ),
        (
            (
                "Ночной лес затянуло туманом, и даже лунный свет едва пробивается сквозь густые ветви. "
                "Впереди мерцает костер, вокруг которого сидят незнакомцы в дорожных плащах, но их лица скрыты капюшонами. "
                "Между деревьями ты слышишь редкий звон металла и шепот на незнакомом языке. "
                "Тропа под ногами уходит в грязь, а рядом видны следы крупных когтей."
            ),
            ["Подойти открыто", "Спрятаться и наблюдать", "Обойти лагерь стороной"],
        ),
        (
            (
                "Под ногами скрипит камень древнего подземелья, а стены покрыты выцветшими рунами и копотью факелов. "
                "Перед тобой развилка: левый коридор уходит вниз, где слышен шум воды, а правый тянется в темноту, где звенит эхо шагов. "
                "На полу заметны следы борьбы и обломки деревянного щита. "
                "Воздух становится холоднее, и где-то впереди звучит приглушенный стон."
            ),
            ["Идти налево", "Идти направо", "Проверить стены на ловушки"],
        ),
    ]
    description, options = random.choice(templates)
    return Scene(description=description, options=options)


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


def generate_scene(state: GameState) -> Scene:
    recent_turns = state.history[-3:]
    history_text = "Нет предыдущих ходов."
    if recent_turns:
        history_lines = []
        for idx, turn in enumerate(recent_turns, start=1):
            result = "успех" if turn.get("success") else "провал"
            history_lines.append(f"{idx}) {turn.get('action', 'действие')} -> {result}")
        history_text = "\n".join(history_lines)

    prompt = f"""
Ты — ведущий текстовой RPG.
Пиши атмосферно и подробно. Сцена должна быть заметно длиннее: 6-9 предложений.
Добавь конкретные детали окружения, звуки, запахи, угрозу или зацепку для сюжета.
Учитывай прошлые действия игрока.
Нельзя писать служебные комментарии, мысли модели или объяснения правил.

Мир: {state.world_scenario}
Главная цель игрока: {state.main_goal}
Персонаж:
- Имя: {state.character.name}
- Раса: {state.character.race}
- Класс: {state.character.character_class}
- Внешность: {state.character.appearance}
- Возраст: {state.character.age}
- Пол: {state.character.gender}
- Инвентарь: {", ".join(state.character.inventory) if state.character.inventory else "пусто"}

Последние ходы игрока:
{history_text}

Формат ответа:
Описание сцены (6-9 предложений, русский язык)
1. Вариант 1 (кратко, до 12 слов)
2. Вариант 2 (кратко, до 12 слов)
3. Вариант 3 (кратко, до 12 слов)
"""
    model_text = call_ollama(prompt)
    scene = parse_scene_response(model_text)
    if scene is None:
        scene = fallback_scene(state)
    return scene


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
    if request.method == "POST":
        selected_items = request.form.getlist("items")
        if len(selected_items) != 3:
            return render_template(
                "character.html",
                starter_items=STARTER_ITEMS,
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
            inventory=selected_items,
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
                starter_items=STARTER_ITEMS,
                error="Заполни все поля персонажа.",
            )
        state.current_scene = generate_scene(state)
        save_game(state)
        return redirect(url_for("game"))
    return render_template("character.html", starter_items=STARTER_ITEMS)


@app.route("/game", methods=["GET", "POST"])
def game():
    state = get_state_from_session()
    if not state:
        return redirect(url_for("index"))
    if state.game_over:
        return render_template("game_over.html", state=state)
    if not state.current_scene:
        state.current_scene = generate_scene(state)
        save_game(state)

    turn_result = None
    if request.method == "POST":
        action = request.form.get("action", "").strip()
        custom_action = request.form.get("custom_action", "").strip()
        chosen_action = custom_action if custom_action else action
        if not chosen_action:
            chosen_action = "Остаться на месте и оценить ситуацию"

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
        if state.character.hp <= 0:
            state.character.hp = 0
            state.game_over = True

        turn_result = {
            "action": chosen_action,
            "roll": roll,
            "dc": dc,
            "success": success,
            "hp_change": hp_change,
            "enemy_damage": enemy_damage,
        }
        state.history.append(turn_result)

        if not state.game_over:
            state.current_scene = generate_scene(state)
        save_game(state)

    return render_template("game.html", state=state, turn_result=turn_result)


@app.route("/exit", methods=["POST"])
def exit_to_menu():
    session.pop("save_name", None)
    return redirect(url_for("index"))


@app.route("/restart", methods=["POST"])
def restart():
    save_name = session.get("save_name")
    if save_name:
        path = save_path(save_name)
        if path.exists():
            path.unlink()
    session.pop("save_name", None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
