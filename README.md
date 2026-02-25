# DND AI MVP

Локальная веб-RPG на `Python + Flask` с генерацией сцен через `Ollama` и генерацией иллюстраций сцен через `ComfyUI`.

## Возможности

- создание и загрузка нескольких сохранений;
- создание персонажа (имя, раса, класс, внешность, возраст, пол, предметы);
- игровой цикл с броском `d20`, проверками действий и системой `HP`;
- генерация текстовых сцен нейросетью (`qwen3:8b`);
- генерация изображений сцен через `ComfyUI`.

## Требования

- Python `3.10+`
- Ollama
- (опционально) ComfyUI

## Быстрый старт

### 1. Запусти Ollama

```powershell
ollama pull qwen3:8b
ollama serve
```

### 2. Установи зависимости

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Запусти приложение

```powershell
python app.py
```

### 4. Открой в браузере

- `http://127.0.0.1:5000`

## ComfyUI (картинки сцен)

Если ComfyUI запущен, игра будет автоматически пытаться генерировать изображение для текущей сцены.

Поддерживаемые URL по умолчанию:

- `http://127.0.0.1:8188`
- `http://127.0.0.1:8000`
- `http://localhost:8188`
- `http://localhost:8000`

Если нужен конкретный адрес, задай `COMFYUI_URL`.

## Переменные окружения

### Основные

- `FLASK_SECRET_KEY`
- `OLLAMA_URL` (по умолчанию `http://127.0.0.1:11434`)
- `OLLAMA_MODEL` (по умолчанию `qwen3:8b`)
- `OLLAMA_TIMEOUT_SECONDS` (по умолчанию `240`)
- `OLLAMA_AUTOSTART` (`1`/`0`, по умолчанию `1`)

### ComfyUI

- `COMFYUI_ENABLED` (`1`/`0`, по умолчанию `1`)
- `COMFYUI_URL` (по умолчанию `http://127.0.0.1:8188`)
- `COMFYUI_FALLBACK_URLS` (доп. адреса через запятую)
- `COMFYUI_TIMEOUT_SECONDS` (по умолчанию `180`)
- `COMFYUI_MAX_WAIT_SECONDS` (по умолчанию `120`)
- `COMFYUI_POLL_INTERVAL_SECONDS` (по умолчанию `1.2`)
- `COMFYUI_CHECKPOINT` (точное имя чекпоинта, опционально)
- `COMFYUI_WORKFLOW_PATH` (по умолчанию `comfyui_workflow_api.json`)
- `COMFYUI_NEGATIVE_PROMPT` (опционально)

## Workflow для ComfyUI

Если файл `COMFYUI_WORKFLOW_PATH` существует, приложение использует его как API-workflow.
Если файла нет, используется встроенный минимальный workflow.

## Примечания

- Сохранения лежат в папке `saves/`.
- При недоступности Ollama или ComfyUI игра не падает: используется безопасный fallback.
