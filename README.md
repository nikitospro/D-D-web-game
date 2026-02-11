# DND AI GAME

Локальная веб-игра по твоему ТЗ: создание сохранения, создание персонажа, игровой цикл с d20, HP и генерацией сцен через Ollama (`qwen3:8b`).

## Запуск

1. Установи Ollama и модель:
```powershell
ollama pull qwen3:8b
ollama serve
```

2. Установи зависимости:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Запусти приложение:
```powershell
python app.py
```

4. Открой в браузере:
`http://127.0.0.1:5000`

## Переменные окружения (опционально)

- `OLLAMA_URL` (по умолчанию `http://127.0.0.1:11434`)
- `OLLAMA_MODEL` (по умолчанию `qwen3:8b`)
- `FLASK_SECRET_KEY` (рекомендуется поменять в проде)

## СКРИНШОТЫ
<img width="647" height="653" alt="image" src="https://github.com/user-attachments/assets/9d495887-819c-4231-8756-a354d9a47e03" />

<img width="942" height="684" alt="image" src="https://github.com/user-attachments/assets/442ec841-f8f9-4958-a904-9e5cc1e0a935" />

<img width="953" height="639" alt="image" src="https://github.com/user-attachments/assets/acd2eb04-6968-4bd4-bb12-c1c69eae46b8" />
