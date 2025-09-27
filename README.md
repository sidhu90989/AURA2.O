# AURA 2.0

> Intelligent multi-modal assistant (NLP, Vision, Voice, System Monitoring, Web Control, Maps, Memory) — initial repository setup.

## 🚀 Features (Planned / Present Code Modules)
- `nlp_processor.py` – Natural language understanding / intent processing.
- `vision_processor.py` – Face recognition & visual analysis (uses `vision_db/face_vectors.db`).
- `voice_interface.py` – Speech input/output handling.
- `memory_manager.py` – Long/short term memory utilities.
- `map_service.py` – Mapping & geospatial utilities.
- `neuro_web_controller.py` – Web automation / control.
- `system_monitor.py` – Host system metrics & health.
- `ethical.py` / `security.py` / `access.py` – Guardrails, permissions & policies.

## ⚠️ Security & Privacy
Biometric data (faces) and encryption keys SHOULD NOT be committed.
This repo's `.gitignore` excludes:
- `nlp-env/` (virtual environment)
- `known_faces/` and encrypted face store `known_faces_.enc/`
- `*.key` encryption keys
- `*.log` runtime logs

If you need to provide sample data, create sanitized assets under `samples/` and whitelist them.

## 🔐 Secrets & Keys
Store runtime secrets in environment variables or a `.env` file that remains untracked.
Example:
```
AURA_OPENAI_API_KEY=sk-...
AURA_ENV=dev
```
Load with `python-dotenv` or your own config loader.

## 🛠️ Quick Start
```bash
# (Windows PowerShell examples)
python -m venv nlp-env
./nlp-env/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

## 📦 Dependencies
All core dependencies should be listed in `requirements.txt`. If something is missing after import errors, add & pin it:
```
pip install somepackage==1.2.3
pip freeze | Select-String somepackage
```
Then append to `requirements.txt`.

## 🧪 Testing (Add Later)
Create a `tests/` folder and use `pytest` or `unittest` to validate modules.

## 🗺️ Suggested Next Steps
- Add a LICENSE (e.g., MIT) if you plan to make this public/open-source.
- Add structured logging configuration (rotate, JSON logs).
- Implement unit tests for safety-critical modules (`security.py`, `ethical.py`).
- Provide an architecture diagram.
- Add type hints & mypy config.

## 🤝 Contributing
1. Fork
2. Create feature branch
3. Commit with conventional style (`feat:`, `fix:`)
4. Open PR

## 📄 License
Not yet selected. Add a `LICENSE` file (MIT/Apache-2.0/BSD-3-Clause) before broad sharing.

---
Initial scaffold generated. Update this README as functionality evolves.
