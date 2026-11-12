"""
Shared application-level singletons.

Import from here instead of instantiating locally in each router,
so all routers share the same ALEngine session dictionary.
"""
from backend.app.services.al_engine import ALEngine
from backend.app.services.gemini_bridge import GeminiBridge
from backend.app.config import Settings

_settings = Settings()

# One ALEngine for the whole process — all sessions live here
al_engine = ALEngine()

# One GeminiBridge for the whole process. Only Expert Mode needs it, so the
# server still starts without a key and /api/detect returns 503 instead.
gemini = GeminiBridge(api_key=_settings.gemini_api_key) if _settings.gemini_api_key else None
