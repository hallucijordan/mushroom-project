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

# One GeminiBridge for the whole process
gemini = GeminiBridge(api_key=_settings.gemini_api_key)
