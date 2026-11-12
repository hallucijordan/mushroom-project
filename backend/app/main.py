from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import Settings
from backend.app.db.database import init_db

settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：初始化数据库
    await init_db(settings.database_url)
    yield
    # 关闭时：清理资源（如果需要）

app = FastAPI(title="MushRoom Academy API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
from backend.app.routers import training, detection, session
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(detection.router, prefix="/api/detect", tags=["detection"])
app.include_router(session.router, prefix="/api/session", tags=["session"])

@app.get("/api/health")
async def health():
    return {"status": "ok"}