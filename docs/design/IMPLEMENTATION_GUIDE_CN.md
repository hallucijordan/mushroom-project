# MushRoom Academy — 完整实现指南

> 本文档是面向开发者的逐步实现教程。按顺序完成每个阶段，每个阶段结束时都会有一个可运行的里程碑。

---

## 目录

- [阶段 0：环境准备与项目重组](#阶段-0环境准备与项目重组)
- [阶段 1：ML 包独立化](#阶段-1ml-包独立化)
- [阶段 2：FastAPI 后端搭建](#阶段-2fastapi-后端搭建)
- [阶段 3：React 前端基础](#阶段-3react-前端基础)
- [阶段 4：猫咪角色系统](#阶段-4猫咪角色系统)
- [阶段 5：训练旅程（核心玩法）](#阶段-5训练旅程核心玩法)
- [阶段 6：专家模式（双裁判增强版）](#阶段-6专家模式双裁判增强版)
- [阶段 7：策略对比可视化](#阶段-7策略对比可视化)
- [阶段 8：Docker 容器化](#阶段-8docker-容器化)
- [阶段 9：AWS 部署](#阶段-9aws-部署)
- [阶段 10：美术资源与打磨](#阶段-10美术资源与打磨)
- [附录：前置知识速查](#附录前置知识速查)

---

## 阶段 0：环境准备与项目重组

### 目标
把现有的单体 Gradio 项目重组为三个独立模块：`ml/`、`backend/`、`frontend/`。

### 步骤

#### 0.1 安装全局工具

```bash
# Node.js (前端需要)
brew install node          # macOS
# 或者去 https://nodejs.org 下载 LTS 版本

# 确认 Python 3.11
python3 --version          # 需要 3.11+

# Docker (后面容器化需要)
brew install --cask docker  # macOS，装完后启动 Docker Desktop
```

#### 0.2 创建新的目录结构

在项目根目录下执行：

```bash
# 创建三大模块
mkdir -p ml/mushroom_ml/learners
mkdir -p ml/mushroom_ml/strategies
mkdir -p ml/tests
mkdir -p backend/app/routers
mkdir -p backend/app/services
mkdir -p backend/app/schemas
mkdir -p backend/app/db
mkdir -p backend/app/ws
mkdir -p frontend
mkdir -p deploy
```

> **注意**：原来的 `src/`、`app.py`、`train.py` 暂时不要删除。我们会逐步迁移代码，确认新模块正常后再清理旧代码。

#### 0.3 Git 分支管理

```bash
git checkout -b v2-restructure
```

在这个分支上做所有重构工作，确认稳定后再合并回 `main`。

### 里程碑检查
- [ ] 目录结构已创建
- [ ] 新分支已建立
- [ ] 原有代码 `python app.py` 仍然可以正常运行（没有破坏旧功能）

---

## 阶段 1：ML 包独立化

### 目标
把所有机器学习代码抽成独立的 `mushroom_ml` Python 包，可以用 `pip install -e ml/` 安装。

### 为什么这一步最先做？
因为 ML 包是后端的依赖。先把它稳定下来，后端和前端的开发才能顺利进行。

### 步骤

#### 1.1 创建包配置文件

创建 `ml/pyproject.toml`：

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "mushroom-ml"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "scikit-learn>=1.3",
    "xgboost>=2.0",
    "numpy>=1.24",
    "pandas>=2.0",
    "joblib>=1.3",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov"]
```

创建 `ml/mushroom_ml/__init__.py`：

```python
from mushroom_ml.active_learning import ALSession
from mushroom_ml.preprocessor import MushroomPreprocessor
from mushroom_ml.learners import get_learner, LEARNER_REGISTRY
from mushroom_ml.strategies import get_strategy, STRATEGY_REGISTRY
```

#### 1.2 迁移 Preprocessor

把 `src/data/preprocessor.py` 复制到 `ml/mushroom_ml/preprocessor.py`。

**需要修改的地方**：
- 把内部的文件路径引用从相对路径改为可配置路径
- `load_secondary()` 函数的数据目录改为参数传入

```python
# 修改前（硬编码路径）
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

# 修改后（可配置）
def load_secondary(data_dir: str = "./data"):
    path = os.path.join(data_dir, "secondary_data.csv")
    ...
```

#### 1.3 定义 AbstractLearner 接口

创建 `ml/mushroom_ml/learners/base.py`：

```python
from abc import ABC, abstractmethod
import numpy as np

class AbstractLearner(ABC):
    """所有蘑菇分类模型必须实现此接口。"""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AbstractLearner": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def get_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """返回每个样本的不确定度分数，越高=越不确定。"""
        ...

    def predict_single(self, X: np.ndarray) -> dict:
        proba = self.predict_proba(X)[0]
        label_idx = int(np.argmax(proba))
        return {
            "label": "edible" if label_idx == 0 else "poisonous",
            "confidence": float(proba[label_idx]),
            "p_edible": float(proba[0]),
            "p_poisonous": float(proba[1]),
        }

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "AbstractLearner": ...
```

> **核心理解**：这个接口就像一份"合同"。任何新的分类器只要签了这份合同（实现所有抽象方法），就能直接插入系统使用，不需要改其他代码。

#### 1.4 迁移 RandomForest 实现

把 `src/models/base_learner.py` 的逻辑迁移到 `ml/mushroom_ml/learners/random_forest.py`，使其实现 `AbstractLearner`：

```python
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from mushroom_ml.learners.base import AbstractLearner

class RandomForestLearner(AbstractLearner):
    def __init__(self, n_estimators=300, max_depth=None, random_state=42):
        rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=-1,
        )
        self.model = CalibratedClassifierCV(rf, method="isotonic", cv=3)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_uncertainty(self, X):
        proba = self.predict_proba(X)
        return 1.0 - np.max(proba, axis=1)  # 最大概率越低=越不确定

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)
```

#### 1.5 实现另外两个 Learner

**`ml/mushroom_ml/learners/logistic.py`** — Bagged Logistic Regression：
- 用 `BaggingClassifier(LogisticRegression(max_iter=1000), n_estimators=10)`
- `get_uncertainty` 用预测概率的**熵**：`-sum(p * log(p))`

**`ml/mushroom_ml/learners/mlp.py`** — MLP 神经网络：
- 用 `MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)`
- `get_uncertainty` 同样用熵

#### 1.6 创建 Learner 注册表

`ml/mushroom_ml/learners/__init__.py`：

```python
from mushroom_ml.learners.random_forest import RandomForestLearner
from mushroom_ml.learners.logistic import BaggedLogisticLearner
from mushroom_ml.learners.mlp import MLPLearner

LEARNER_REGISTRY = {
    "random_forest": RandomForestLearner,
    "logistic": BaggedLogisticLearner,
    "mlp": MLPLearner,
}

def get_learner(key: str, **kwargs):
    if key not in LEARNER_REGISTRY:
        raise ValueError(f"未知的 learner: {key}，可用: {list(LEARNER_REGISTRY)}")
    return LEARNER_REGISTRY[key](**kwargs)
```

#### 1.7 定义 AbstractStrategy 并实现四种策略

**`ml/mushroom_ml/strategies/base.py`**：

```python
from abc import ABC, abstractmethod
import numpy as np
from mushroom_ml.learners.base import AbstractLearner

class AbstractStrategy(ABC):
    name: str
    key: str

    @abstractmethod
    def query(self, learner: AbstractLearner, X_pool: np.ndarray, n_samples: int) -> np.ndarray:
        """从 X_pool 中选出下一批要标注的样本索引。"""
        ...
```

四种实现：

| 文件 | 类名 | 核心逻辑 |
|---|---|---|
| `random.py` | `RandomSampling` | `np.random.choice(len(X_pool), n_samples, replace=False)` |
| `uncertainty.py` | `UncertaintySampling` | 选不确定度最高的 n 个：`np.argsort(learner.get_uncertainty(X_pool))[-n:]` |
| `committee.py` | `QueryByCommittee` | 训练 5 个 bootstrap 模型，选投票分歧最大的样本 |
| `bald.py` | `BALDSampling` | 最大化互信息（可以用 MC Dropout 近似） |

> **建议实现顺序**：先做 Random 和 Uncertainty（最简单），确认流程通了再做 Committee 和 BALD。

#### 1.8 实现 ALSession

把 `main.ipynb` 里的 active learning 循环重构为 `ml/mushroom_ml/active_learning.py` 中的 `ALSession` 类。详见 [04_ml_packages.md](04_ml_packages.md) 中的完整实现。

核心方法：
- `predict_sample(X)` — 预测单个样本
- `label_sample(pool_index)` — 揭示标签，加入训练集
- `query_batch(n)` — 用策略选下一批
- `retrain()` — 重新训练模型
- `get_learning_curve()` — 获取学习曲线数据

#### 1.9 创建 evaluation.py

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def compute_metrics(learner, X, y, cv=5):
    cv_scores = cross_val_score(learner.model, X, y, cv=cv, scoring='accuracy')
    return {
        "cv_accuracy": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
    }
```

#### 1.10 安装并测试

```bash
cd ml
pip install -e ".[dev]"

# 跑测试
pytest tests/ -v
```

写一个简单测试 `ml/tests/test_learners.py`：

```python
import numpy as np
from mushroom_ml.learners import get_learner

def test_random_forest_fit_predict():
    learner = get_learner("random_forest", n_estimators=10)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    learner.fit(X, y)
    pred = learner.predict(X)
    assert pred.shape == (100,)
    assert set(pred).issubset({0, 1})
```

### 里程碑检查
- [ ] `pip install -e ml/` 成功
- [ ] `from mushroom_ml import get_learner, get_strategy, ALSession` 不报错
- [ ] `pytest ml/tests/` 全部通过
- [ ] 三种 learner 都可以 fit + predict
- [ ] Random 和 Uncertainty 两种策略可以正确选出样本

---

## 阶段 2：FastAPI 后端搭建

### 目标
用 FastAPI 替代原来的 Gradio `app.py`，暴露 REST API + WebSocket。

### 步骤

#### 2.1 安装后端依赖

创建 `backend/requirements.txt`：

```
fastapi>=0.110
uvicorn[standard]>=0.27
pydantic>=2.0
pydantic-settings>=2.0
sqlalchemy>=2.0
aiosqlite>=0.19
python-multipart>=0.0.6
websockets>=12.0
google-genai>=1.0
pillow>=10.0
python-dotenv>=1.0
```

```bash
pip install -r backend/requirements.txt
```

#### 2.2 配置管理

创建 `backend/app/config.py`：

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    env: str = "development"
    gemini_api_key: str = ""
    database_url: str = "sqlite+aiosqlite:///./data/sessions.db"
    model_dir: str = "./models"
    data_dir: str = "./data"
    allowed_origins: list[str] = ["http://localhost:3000"]

    class Config:
        env_file = ".env"
```

> **关键点**：所有敏感信息（API key）都从环境变量或 `.env` 读取，永远不要硬编码。

#### 2.3 数据库模型

创建 `backend/app/db/database.py`：

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

# engine 和 session_factory 在 main.py 的 lifespan 中初始化
engine = None
SessionLocal = None

async def init_db(database_url: str):
    global engine, SessionLocal
    engine = create_async_engine(database_url)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

创建 `backend/app/db/models.py`（ORM 模型）：

```python
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.sql import func
from backend.app.db.database import Base

class TrainingSession(Base):
    __tablename__ = "training_session"
    id = Column(String, primary_key=True)
    cat_name = Column(String, default="Professor Whiskers")
    cat_level = Column(Integer, default=0)
    cat_xp = Column(Integer, default=0)
    learner_type = Column(String, nullable=False)
    strategy = Column(String, nullable=False)
    init_strategy = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

class LabeledSample(Base):
    __tablename__ = "labeled_sample"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("training_session.id"))
    sample_index = Column(Integer)
    true_label = Column(Integer)
    predicted_label = Column(Integer)
    predicted_conf = Column(Float)
    was_correct = Column(Boolean)
    labeled_at = Column(DateTime, server_default=func.now())

# TrainingSnapshot 和 Achievement 同理，字段见 03_training_journey.md
```

#### 2.4 FastAPI 主入口

创建 `backend/app/main.py`：

```python
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
```

#### 2.5 实现核心路由

**Training 路由** (`backend/app/routers/training.py`)：

先实现这三个最核心的端点：
1. `POST /start` — 创建新训练会话
2. `POST /{session_id}/predict` — 猫做预测
3. `POST /{session_id}/reveal` — 用户揭示标签

```python
from fastapi import APIRouter, HTTPException
from backend.app.schemas.training import StartRequest, StartResponse, PredictResponse, RevealRequest, RevealResponse
from backend.app.services.al_engine import ALEngine

router = APIRouter()
al_engine = ALEngine()  # 单例，管理所有活跃的 ALSession

@router.post("/start", response_model=StartResponse)
async def start_training(req: StartRequest):
    session_id = al_engine.create_session(
        learner_key=req.learner_type,
        strategy_key=req.strategy,
        init_method=req.init_strategy,
    )
    state = al_engine.get_state(session_id)
    return StartResponse(session_id=session_id, **state)

@router.post("/{session_id}/predict", response_model=PredictResponse)
async def predict(session_id: str):
    # 从池中取下一个样本，猫做预测
    result = al_engine.predict_next(session_id)
    return result

@router.post("/{session_id}/reveal", response_model=RevealResponse)
async def reveal(session_id: str, req: RevealRequest):
    result = al_engine.reveal_and_retrain(session_id, req.pool_index)
    return result
```

#### 2.6 AL Engine 服务

`backend/app/services/al_engine.py`：

```python
import uuid
import numpy as np
from mushroom_ml import ALSession, get_learner, get_strategy, MushroomPreprocessor

class ALEngine:
    """管理所有活跃的训练会话。内存中保存 ALSession 实例。"""

    def __init__(self):
        self.sessions: dict[str, ALSession] = {}
        self._preprocessor = None
        self._X_full = None
        self._y_full = None

    def _load_data(self):
        if self._preprocessor is None:
            self._preprocessor = MushroomPreprocessor.load()
            self._X_full, self._y_full = self._preprocessor.load_pool_data()

    def create_session(self, learner_key: str, strategy_key: str, init_method: str) -> str:
        self._load_data()
        session_id = str(uuid.uuid4())

        learner = get_learner(learner_key)
        strategy = get_strategy(strategy_key)

        # 初始化标注集（200个样本）
        n_init = 200
        if init_method == "d_optimal":
            indices = self._d_optimal_init(self._X_full, n_init)
        else:
            indices = np.random.choice(len(self._X_full), n_init, replace=False)

        X_init = self._X_full[indices]
        y_init = self._y_full[indices]
        mask = np.ones(len(self._X_full), dtype=bool)
        mask[indices] = False
        X_pool = self._X_full[mask]
        y_pool = self._y_full[mask]

        self.sessions[session_id] = ALSession(
            learner=learner, strategy=strategy,
            X_pool=X_pool, y_pool=y_pool,
            X_initial=X_init, y_initial=y_init,
        )
        return session_id

    def predict_next(self, session_id: str) -> dict:
        session = self.sessions[session_id]
        # 用策略选1个样本
        indices = session.query_batch(n_samples=1)
        idx = indices[0]
        X_sample = session.X_pool[idx:idx+1]
        prediction = session.predict_sample(X_sample)
        return {"pool_index": int(idx), **prediction}

    def reveal_and_retrain(self, session_id: str, pool_index: int) -> dict:
        session = self.sessions[session_id]
        result = session.label_sample(pool_index)

        # 每积累50个新标签就重新训练
        if session.labeled_count % 50 == 0:
            metrics = session.retrain()
            result["retrained"] = True
            result["metrics"] = metrics
        else:
            result["retrained"] = False

        return result
```

#### 2.7 Gemini Bridge 服务

把 `src/agents/gemini_extractor.py` 和 `src/judges/direct_judge.py` 合并迁移到 `backend/app/services/gemini_bridge.py`：

```python
from google import genai
from PIL import Image
# 复用原有的 GeminiExtractor 和 DirectJudge 逻辑
# 但统一在一个服务类中管理 API client
```

#### 2.8 启动测试

```bash
# 创建 .env
echo "GEMINI_API_KEY=your-key-here" > .env

# 启动后端
uvicorn backend.app.main:app --reload --port 8000

# 测试健康检查
curl http://localhost:8000/api/health
# 应返回 {"status": "ok"}
```

### 里程碑检查
- [ ] `uvicorn` 启动不报错
- [ ] `/api/health` 返回 200
- [ ] `POST /api/training/start` 能创建会话并返回 session_id
- [ ] `POST /api/training/{id}/predict` 返回预测结果
- [ ] `POST /api/training/{id}/reveal` 能正确记录标签和重训练

---

## 阶段 3：React 前端基础

### 目标
搭建 React + TypeScript + Vite 项目骨架，实现路由和基础页面。

### 步骤

#### 3.1 初始化项目

```bash
cd frontend
npm create vite@latest . -- --template react-ts
npm install
npm install react-router-dom zustand axios framer-motion
npm install -D tailwindcss @tailwindcss/vite
```

#### 3.2 配置 Tailwind

`frontend/src/index.css`：

```css
@import "tailwindcss";
```

`frontend/vite.config.ts`：

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://localhost:8000',    // 开发时代理到后端
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
})
```

#### 3.3 创建路由

`frontend/src/App.tsx`：

```tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import HomePage from './pages/HomePage'
import TrainingPage from './pages/TrainingPage'
import ExpertPage from './pages/ExpertPage'
import JourneyPage from './pages/JourneyPage'
import ComparisonPage from './pages/ComparisonPage'
import Header from './components/shared/Header'

export default function App() {
  return (
    <BrowserRouter>
      <Header />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/training" element={<TrainingPage />} />
        <Route path="/expert" element={<ExpertPage />} />
        <Route path="/journey" element={<JourneyPage />} />
        <Route path="/comparison" element={<ComparisonPage />} />
      </Routes>
    </BrowserRouter>
  )
}
```

#### 3.4 创建占位页面

每个页面先写一个最简版本，确保路由工作：

```tsx
// frontend/src/pages/HomePage.tsx
export default function HomePage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      <h1 className="text-4xl font-bold">🍄 MushRoom Academy</h1>
      <p className="mt-4 text-gray-600">遇见你的猫咪学徒，开始蘑菇鉴定之旅</p>
    </div>
  )
}
```

其他页面类似，先放占位文字。

#### 3.5 API 服务层

`frontend/src/services/api.ts`：

```typescript
import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const trainingApi = {
  start: (data: { learner_type: string; strategy: string; init_strategy: string }) =>
    api.post('/training/start', data),
  predict: (sessionId: string) =>
    api.post(`/training/${sessionId}/predict`),
  reveal: (sessionId: string, poolIndex: number) =>
    api.post(`/training/${sessionId}/reveal`, { pool_index: poolIndex }),
  getCurve: (sessionId: string) =>
    api.get(`/training/${sessionId}/curve`),
}

export const detectApi = {
  analyze: (image: File) => {
    const form = new FormData()
    form.append('image', image)
    return api.post('/detect/analyze', form)
  },
}
```

#### 3.6 TypeScript 类型定义

`frontend/src/types/cat.ts`：

```typescript
export type CatMood = 'idle' | 'thinking' | 'happy' | 'sick' | 'proud' | 'curious' | 'sleeping' | 'confused' | 'celebrating'

export interface CatState {
  level: number
  xp: number
  mood: CatMood
  name: string
  accuracy: number
  totalPredictions: number
  correctPredictions: number
  currentDialogue: string
}
```

`frontend/src/types/training.ts`：

```typescript
export type QueryStrategy = 'random' | 'uncertainty' | 'committee' | 'bald'
export type LearnerType = 'random_forest' | 'logistic' | 'mlp'

export interface Prediction {
  pool_index: number
  label: 'edible' | 'poisonous'
  confidence: number
  p_edible: number
  p_poisonous: number
}
```

#### 3.7 Zustand Stores

`frontend/src/stores/catStore.ts`：

```typescript
import { create } from 'zustand'
import type { CatMood } from '../types/cat'

interface CatStore {
  level: number
  xp: number
  mood: CatMood
  name: string
  accuracy: number
  currentDialogue: string
  setMood: (mood: CatMood) => void
  addXP: (amount: number) => void
  setDialogue: (text: string) => void
}

const XP_THRESHOLDS = [0, 100, 300, 600, 1000, 1500] // 升级所需累计 XP

export const useCatStore = create<CatStore>((set, get) => ({
  level: 0,
  xp: 0,
  mood: 'idle',
  name: 'Professor Whiskers',
  accuracy: 0,
  currentDialogue: '',
  setMood: (mood) => set({ mood }),
  addXP: (amount) => {
    const newXP = get().xp + amount
    let newLevel = get().level
    while (newLevel < 5 && newXP >= XP_THRESHOLDS[newLevel + 1]) {
      newLevel++
    }
    set({ xp: newXP, level: newLevel })
  },
  setDialogue: (text) => set({ currentDialogue: text }),
}))
```

#### 3.8 启动前端

```bash
cd frontend
npm run dev
# 打开 http://localhost:3000
```

### 里程碑检查
- [ ] `npm run dev` 启动成功，浏览器能看到首页
- [ ] 所有 5 个路由可以导航
- [ ] Vite proxy 配置正确：在前端访问 `/api/health` 能打通后端
- [ ] TypeScript 编译无错误

---

## 阶段 4：猫咪角色系统

### 目标
实现猫咪的视觉展示、动画状态机和对话系统。

### 步骤

#### 4.1 生成美术资源

用 AI 图像生成工具（推荐 DALL-E 3 或 Midjourney）生成猫咪素材。

**最低可行集**（先做 Level 0 的 4 个关键状态）：

1. `idle` — 猫咪坐着，尾巴轻摇
2. `thinking` — 猫咪歪头思考
3. `happy` — 猫咪开心跳起
4. `sick` — 猫咪中毒倒地（绿脸）

使用 [02_cat_agent.md](02_cat_agent.md) 中的 prompt 模板生成。

> **务实建议**：第一轮用静态 PNG 就够了。先让功能跑通，后面再升级为 Lottie 动画或 sprite sheet。

放置路径：`frontend/public/assets/cat/level0/{idle,thinking,happy,sick}.png`

#### 4.2 CatAvatar 组件

```tsx
// frontend/src/components/cat/CatAvatar.tsx
import { useCatStore } from '../../stores/catStore'
import { motion, AnimatePresence } from 'framer-motion'

const moodAnimations = {
  idle: { y: [0, -5, 0], transition: { repeat: Infinity, duration: 2 } },
  thinking: { rotate: [-5, 5, -5], transition: { repeat: Infinity, duration: 1.5 } },
  happy: { y: [0, -20, 0], transition: { duration: 0.5 } },
  sick: { rotate: 90, x: 20, transition: { duration: 0.3 } },
  // ... 其他状态
}

export default function CatAvatar() {
  const { level, mood } = useCatStore()
  const src = `/assets/cat/level${level}/${mood}.png`

  return (
    <AnimatePresence mode="wait">
      <motion.img
        key={`${level}-${mood}`}
        src={src}
        alt={`Cat - ${mood}`}
        className="w-48 h-48 object-contain"
        animate={moodAnimations[mood] || {}}
        initial={{ opacity: 0, scale: 0.8 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.3 }}
      />
    </AnimatePresence>
  )
}
```

#### 4.3 CatDialogue 组件（打字机效果）

```tsx
// frontend/src/components/cat/CatDialogue.tsx
import { useState, useEffect } from 'react'
import { useCatStore } from '../../stores/catStore'

export default function CatDialogue() {
  const dialogue = useCatStore(s => s.currentDialogue)
  const [displayed, setDisplayed] = useState('')

  useEffect(() => {
    setDisplayed('')
    let i = 0
    const timer = setInterval(() => {
      if (i < dialogue.length) {
        setDisplayed(dialogue.slice(0, i + 1))
        i++
      } else {
        clearInterval(timer)
      }
    }, 30) // 每字 30ms
    return () => clearInterval(timer)
  }, [dialogue])

  return (
    <div className="relative bg-white rounded-2xl p-4 shadow-lg border-2 border-orange-200 max-w-md">
      {/* 气泡尖角 */}
      <div className="absolute -left-3 top-4 w-0 h-0 border-t-8 border-b-8 border-r-8 border-transparent border-r-white" />
      <p className="text-gray-800 min-h-[3rem]">{displayed}<span className="animate-pulse">|</span></p>
    </div>
  )
}
```

#### 4.4 动画状态机 Hook

```typescript
// frontend/src/hooks/useCatAnimation.ts
import { useCallback } from 'react'
import { useCatStore } from '../stores/catStore'
import type { CatMood } from '../types/cat'

export function useCatAnimation() {
  const setMood = useCatStore(s => s.setMood)
  const setDialogue = useCatStore(s => s.setDialogue)

  const playSequence = useCallback(async (steps: { mood: CatMood; dialogue?: string; duration: number }[]) => {
    for (const step of steps) {
      setMood(step.mood)
      if (step.dialogue) setDialogue(step.dialogue)
      await new Promise(r => setTimeout(r, step.duration))
    }
    setMood('idle')
  }, [setMood, setDialogue])

  // 预设序列
  const onNewMushroom = () => playSequence([
    { mood: 'curious', dialogue: '哦！让我看看这个...', duration: 1500 },
    { mood: 'thinking', dialogue: '嗯...让我分析一下特征...', duration: 2000 },
  ])

  const onCorrectPrediction = () => playSequence([
    { mood: 'happy', dialogue: '耶！我说对了！🎉', duration: 2000 },
  ])

  const onPoisonEaten = () => playSequence([
    { mood: 'sick', dialogue: '呃...那个有毒?! *倒地* 这就是为什么要多标注数据...', duration: 3000 },
  ])

  return { playSequence, onNewMushroom, onCorrectPrediction, onPoisonEaten }
}
```

#### 4.5 对话模板管理

创建 `backend/app/services/cat_brain.py`，管理不同场景的对话生成：

```python
import random

DIALOGUES = {
    "greeting": {
        0: ["喵！我是{name}...我对蘑菇一无所知，但是我很饿。教教我？"],
        3: ["又一个标本！让我穿上我的白大褂。我最近进步很大呢。"],
        5: ["欢迎回来。我已经分析了{n}个标本，准确率{acc}%。继续？"],
    },
    "predict_confident": [
        "我挺确定这个是{verdict}。{feature}让我想起之前见过的样本。",
    ],
    "predict_uncertain": [
        "嗯...这个好难。我猜是{verdict}？说实话我只有{conf}%的把握...",
    ],
    "correct": [
        "耶！我就知道！{feature}就是关键线索！",
        "又对一个！我在连胜中！({correct}/{total})",
    ],
    "wrong_false_negative": [
        "*脸变绿* 什么?! 那个有毒?! 我差点吃了它!",
        "呃! *戏剧性地倒下* 笔记：{feature}可能具有欺骗性！",
    ],
}

def get_dialogue(category: str, level: int = 0, **kwargs) -> str:
    templates = DIALOGUES.get(category, [])
    if isinstance(templates, dict):
        templates = templates.get(level, templates.get(0, ["..."]))
    template = random.choice(templates)
    return template.format(**kwargs)
```

### 里程碑检查
- [ ] 首页能看到猫咪图片（至少 idle 状态）
- [ ] 点击测试按钮能触发 mood 切换动画
- [ ] 对话气泡有打字机效果
- [ ] 动画状态机能按序列播放（curious → thinking → idle）

---

## 阶段 5：训练旅程（核心玩法）

### 目标
实现完整的训练循环：选择策略 → 猫预测 → 用户揭示 → 猫反应 → 模型重训练 → 升级。

### 步骤

#### 5.1 Onboarding 页面

在 `HomePage.tsx` 实现：
- 猫咪 idle 动画
- 输入猫咪名字
- 选择初始化策略（Random / D-Optimal）
- 选择分类器（Random Forest / Logistic / MLP）
- "开始训练！" 按钮 → 调用 `POST /api/training/start` → 跳转到 `/training`

#### 5.2 TrainingArena 组件（核心 UI）

`frontend/src/components/training/TrainingArena.tsx`：

布局分三栏：
1. **左侧**：蘑菇标本卡片（特征展示）+ 上传按钮
2. **中间**：猫咪 + 对话气泡
3. **底部**：进度条（已标注数、池剩余、准确率、等级进度）

#### 5.3 训练循环交互流程

```
用户点击 "下一个蘑菇" 按钮
    │
    ├─ 前端调用 POST /api/training/{id}/predict
    │  └─ 后端用策略从池中选样本，模型预测，返回预测结果
    │
    ├─ 前端播放 curious → thinking 动画
    │  └─ 显示猫的预测 + 置信度
    │
    ├─ 用户点击 "✅ 可食用" 或 "☠️ 有毒" 按钮
    │
    ├─ 前端调用 POST /api/training/{id}/reveal
    │  └─ 后端揭示真实标签，计算对错，可能重训练
    │
    ├─ 前端根据对错播放动画：
    │  ├─ 正确 → happy 动画 + 对话
    │  ├─ 错误（假阳性）→ confused 动画 + 对话
    │  └─ 错误（假阴性/吃了毒蘑菇）→ sick 动画 + 对话
    │
    ├─ 前端更新 XP、检查是否升级
    │  └─ 如果升级 → proud 动画 + 解锁通知
    │
    └─ 回到 idle，等待下一轮
```

#### 5.4 实现标本卡片

对于池中的表格数据（没有真实图片），展示一个程式化的"标本卡"：

```tsx
// frontend/src/components/shared/MushroomCard.tsx
interface Props {
  features: Record<string, string | number | null>
  specimenNumber: number
}

export default function MushroomCard({ features, specimenNumber }: Props) {
  return (
    <div className="bg-amber-50 rounded-xl p-4 border-2 border-amber-200 shadow">
      <h3 className="text-lg font-bold text-amber-800">🍄 标本 #{specimenNumber}</h3>
      <div className="grid grid-cols-2 gap-2 mt-3 text-sm">
        {Object.entries(features).map(([key, val]) => (
          val != null && (
            <div key={key}>
              <span className="text-gray-500">{key}:</span>{' '}
              <span className="font-medium">{String(val)}</span>
            </div>
          )
        ))}
      </div>
    </div>
  )
}
```

#### 5.5 批量模式

添加一个"批量查询"按钮，一次标注 50 个样本（自动揭示池中已知标签），快速推进训练：

```tsx
const handleBatchQuery = async () => {
  const res = await trainingApi.batchQuery(sessionId, 50)
  // 显示批量结果：标注了多少、准确率变化
  // 播放短动画
}
```

#### 5.6 学习曲线图表

使用 Recharts（轻量级 React 图表库）：

```bash
npm install recharts
```

```tsx
// frontend/src/components/training/LearningCurve.tsx
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface Props {
  data: { labels: number; accuracy: number }[]
}

export default function LearningCurve({ data }: Props) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="labels" label={{ value: "已标注样本数", position: "bottom" }} />
        <YAxis domain={[0.5, 1]} label={{ value: "准确率", angle: -90 }} />
        <Tooltip />
        <Line type="monotone" dataKey="accuracy" stroke="#f97316" strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  )
}
```

#### 5.7 等级进度和成就

- 进度条组件显示当前 XP / 下一级所需 XP
- 升级时弹出全屏通知（猫咪 proud 动画 + 新称号 + 新解锁功能）
- 成就列表页面展示已解锁的里程碑

#### 5.8 WebSocket 实时更新

用于模型重训练时的实时进度：

```typescript
// frontend/src/hooks/useWebSocket.ts
import { useEffect, useRef } from 'react'

export function useTrainingWS(sessionId: string, onMessage: (msg: any) => void) {
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!sessionId) return
    const ws = new WebSocket(`ws://localhost:8000/ws/training/${sessionId}`)
    ws.onmessage = (e) => onMessage(JSON.parse(e.data))
    wsRef.current = ws
    return () => ws.close()
  }, [sessionId])

  return wsRef
}
```

### 里程碑检查
- [ ] 能从首页选择配置并开始训练
- [ ] 训练循环完整：预测 → 揭示 → 反应 → 重训练
- [ ] 猫咪动画在正确/错误预测时有不同反应
- [ ] 学习曲线实时更新
- [ ] 达到准确率阈值时能升级
- [ ] 批量模式可用

---

## 阶段 6：专家模式（双裁判增强版）

### 目标
重建原始 Gradio 的双裁判系统，但用猫咪作为主持人。

### 步骤

#### 6.1 Detection 后端端点

`backend/app/routers/detection.py`：

```python
@router.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    img = Image.open(image.file).convert("RGB")

    # Judge A: Gemini 提取特征 → ML 分类器
    features = gemini_bridge.extract_features(img)
    prediction_a = learner.predict_single(preprocessor.transform(features))

    # Judge B: Gemini 直接视觉判断
    result_b = gemini_bridge.direct_judge(img)

    # 加权共识
    consensus = compute_consensus(prediction_a, result_b)

    return { "judge_a": prediction_a, "judge_b": result_b, "consensus": consensus }
```

#### 6.2 ExpertPage 前端

三栏布局：
- Judge A 面板（特征列表 + ML 判定）
- Judge B 面板（视觉分析 + 直接判定）
- 底部：猫咪综合评论 + 最终裁定

#### 6.3 猫咪解说

猫咪根据两个裁判的结果生成个性化评论，调用 `cat_brain.py` 的对话模板。

### 里程碑检查
- [ ] 上传蘑菇图片后能看到两个裁判的分析结果
- [ ] 猫咪给出综合评论
- [ ] 最终裁定正确显示

---

## 阶段 7：策略对比可视化

### 目标
运行多个 AL 策略的完整模拟，生成对比学习曲线。

### 步骤

#### 7.1 后端批量模拟端点

```python
@router.post("/api/training/compare")
async def compare_strategies(req: CompareRequest):
    """运行多种策略的完整 AL 模拟（非交互式，纯计算）。"""
    results = {}
    for strategy_key in req.strategies:
        session = ALSession(
            learner=get_learner(req.learner_type),
            strategy=get_strategy(strategy_key),
            X_pool=X_pool, y_pool=y_pool,
            X_initial=X_init, y_initial=y_init,
        )
        # 运行到预算耗尽
        while session.pool_remaining > 0 and session.labeled_count < req.budget:
            indices = session.query_batch(50)
            for idx in indices:
                session.label_sample(idx)
            session.retrain()

        results[strategy_key] = session.get_learning_curve()

    return results
```

> **注意**：这个端点计算量大（可能要跑几分钟），应该用 WebSocket 推送进度，或者用后台任务。

#### 7.2 ComparisonPage 前端

- 多选策略（勾选框）
- 选择分类器和标签预算
- 点击 "运行对比"
- 多条学习曲线叠加在一张图上（不同颜色）
- 底部表格：每种策略达到 90%/95% 准确率所需的标签数

### 里程碑检查
- [ ] 能选择至少两种策略运行对比
- [ ] 学习曲线图正确显示多条线
- [ ] 能看出不同策略的标签效率差异

---

## 阶段 8：Docker 容器化

### 目标
把整个应用打包成 Docker 容器，一条命令启动。

### 步骤

#### 8.1 创建 backend/Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY ml/ /app/ml/
RUN pip install /app/ml/
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install -r /app/backend/requirements.txt
COPY backend/ /app/backend/
COPY data/ /app/data/
COPY models/ /app/models/
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 8.2 创建 frontend/Dockerfile

```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 3000
```

#### 8.3 创建 docker-compose.yml

```yaml
version: "3.9"
services:
  backend:
    build: { context: ., dockerfile: backend/Dockerfile }
    ports: ["8000:8000"]
    env_file: .env
  frontend:
    build: { context: ./frontend }
    ports: ["3000:3000"]
    depends_on: [backend]
```

#### 8.4 测试

```bash
docker-compose up --build
# 访问 http://localhost:3000
```

### 里程碑检查
- [ ] `docker-compose up` 一条命令启动整个应用
- [ ] 前端能正常访问并与后端通信
- [ ] 所有功能在容器中正常工作

---

## 阶段 9：AWS 部署

### 目标
部署到 AWS EC2，可通过公网访问。

### 步骤

#### 9.1 启动 EC2 实例

- 选择 **t3.medium**（2 vCPU, 4GB RAM）
- AMI: Amazon Linux 2023 或 Ubuntu 22.04
- 安全组：开放 80、443、22 端口
- 分配弹性 IP

#### 9.2 安装 Docker

```bash
# SSH 到 EC2 后
sudo yum install docker git -y
sudo systemctl start docker
sudo usermod -aG docker ec2-user

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### 9.3 部署

```bash
git clone https://github.com/hallucijordan/mushroom-project.git
cd mushroom-project
cp deploy/.env.example .env
nano .env  # 填写 GEMINI_API_KEY

docker-compose -f docker-compose.prod.yml up -d --build
```

#### 9.4 验证

浏览器访问 `http://<EC2-公网-IP>` 确认应用正常运行。

#### 9.5 （可选）配置域名和 HTTPS

- Route 53 配置域名
- 用 Let's Encrypt / Certbot 配置 HTTPS

### 里程碑检查
- [ ] EC2 实例运行中
- [ ] 公网 IP 可以访问应用
- [ ] Gemini API 调用正常工作
- [ ] 训练旅程和专家模式在生产环境正常

---

## 阶段 10：美术资源与打磨

### 目标
补齐所有猫咪等级的美术资源，添加音效，打磨交互细节。

### 步骤

#### 10.1 生成全套猫咪素材

用 [02_cat_agent.md](02_cat_agent.md) 中的 AI prompt，为 6 个等级 × 9 种心情 = **54 张图**：

```
每个等级一个文件夹：
frontend/public/assets/cat/level0/  (idle, thinking, happy, sick, ...)
frontend/public/assets/cat/level1/  (同上，但猫拿着放大镜)
...
frontend/public/assets/cat/level5/  (教授装扮)
```

> **提示**：先用 DALL-E 3 / Midjourney 生成，然后用 remove.bg 去背景得到透明 PNG。

#### 10.2 配饰素材

为每个等级的独特配饰单独生成：
- 蘑菇贝雷帽
- 放大镜（Level 1）
- 野外笔记本（Level 2）
- 迷你白大褂（Level 3）
- 鉴定图鉴卷轴（Level 4）
- 毕业证书 + 金蘑菇徽章（Level 5）

#### 10.3 背景场景

- 森林空地（训练背景）
- 实验桌（专家模式背景）
- 羊皮纸地图（旅程页面背景）

#### 10.4 音效（可选）

添加简短的音效增强反馈感：
- 好奇的喵叫
- 开心的喵叫
- 中毒的呻吟
- 升级的号角声
- 咀嚼声

#### 10.5 UI 打磨

- 加载状态的骨架屏
- 页面切换动画
- 响应式适配（移动端）
- 深色模式支持（可选）
- 错误边界和友好的错误提示

### 里程碑检查
- [ ] 所有 6 个等级的猫咪都有对应外观
- [ ] 动画过渡流畅
- [ ] 整体 UI 视觉一致、美观

---

## 附录：前置知识速查

如果你对某些技术不熟悉，这里是快速上手指南：

### React + TypeScript

```
如果你会 Python，把组件理解为"函数"就行：
- 输入 = Props（参数）
- 输出 = JSX（界面描述）
- useState = 函数内部的变量
- useEffect = "在某个时机自动执行一段代码"
```

推荐教程：[React 官方教程](https://react.dev/learn)（约 2-3 小时）

### FastAPI

```
如果你会 Flask，FastAPI 几乎一样，多了：
- 自动生成 API 文档（/docs）
- 用 Pydantic 模型自动验证请求体
- 支持 async/await（Gemini API 调用很有用）
```

推荐教程：[FastAPI 官方文档](https://fastapi.tiangolo.com/tutorial/)（约 1-2 小时）

### Zustand（状态管理）

```
把它理解为一个"全局变量仓库"：
- create() 定义仓库
- useStore() 在任何组件中读写
- 比 Redux 简单 10 倍
```

### Docker

```
把它理解为"可复制的虚拟环境"：
- Dockerfile = 描述如何搭建环境
- docker-compose = 同时启动多个服务
- 你只需要会 build 和 up 两个命令
```

### Framer Motion（动画）

```
给任何 HTML 元素加上 <motion.div> 就能动画：
- animate={{ x: 100 }}  → 移动到 x=100
- transition={{ duration: 0.5 }}  → 0.5秒完成
- 就这么简单
```

---

## 开发时间估算

| 阶段 | 预计时间 | 前置条件 |
|---|---|---|
| 阶段 0: 环境准备 | 1-2 小时 | 无 |
| 阶段 1: ML 包 | 2-3 天 | Python 经验 |
| 阶段 2: FastAPI 后端 | 2-3 天 | 阶段 1 完成 |
| 阶段 3: React 基础 | 1-2 天 | 基础前端知识 |
| 阶段 4: 猫咪系统 | 2-3 天 | 阶段 3 完成 |
| 阶段 5: 训练旅程 | 3-4 天 | 阶段 2+4 完成 |
| 阶段 6: 专家模式 | 1-2 天 | 阶段 2+4 完成 |
| 阶段 7: 策略对比 | 1-2 天 | 阶段 5 完成 |
| 阶段 8: Docker | 半天 | 阶段 5+6 完成 |
| 阶段 9: AWS | 半天-1天 | 阶段 8 完成 |
| 阶段 10: 美术打磨 | 2-3 天 | 可以和其他阶段并行 |

**总计：约 15-23 天**（一个人全职开发），如果两个人并行（一人前端+一人后端/ML），可以压缩到 10-15 天。

> **优先级建议**：如果时间紧，先完成阶段 0-5（核心训练旅程）就已经有一个可演示的产品了。阶段 6-10 可以在后续迭代中补充。
