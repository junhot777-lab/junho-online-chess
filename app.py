# app.py
import json
from pathlib import Path
from typing import Dict, DefaultDict
from collections import defaultdict
import random

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chess

# ----------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = BASE_DIR / "junho_online.pt"

SAVE_EVERY_N_UPDATES = 20  # 20번 학습마다 한 번씩 저장

# ----------------------------------------------------------------------
# 간단한 "학습" 메모리 (진짜 딥러닝은 아니고, FEN별로 수 통계 저장)
# torch.save / load 로만 torch를 써서 Render 설치 실패 안 나게 함
# ----------------------------------------------------------------------
# memory[fen][move_uci] = count
MemoryType = DefaultDict[str, Dict[str, int]]
memory: MemoryType = defaultdict(dict)
update_counter = 0


def load_memory():
    global memory
    if MODEL_PATH.exists():
        try:
            obj = torch.load(MODEL_PATH, map_location="cpu")
            if isinstance(obj, dict):
                memory = defaultdict(dict, obj)
            print(f"[INFO] Loaded memory from {MODEL_PATH}")
        except Exception as e:
            print(f"[WARN] Failed to load memory: {e}")
    else:
        print("[INFO] No existing memory file, starting fresh.")


def save_memory():
    try:
        torch.save(dict(memory), MODEL_PATH)
        print(f"[INFO] Saved memory to {MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to save memory: {e}")


def record_move(fen: str, move_uci: str):
    """사용자가 둔 수를 FEN별로 카운트."""
    global update_counter
    table = memory.get(fen, {})
    table[move_uci] = table.get(move_uci, 0) + 1
    memory[fen] = table
    update_counter += 1
    if update_counter >= SAVE_EVERY_N_UPDATES:
        save_memory()
        update_counter = 0


def choose_move(fen: str, legal_uci_list):
    """
    FEN에 대해 과거에 자주 둔 수를 더 자주 선택.
    기억 없으면 랜덤.
    """
    table = memory.get(fen)
    if not table:
        return random.choice(legal_uci_list)

    # 기억된 수만 필터링
    weighted = []
    total_weight = 0
    for uci in legal_uci_list:
        w = table.get(uci, 1)  # 안 본 수라도 weight 1
        weighted.append((uci, w))
        total_weight += w

    r = random.uniform(0, total_weight)
    upto = 0.0
    for uci, w in weighted:
        upto += w
        if upto >= r:
            return uci
    # 혹시 모를 예외
    return random.choice(legal_uci_list)


# ----------------------------------------------------------------------
# FastAPI 앱 설정 + static 서빙
# ----------------------------------------------------------------------
app = FastAPI(title="Junho Online-Learning Chess")

# CORS (필요하면 나중에 수정)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /static 경로로 정적 파일 서빙 (css, js, 이미지 다 여기서 나감)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ----------------------------------------------------------------------
# 요청 스키마
# ----------------------------------------------------------------------
class TrainRequest(BaseModel):
    fen: str
    move: str  # UCI string, e.g. "e2e4", "g1f3"


class NextMoveRequest(BaseModel):
    fen: str


# ----------------------------------------------------------------------
# 라우트
# ----------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    load_memory()


@app.get("/")
async def index():
    """
    메인 페이지: static/index.html 반환
    """
    index_file = STATIC_DIR / "index.html"
    return FileResponse(str(index_file))


@app.post("/api/train")
async def api_train(req: TrainRequest):
    """
    사용자가 둔 수를 학습 메모리에 저장.
    """
    try:
        # FEN 검증 + 합법성 체크
        board = chess.Board(req.fen)
        move = chess.Move.from_uci(req.move)

        if move not in board.legal_moves:
            return {"ok": False, "reason": "illegal move"}

        record_move(req.fen, req.move)
        return {"ok": True}
    except Exception as e:
        # 클라이언트 쪽에서 그냥 무시해도 되는 에러
        return {"ok": False, "error": str(e)}


@app.post("/api/next_move")
async def api_next_move(req: NextMoveRequest):
    """
    서버가 다음 수를 선택해서 반환.
    FEN 기반으로 지금까지 유저가 자주 둔 수를 흉내냄.
    """
    try:
        board = chess.Board(req.fen)
    except Exception as e:
        return {"ok": False, "error": f"invalid fen: {e}", "move": None}

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return {"ok": True, "move": None}

    legal_uci = [m.uci() for m in legal_moves]
    best_uci = choose_move(req.fen, legal_uci)

    return {"ok": True, "move": best_uci}


# ----------------------------------------------------------------------
# 헬스체크 (Render ping용)
# ----------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}
