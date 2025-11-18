# app.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ====== 기본 설정 ======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "junho_online.pt"
TRAIN_SAVE_INTERVAL = 50

# ====== 체스 인코딩 ======

PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]

def board_to_tensor(fen: str) -> torch.Tensor:
    """
    FEN -> (12, 8, 8) tensor
    white 6채널, black 6채널
    """
    board = chess.Board(fen)
    planes = torch.zeros((12, 8, 8), dtype=torch.float32)

    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        offset = 0 if piece.color == chess.WHITE else 6
        idx = offset + PIECE_TYPES.index(piece.piece_type)
        planes[idx, row, col] = 1.0

    return planes  # (12,8,8)

def move_to_index(move_uci: str) -> int:
    """
    "e2e4" -> idx (0~4095)
    """
    move = chess.Move.from_uci(move_uci)
    return move.from_square * 64 + move.to_square

def index_to_legal_move(idx: int, board: chess.Board):
    """
    index -> Move (합법이면 리턴, 아니면 None)
    """
    from_sq = idx // 64
    to_sq = idx % 64
    move = chess.Move(from_sq, to_sq)
    if move in board.legal_moves:
        return move
    return None

# ====== 모델 정의 ======

class JunhoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 64 * 64)

    def forward(self, x):
        # x: (B, 12, 8, 8)
        x = x.view(x.size(0), -1)  # (B, 768)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # (B, 4096)
        return x

# ====== 전역 모델 / 옵티마이저 ======

model = JunhoNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_steps = 0

def load_model():
    global model, optimizer, train_steps
    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        train_steps = ckpt.get("steps", 0)
        print("모델 로드 완료:", MODEL_PATH, "steps:", train_steps)
    except Exception as e:
        print("저장된 모델 없음 / 로드 실패, 새로 시작:", e)

def save_model():
    global model, optimizer, train_steps
    torch.save(
        {
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "steps": train_steps,
        },
        MODEL_PATH,
    )
    print("모델 저장:", MODEL_PATH, "steps:", train_steps)

load_model()

# ====== 학습 함수 ======

def train_one(fen: str, move_uci: str):
    global model, optimizer, train_steps

    board = chess.Board(fen)
    try:
        move = chess.Move.from_uci(move_uci)
    except Exception:
        return

    # 실제 합법 수 아니면 학습 안 함
    if move not in board.legal_moves:
        return

    x = board_to_tensor(fen).unsqueeze(0).to(DEVICE)  # (1,12,8,8)
    y_idx = move_to_index(move_uci)
    y = torch.tensor([y_idx], dtype=torch.long, device=DEVICE)  # (1,)

    model.train()
    logits = model(x)
    loss = F.cross_entropy(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_steps += 1
    if train_steps % TRAIN_SAVE_INTERVAL == 0:
        save_model()

# ====== 수 선택 함수 ======

def pick_move(fen: str) -> str:
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return ""

    x = board_to_tensor(fen).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(x)[0]  # (4096,)

    # 합법 수 마스크
    mask = torch.zeros_like(logits)
    for mv in legal_moves:
        idx = mv.from_square * 64 + mv.to_square
        mask[idx] = 1.0

    probs = F.softmax(logits, dim=0) * mask
    if probs.sum().item() <= 0:
        # 아직 학습 거의 안된 상태일 때
        move = legal_moves[0]
        return move.uci()

    probs = probs / probs.sum()
    idx = torch.multinomial(probs, 1).item()
    move = index_to_legal_move(idx, board)
    if move is None:
        move = legal_moves[0]
    return move.uci()

# ====== FastAPI 설정 ======

app = FastAPI()

# CORS (필요시)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

# 정적 파일 (index.html 서빙)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

# ====== API 스키마 ======

class TrainRequest(BaseModel):
    fen: str
    move: str   # 사용자가 둔 수 (uci)

class MoveRequest(BaseModel):
    fen: str

class MoveResponse(BaseModel):
    move: str

# ====== 엔드포인트 ======

@app.post("/api/train")
def api_train(req: TrainRequest):
    """
    사용자가 한 수를 학습
    """
    train_one(req.fen, req.move)
    return {"status": "ok"}

@app.post("/api/next_move", response_model=MoveResponse)
def api_next_move(req: MoveRequest):
    """
    현재 FEN에서 엔진 수 선택
    """
    mv = pick_move(req.fen)
    return MoveResponse(move=mv)
