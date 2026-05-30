#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage4 Visualizer Backend - FastAPI Application
添加解压功能和实时进度同步
"""

import os
import sys
import json
import uuid
import shutil
import asyncio
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ALGORITHM_DIR = PROJECT_ROOT / "algorithm"
sys.path.insert(0, str(ALGORITHM_DIR))

from core.sgy_extractor import (
    extract_sgy_headers, extract_sgy_float32, extract_sgy_components,
    reconstruct_sgy, reconstruct_float32_from_components
)

try:
    from core.demo_pipeline import SmallVolumeProcessor
except ImportError as e:
    print(f"[WARN] demo_pipeline not available: {e}")
    SmallVolumeProcessor = None

app = FastAPI(title="Stage4 Visualizer", version="2.0.0")

def _detect_device() -> str:
    """检测最佳计算设备"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)[:40]
            print(f"[INFO] GPU detected: {name}")
            return "cuda"
    except Exception as e:
        print(f"[INFO] CUDA check failed: {e}")
    return "cpu"

DEFAULT_DEVICE = _detect_device()
DEFAULT_INFERENCE_BATCH = 8192 if DEFAULT_DEVICE == "cuda" else 1
print(f"[INFO] Default device: {DEFAULT_DEVICE}, inference_batch: {DEFAULT_INFERENCE_BATCH}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

tasks: Dict[str, Dict[str, Any]] = {}
demo_processors: Dict[str, Any] = {}  # task_id -> SmallVolumeProcessor

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        self.active_connections[task_id] = websocket
    
    def disconnect(self, task_id: str):
        if task_id in self.active_connections:
            del self.active_connections[task_id]
    
    async def send_progress(self, task_id: str, data: dict):
        if task_id in self.active_connections:
            try:
                await self.active_connections[task_id].send_json(data)
            except:
                pass

manager = ConnectionManager()

class CompressConfig(BaseModel):
    feature_mode: str = "diagonal_causal_edge"
    target_mode: str = "residual"
    patch_shape: List[int] = [9, 17]
    inference_batch: int = DEFAULT_INFERENCE_BATCH
    device: str = DEFAULT_DEVICE

class DecompressRequest(BaseModel):
    output_filename: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Stage4 Visualizer API", "version": "2.0.0"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        task_id = str(uuid.uuid4())[:8]
        task_dir = UPLOAD_DIR / task_id
        task_dir.mkdir(exist_ok=True)
        
        ext = Path(file.filename).suffix.lower()
        is_sgy = ext in (".sgy", ".segy")
        
        original_path = task_dir / file.filename
        with open(original_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        file_size = os.path.getsize(original_path)
        
        task_info = {
            "id": task_id,
            "status": "uploaded",
            "filename": file.filename,
            "file_path": str(original_path),
            "file_size": file_size,
            "created_at": datetime.now().isoformat(),
            "progress": 0,
            "is_sgy": is_sgy,
        }
        
        tasks[task_id] = task_info
        
        if is_sgy:
            return {
                "task_id": task_id,
                "status": "uploaded",
                "message": "SEG-Y file saved, will convert on compression",
                "created_at": task_info["created_at"],
                "is_sgy": True,
                "file_size": file_size,
            }
        
        return {"task_id": task_id, "status": "uploaded", "message": "File uploaded", "created_at": task_info["created_at"]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compress/{task_id}")
async def start_compression(task_id: str, config: CompressConfig, background_tasks: BackgroundTasks):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    task["config"] = config.model_dump()
    task["status"] = "compressing"
    task["operation"] = "compress"
    task["progress"] = 0
    
    background_tasks.add_task(run_compression_with_progress, task_id, config)
    
    return {"task_id": task_id, "status": "compressing", "config": config.model_dump()}

@app.post("/api/decompress/{task_id}")
async def start_decompression(task_id: str, request: DecompressRequest, background_tasks: BackgroundTasks):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    bitstream_path = task.get("output", {}).get("bitstream_path")
    
    if not bitstream_path or not os.path.exists(bitstream_path):
        raise HTTPException(status_code=404, detail="Bitstream not found")
    
    task["status"] = "decompressing"
    task["operation"] = "decompress"
    task["progress"] = 0
    task["output_filename"] = request.output_filename or f"{task_id}_decompressed.bin"
    
    background_tasks.add_task(run_decompression_with_progress, task_id)
    
    return {"task_id": task_id, "status": "decompressing"}

class DecompressFileRequest(BaseModel):
    output_format: str = "bin"

@app.post("/api/decompress-file")
async def decompress_uploaded_file(
    file: UploadFile = File(...),
    sign_file: Optional[UploadFile] = None,
    mant_file: Optional[UploadFile] = None,
    output_format: str = "bin",
    background_tasks: BackgroundTasks = None,
):
    """独立解压：上传 .s4rc 文件直接解压"""
    try:
        task_id = str(uuid.uuid4())[:8]
        task_dir = UPLOAD_DIR / task_id
        task_dir.mkdir(exist_ok=True)
        output_dir = str(OUTPUT_DIR / task_id)
        os.makedirs(output_dir, exist_ok=True)
        
        s4rc_path = task_dir / file.filename
        with open(s4rc_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        task_info = {
            "id": task_id,
            "status": "uploaded",
            "filename": file.filename,
            "file_path": str(s4rc_path),
            "file_size": os.path.getsize(s4rc_path),
            "created_at": datetime.now().isoformat(),
            "progress": 0,
            "is_sgy": False,
            "output_format": output_format,
            "output_filename": f"{task_id}_decompressed.{output_format}",
            "bitstream_path": str(s4rc_path),
        }
        
        if sign_file:
            sign_path = task_dir / "sign.zlib"
            with open(sign_path, "wb") as f:
                shutil.copyfileobj(sign_file.file, f)
            task_info["sign_path"] = str(sign_path)
        
        if mant_file:
            mant_path = task_dir / "mant.zlib"
            with open(mant_path, "wb") as f:
                shutil.copyfileobj(mant_file.file, f)
            task_info["mant_path"] = str(mant_path)
        
        task_info["status"] = "decompressing"
        task_info["progress"] = 0
        task_info["operation"] = "decompress"
        
        tasks[task_id] = task_info
        
        background_tasks.add_task(run_standalone_decompression, task_id)
        
        return {"task_id": task_id, "status": "decompressing", "message": "Decompression started"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/progress/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    await manager.connect(websocket, task_id)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(task_id)

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    result = dict(tasks[task_id])
    result.pop("demo_trace", None)
    result.pop("demo_trace_list", None)
    return result

@app.get("/api/download/{task_id}")
async def download_result(task_id: str, file_type: str = "compressed"):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    is_sgy = task.get("is_sgy", False)
    
    if file_type == "compressed":
        output_path = task.get("output", {}).get("bitstream_path")
        filename = f"{task_id}_compressed.s4rc"
    else:
        output_path = task.get("decompressed_path")
        if is_sgy:
            filename = f"{task_id}_reconstructed.sgy"
        else:
            filename = task.get("output_filename", f"{task_id}_decompressed.bin")
    
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(output_path, filename=filename)

async def send_progress_update(task_id: str, progress: int, status: str, message: str = ""):
    data = {"type": "progress", "task_id": task_id, "progress": progress, "status": status, "message": message}
    await manager.send_progress(task_id, data)
    if task_id in tasks:
        tasks[task_id]["progress"] = progress
        tasks[task_id]["status"] = status

def _generate_demo_trace(task_id: str):
    """压缩完成后，预计算第一条 trace 的完整中间数据，供演示实时查询"""
    try:
        import torch
        from core.stage4_bridge import _infer_shape, load_stage4_model
        from stage4 import build_single_stage4_feature_causal_edge, target_symbol_for_coord

        task = tasks.get(task_id)
        if not task:
            return

        compress_file = task.get("compress_path") or task.get("file_path")
        if not compress_file or not os.path.exists(compress_file):
            return

        shape = tuple(task.get("float32_shape") or _infer_shape(compress_file))

        checkpoint_paths = [
            str(ALGORITHM_DIR / "outputs_tui_smoke" / "stage4" / "causal" / "checkpoint.pt"),
            str(ALGORITHM_DIR / "outputs_tui_heldout_materialized" / "stage4" / "causal" / "checkpoint.pt"),
        ]
        checkpoint_path = None
        for cp in checkpoint_paths:
            if os.path.exists(cp):
                checkpoint_path = cp
                break

        if not checkpoint_path:
            print(f"[WARN] No checkpoint found for demo trace, skipping")
            return

        config = task.get("config", {})
        feature_mode = config.get("feature_mode", "diagonal_causal_edge")
        target_mode = config.get("target_mode", "residual")
        patch_shape = tuple(config.get("patch_shape", [9, 17]))
        device = config.get("device", DEFAULT_DEVICE)

        print(f"[INFO] Loading model for demo trace: {checkpoint_path}")
        model, _ = load_stage4_model(checkpoint_path, device=device)
        model.eval()

        volume = np.memmap(compress_file, dtype=np.float32, mode='r').reshape(shape)
        exps = ((volume.view(np.uint32) >> 23) & 0xFF).astype(np.uint8).reshape(shape)

        # Scan first few traces to find one with non-zero data (avoid dead/padding traces)
        best_p, best_t, best_energy = 0, 0, 0.0
        scan_limit_p = min(shape[0], 5)
        scan_limit_t = min(shape[1], 20)
        for sp in range(scan_limit_p):
            for st in range(scan_limit_t):
                energy = float(np.mean(np.abs(volume[sp, st, :min(shape[2], 500)])))
                if energy > best_energy:
                    best_energy = energy
                    best_p, best_t = sp, st
        p, t = best_p, best_t
        print(f"[INFO] Selected demo trace: p={p}, t={t}, energy={best_energy:.6f}")

        trace_list = []
        max_s = min(shape[2], 1000)

        print(f"[INFO] Generating demo trace: {max_s} samples...")

        for s in range(max_s):
            coord = (p, t, s)

            v = float(volume[p, t, s])
            u32 = np.array([v], dtype=np.float32).view(np.uint32)[0]
            sign = int((u32 >> 31) & 0x1)
            exp_raw = int((u32 >> 23) & 0xFF)
            mant = int(u32 & 0x7FFFFF)

            feature = build_single_stage4_feature_causal_edge(
                volume=exps, coord=coord, patch_shape=patch_shape,
                target_mode=target_mode, feature_mode=feature_mode
            )

            with torch.inference_mode():
                logits = model(feature.to(device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            actual = int(target_symbol_for_coord(exps, coord, target_mode))
            predicted = int(np.argmax(probs))
            entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
            top5_idx = np.argsort(probs)[-5:][::-1]
            top5 = [{"symbol": int(i), "prob": float(probs[i])} for i in top5_idx]

            trace_list.append({
                "coord": list(coord),
                "original": v,
                "sign": sign,
                "exp_raw": exp_raw,
                "exp_value": exp_raw - 127,
                "mant": mant,
                "binary": f"{sign}|{format(exp_raw, '08b')}|{format(mant, '023b')}",
                "probabilities": probs.tolist(),
                "predicted_symbol": predicted,
                "actual_symbol": actual,
                "entropy": entropy,
                "top5": top5,
            })

            if (s + 1) % 200 == 0:
                print(f"[INFO] Demo trace: {s+1}/{max_s}")

        trace_dict = {tuple(item["coord"]): item for item in trace_list}
        task["demo_trace"] = trace_dict
        task["demo_trace_list"] = trace_list

        trace_path = str(OUTPUT_DIR / task_id / "demo_trace.json")
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace_list, f)

        print(f"[INFO] Demo trace saved: {len(trace_list)} samples -> {trace_path}")

    except Exception as e:
        print(f"[WARN] Demo trace generation failed: {e}")
        import traceback
        traceback.print_exc()


def run_compression_with_progress(task_id: str, config: CompressConfig):
    import zlib
    import json as json_module
    import base64
    from core.stage4_bridge import compress_data_with_progress, _infer_shape
    
    try:
        task = tasks[task_id]
        is_sgy = task.get("is_sgy", False)
        output_dir = str(OUTPUT_DIR / task_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # SEG-Y 提取 (延迟到压缩阶段，避免上传超时)
        if is_sgy and "compress_path" not in task:
            task["progress"] = 0
            task["status"] = "compressing"
            print("[INFO] Extracting SEG-Y headers and data...")
            
            sgy_path = task["file_path"]
            sgy_headers = extract_sgy_headers(str(sgy_path))
            sgy_meta = sgy_headers["meta"]
            
            task_dir = Path(task["file_path"]).parent
            headers_serializable = {
                "text_header": base64.b64encode(sgy_headers["text_header"]).decode("ascii"),
                "binary_header": base64.b64encode(sgy_headers["binary_header"]).decode("ascii"),
                "trace_headers": [
                    base64.b64encode(h).decode("ascii") for h in sgy_headers["trace_headers"]
                ],
                "meta": sgy_meta,
            }
            sgy_headers_path = task_dir / "sgy_headers.json"
            with open(sgy_headers_path, "w", encoding="utf-8") as f:
                json_module.dump(headers_serializable, f, ensure_ascii=False, default=str)
            
            print("[INFO] Extracting float32 data...")
            float32_2d = extract_sgy_float32(str(sgy_path), meta=sgy_meta, dtype=np.float32)
            
            profile_count = sgy_meta.get("profile_count")
            traces_per_profile = sgy_meta.get("traces_per_profile")
            if profile_count and traces_per_profile and profile_count * traces_per_profile == sgy_meta["trace_count"]:
                float32_3d = float32_2d.reshape(profile_count, traces_per_profile, sgy_meta["sample_count"])
            else:
                float32_3d = float32_2d.reshape(1, sgy_meta["trace_count"], sgy_meta["sample_count"])
                sgy_meta["profile_count"] = 1
                sgy_meta["traces_per_profile"] = sgy_meta["trace_count"]
            
            dat_path = task_dir / (Path(sgy_path).stem + ".dat")
            float32_3d.tofile(str(dat_path))
            
            task["sgy_headers_path"] = str(sgy_headers_path)
            task["sgy_meta"] = sgy_meta
            task["compress_path"] = str(dat_path)
            task["original_sgy_path"] = str(sgy_path)
            task["float32_shape"] = list(float32_3d.shape)
            task["float32_2d_shape"] = list(float32_2d.shape)
            
            print(f"[INFO] SEG-Y extracted: {sgy_meta.get('profile_count')}x{sgy_meta.get('traces_per_profile')}x{sgy_meta['sample_count']}")
        
        compress_file = task.get("compress_path", task["file_path"])

        if not os.path.exists(compress_file):
            raise FileNotFoundError(f"Compression file not found: {compress_file}")
        
        volume = np.memmap(compress_file, dtype=np.float32, mode='r')
        vol_shape = task.get("float32_shape") or _infer_shape(compress_file)
        task["float32_shape"] = list(vol_shape)
        
        u32 = volume.view(np.uint32)
        signs = ((u32 >> 31) & 0x1).astype(np.uint8)
        mants = (u32 & 0x7FFFFF).astype(np.uint32)
        
        packed_signs = np.packbits(signs.reshape(-1))
        sign_path = os.path.join(output_dir, "sign.zlib")
        with open(sign_path, 'wb') as f:
            f.write(zlib.compress(packed_signs.tobytes(), level=1))
        
        mant_bytes = np.zeros((mants.size, 3), dtype=np.uint8)
        mant_bytes[:, 0] = (mants & 0xFF).astype(np.uint8)
        mant_bytes[:, 1] = ((mants >> 8) & 0xFF).astype(np.uint8)
        mant_bytes[:, 2] = ((mants >> 16) & 0xFF).astype(np.uint8)
        mant_path = os.path.join(output_dir, "mant.zlib")
        with open(mant_path, 'wb') as f:
            f.write(zlib.compress(mant_bytes.tobytes(), level=1))
        
        task["sign_path"] = sign_path
        task["mant_path"] = mant_path
        volume._mmap.close()
        
        task["progress"] = 2
        task["status"] = "compressing"
        
        result = compress_data_with_progress(
            file_path=compress_file,
            config=config.model_dump(),
            output_dir=output_dir,
            progress_callback=None,
            shape=task.get("float32_shape"),
        )
        
        sign_size = os.path.getsize(sign_path)
        mant_size = os.path.getsize(mant_path)
        total_compressed = result["compressed_size"] + sign_size + mant_size
        
        if is_sgy:
            original_size = task.get("sgy_meta", {}).get("file_size_bytes", os.path.getsize(compress_file))
        else:
            original_size = result["original_size"]
        
        result["sign_bytes"] = sign_size
        result["mant_bytes"] = mant_size
        result["total_compressed_bytes"] = total_compressed
        result["total_compression_ratio"] = original_size / total_compressed if total_compressed > 0 else 0
        result["sign_mant_ratio"] = f"{sign_size + mant_size} bytes ({100*(sign_size+mant_size)/total_compressed:.1f}% of total)" if total_compressed > 0 else "N/A"
        result["original_size"] = original_size
        result["exponent_bytes"] = result["compressed_size"]
        result["compressed_size"] = total_compressed
        result["compression_ratio"] = result["total_compression_ratio"]
        
        print(f"[INFO] Compression complete: {original_size/1024/1024:.1f}MB -> {total_compressed/1024/1024:.1f}MB ({result['compression_ratio']:.2f}x)")

        # 先生成演示用的中间数据 trace，再标记完成，避免前端查到 completed 时缓存还未就绪
        print(f"[INFO] Starting demo trace generation for task {task_id}...")
        try:
            _generate_demo_trace(task_id)
            task["demo_ready"] = True
            print(f"[INFO] Demo trace ready for task {task_id}")
        except Exception as e:
            print(f"[WARN] Demo trace generation failed for task {task_id}: {e}")
            import traceback
            traceback.print_exc()
            task["demo_ready"] = False

        task["status"] = "completed"
        task["output"] = result
        task["progress"] = 100

    except Exception as e:
        import traceback
        traceback.print_exc()
        task["status"] = "failed"
        task["error"] = str(e)
        task["progress"] = 0

def run_decompression_with_progress(task_id: str):
    import zlib
    from core.stage4_bridge import decompress_data_with_progress
    
    try:
        task = tasks[task_id]
        bitstream_path = task.get("output", {}).get("bitstream_path")
        is_sgy = task.get("is_sgy", False)
        
        task["progress"] = 5
        task["status"] = "decompressing"
        
        exponent_output = str(OUTPUT_DIR / task_id / f"{task_id}_exps.bin")
        decompress_data_with_progress(
            bitstream_path=bitstream_path,
            output_path=exponent_output,
            progress_callback=None
        )
        
        sign_path = task.get("sign_path")
        mant_path = task.get("mant_path")
        
        if sign_path and mant_path and os.path.exists(sign_path) and os.path.exists(mant_path):
            shape = tuple(task.get("float32_shape", [1, 1]))
            total = int(np.prod(shape))
            
            with open(sign_path, 'rb') as f:
                packed_signs = np.frombuffer(zlib.decompress(f.read()), dtype=np.uint8)
            signs = np.unpackbits(packed_signs)[:total].reshape(shape).astype(np.uint8)
            
            with open(mant_path, 'rb') as f:
                mant_bytes = np.frombuffer(zlib.decompress(f.read()), dtype=np.uint8).reshape(-1, 3)
            mants = (mant_bytes[:total, 0].astype(np.uint32)
                     | (mant_bytes[:total, 1].astype(np.uint32) << 8)
                     | (mant_bytes[:total, 2].astype(np.uint32) << 16))
            mants = mants.reshape(shape)
            
            exps = np.fromfile(exponent_output, dtype=np.uint8).reshape(shape)
            
            task["progress"] = 90
            task["status"] = "decompressing"
            
            u32 = ((signs.astype(np.uint32) << 31)
                   | (exps.astype(np.uint32) << 23)
                   | (mants.astype(np.uint32) & 0x7FFFFF))
            float32_data = u32.view(np.float32)
            
            if is_sgy:
                import json as json_module
                import base64
                
                sgy_headers_path = task.get("sgy_headers_path")
                with open(sgy_headers_path, "r", encoding="utf-8") as f:
                    headers_serializable = json_module.load(f)
                
                headers = {
                    "text_header": base64.b64decode(headers_serializable["text_header"]),
                    "binary_header": base64.b64decode(headers_serializable["binary_header"]),
                    "trace_headers": [
                        base64.b64decode(h) for h in headers_serializable["trace_headers"]
                    ],
                    "meta": headers_serializable["meta"],
                }
                
                trace_count = headers["meta"]["trace_count"]
                sample_count = headers["meta"]["sample_count"]
                float32_2d = float32_data.reshape(trace_count, sample_count)
                
                output_path = str(OUTPUT_DIR / task_id / f"{task_id}_reconstructed.sgy")
                reconstruct_sgy(headers, float32_2d, output_path)
                task["decompressed_path"] = output_path
                task["decompressed_size"] = os.path.getsize(output_path)
            else:
                output_path = str(OUTPUT_DIR / task_id / task["output_filename"])
                float32_data.tofile(output_path)
                task["decompressed_path"] = output_path
                task["decompressed_size"] = os.path.getsize(output_path)
        else:
            output_path = str(OUTPUT_DIR / task_id / task["output_filename"])
            shutil.copy(exponent_output, output_path)
            task["decompressed_path"] = output_path
            task["decompressed_size"] = os.path.getsize(output_path)
        
        task["status"] = "decompress_completed"
        task["progress"] = 100
        
        compress_file = task.get("compress_path", task["file_path"])
        if os.path.exists(compress_file):
            if is_sgy and sign_path and mant_path:
                reconstructed_2d = extract_sgy_float32(task["decompressed_path"], dtype=np.float32)
                original = np.fromfile(compress_file, dtype=np.float32).reshape(reconstructed_2d.shape)
                task["verify_match"] = bool(np.array_equal(original, reconstructed_2d))
            else:
                original = np.fromfile(compress_file, dtype=np.float32)
                reconstructed = np.fromfile(task["decompressed_path"], dtype=np.float32)
                task["verify_match"] = bool(np.array_equal(original[:reconstructed.size], reconstructed))
        
        print(f"[INFO] Decompression complete: {task.get('decompressed_size', 0)/1024/1024:.1f}MB, verify={task.get('verify_match')}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        task["status"] = "failed"
        task["error"] = str(e)
        task["progress"] = 0

def run_standalone_decompression(task_id: str):
    """独立解压：上传 .s4rc + 可选 .zlib → 输出 .bin"""
    import zlib
    from core.stage4_bridge import decompress_data_with_progress
    from codec import read_bitstream
    
    try:
        task = tasks[task_id]
        bitstream_path = task["bitstream_path"]
        output_dir = str(OUTPUT_DIR / task_id)
        task["progress"] = 5
        task["status"] = "decompressing"
        
        # 读取 bitstream 头获取形状
        header, _ = read_bitstream(bitstream_path)
        shape = tuple(int(v) for v in header["shape"])
        task["float32_shape"] = list(shape)
        print(f"[INFO] Standalone decompress: shape={shape}")
        
        # Step 1: CNN 解压指数
        exponent_output = os.path.join(output_dir, f"{task_id}_exps.bin")
        decompress_data_with_progress(
            bitstream_path=bitstream_path,
            output_path=exponent_output,
            progress_callback=None
        )
        
        sign_path = task.get("sign_path")
        mant_path = task.get("mant_path")
        
        if sign_path and mant_path and os.path.exists(sign_path) and os.path.exists(mant_path):
            total = int(np.prod(shape))
            
            with open(sign_path, 'rb') as f:
                packed_signs = np.frombuffer(zlib.decompress(f.read()), dtype=np.uint8)
            signs = np.unpackbits(packed_signs)[:total].reshape(shape).astype(np.uint8)
            
            with open(mant_path, 'rb') as f:
                mant_bytes = np.frombuffer(zlib.decompress(f.read()), dtype=np.uint8).reshape(-1, 3)
            mants = (mant_bytes[:total, 0].astype(np.uint32)
                     | (mant_bytes[:total, 1].astype(np.uint32) << 8)
                     | (mant_bytes[:total, 2].astype(np.uint32) << 16))
            mants = mants.reshape(shape)
            
            exps = np.fromfile(exponent_output, dtype=np.uint8).reshape(shape)
            
            task["progress"] = 90
            
            u32 = ((signs.astype(np.uint32) << 31)
                   | (exps.astype(np.uint32) << 23)
                   | (mants.astype(np.uint32) & 0x7FFFFF))
            float32_data = u32.view(np.float32)
            
            output_path = os.path.join(output_dir, task["output_filename"])
            float32_data.tofile(output_path)
        else:
            output_path = os.path.join(output_dir, task["output_filename"])
            shutil.copy(exponent_output, output_path)
        
        task["decompressed_path"] = output_path
        task["decompressed_size"] = os.path.getsize(output_path)
        task["file_size"] = os.path.getsize(bitstream_path)
        task["has_aux_files"] = bool(sign_path and mant_path)
        task["status"] = "decompress_completed"
        task["progress"] = 100
        
        print(f"[INFO] Standalone decompression complete: {task['decompressed_size']/1024/1024:.1f}MB -> {output_path}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        task["status"] = "failed"
        task["error"] = str(e)
        task["progress"] = 0

# ====================== SEG-Y Parse Endpoint ======================

@app.post("/api/demo/parse-sgy")
async def parse_sgy_file(file: UploadFile = File(...)):
    """上传 SEG-Y 文件并解析元数据"""
    try:
        import shutil
        from core.sgy_parser import parse_sgy

        # 保存临时文件
        tmp_dir = BASE_DIR / "uploads" / "_sgy_tmp"
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = tmp_dir / file.filename
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = parse_sgy(str(tmp_path))

        # 保留文件用于后续热力图切片
        sgy_dir = BASE_DIR / "uploads" / "_sgy_store"
        sgy_dir.mkdir(exist_ok=True)
        stored_path = sgy_dir / f"{uuid.uuid4().hex[:8]}_{file.filename}"
        shutil.move(str(tmp_path), str(stored_path))
        result["stored_path"] = str(stored_path)

        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ====================== SEG-Y Heatmap Endpoint ======================

_sgy_meta_cache = {}

def _get_sgy_meta(file_path):
    if file_path not in _sgy_meta_cache:
        from core.sgy_parser import parse_sgy
        _sgy_meta_cache[file_path] = parse_sgy(file_path)
    return _sgy_meta_cache[file_path]

@app.get("/api/sgy/heatmap")
async def sgy_heatmap_slice(
    file_path: str = Query(...),
    inline_idx: int = Query(0),
    max_cols: int = Query(0),
    max_rows: int = Query(0)
):
    """返回指定剖面的二维热力图数据（支持降采样）"""
    import struct

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    try:
        meta = _get_sgy_meta(file_path)
    except:
        raise HTTPException(status_code=500, detail="解析失败")

    profile_count = meta.get("profile_count")
    traces_per_profile = meta.get("traces_per_profile")
    sample_count = meta.get("sample_count")
    format_code = meta.get("format_code", 5)
    trace_total = meta.get("trace_total_bytes", 0)
    bps = meta.get("bytes_per_sample", 4)

    if not all([profile_count, traces_per_profile, sample_count, trace_total]):
        raise HTTPException(status_code=500, detail="元数据不完整")
    if inline_idx < 0 or inline_idx >= profile_count:
        raise HTTPException(status_code=400, detail=f"inline_idx 超出范围 0-{profile_count-1}")

    # 降采样步长
    trace_step = max(1, traces_per_profile // max_cols) if max_cols > 0 else 1
    sample_step = max(1, sample_count // max_rows) if max_rows > 0 else 1
    out_cols = min(traces_per_profile, max_cols) if max_cols > 0 else traces_per_profile
    out_rows = min(sample_count, max_rows) if max_rows > 0 else sample_count

    offset = 3600 + inline_idx * traces_per_profile * trace_total

    def _ibm_to_ieee(raw4):
        bits = int.from_bytes(raw4, 'big')
        sign = -1 if (bits >> 31) & 1 else 1
        exp = (bits >> 24) & 0x7F
        mant = bits & 0x00FFFFFF
        if exp == 0 and mant == 0:
            return 0.0
        return sign * (mant / (1 << 24)) * (16.0 ** (exp - 64))

    grid = []
    vmin, vmax = float('inf'), float('-inf')

    with open(file_path, 'rb') as f:
        for t in range(0, traces_per_profile, trace_step):
            f.seek(offset + t * trace_total + 240)
            raw = f.read(sample_count * bps)
            if len(raw) < sample_count * bps:
                break

            row = []
            for s in range(0, sample_count, sample_step):
                pos = s * bps
                if format_code == 1:
                    v = _ibm_to_ieee(raw[pos:pos+4])
                elif format_code in (5,):
                    v = struct.unpack('>f', raw[pos:pos+4])[0]
                else:
                    v = float(int.from_bytes(raw[pos:pos+bps], 'big', signed=(format_code==2)))
                row.append(v)
                if v < vmin: vmin = v
                if v > vmax: vmax = v
            grid.append(row)
            if len(grid) >= out_cols:
                break

    return {
        "inline_idx": inline_idx,
        "shape": [len(grid), len(grid[0]) if grid else 0],
        "data": grid,
        "vmin": vmin,
        "vmax": vmax,
        "original_shape": [traces_per_profile, sample_count],
        "downsampled": (trace_step > 1 or sample_step > 1),
    }

# ====================== Demo APIs ======================

class DecomposeRequest(BaseModel):
    values: Optional[List[float]] = None
    file_path: Optional[str] = None
    coord: Optional[List[int]] = None
    task_id: Optional[str] = None

@app.post("/api/demo/decompose")
async def demo_decompose(req: DecomposeRequest):
    """Float32 位拆解 — 优先从缓存 demo trace 读取，否则从文件坐标读取"""
    try:
        coord = tuple(req.coord) if req.coord else None
        print(f"[API] /api/demo/decompose task_id={req.task_id} coord={coord} file_path={req.file_path}")

        # 1) 优先从 demo trace 缓存读取（压缩后预计算的真实中间数据）
        if req.task_id and req.task_id in tasks:
            has_trace = "demo_trace" in tasks[req.task_id]
            print(f"[API] task found, demo_trace exists={has_trace}")
            if has_trace:
                trace = tasks[req.task_id]["demo_trace"]
                if coord and coord in trace:
                    print(f"[API] decompose CACHE HIT for {coord}")
                    return {"decomposed": [trace[coord]]}
                else:
                    print(f"[API] decompose cache miss: coord {coord} not in trace (keys={list(trace.keys())[:3]}...)")
        else:
            print(f"[API] task not found or no task_id provided")

        # 2) 从文件坐标读取
        print(f"[API] decompose falling back to file read: {req.file_path}")
        if req.file_path and req.coord:
            shape_tuple = _infer_demo_shape(req.file_path)
            vol = np.memmap(req.file_path, dtype=np.float32, mode='r').reshape(shape_tuple)
            p, t, s = req.coord[0], req.coord[1], req.coord[2]
            if 0 <= p < shape_tuple[0] and 0 <= t < shape_tuple[1] and 0 <= s < shape_tuple[2]:
                v = float(vol[p, t, s])
            else:
                v = 0.0
            values = [v]
        elif req.values:
            values = list(req.values)
        else:
            raise HTTPException(status_code=400, detail="提供 values 或 file_path+coord")

        data = np.array(values, dtype=np.float32)
        u32 = data.view(np.uint32)
        signs = ((u32 >> 31) & 0x1).astype(np.uint8)
        exps = ((u32 >> 23) & 0xFF).astype(np.uint8)
        mants = (u32 & 0x7FFFFF).astype(np.uint32)

        results = []
        for i, v in enumerate(values):
            results.append({
                "original": float(v),
                "sign": int(signs[i]),
                "exp_raw": int(exps[i]),
                "exp_value": int(exps[i]) - 127,
                "mant": int(mants[i]),
                "binary": f"{signs[i]}|{format(int(exps[i]), '08b')}|{format(int(mants[i]), '023b')}"
            })
        return {"decomposed": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _infer_demo_shape(file_path: str):
    """推断 demo 数据文件的三维形状"""
    num = os.path.getsize(file_path) // 4
    # Try to load from .shape sidecar
    sp = file_path + '.shape'
    if os.path.exists(sp):
        import json as _j
        with open(sp) as f: return tuple(_j.load(f))
    # Fallback heuristics
    for s in [2001, 1500, 1000, 500]:
        if num % s == 0:
            mid = num // s
            for t in range(100, 2000):
                if mid % t == 0:
                    p = mid // t
                    if 1 <= p <= 2000:
                        return (p, t, s)
    return (1, num, 1)


# ====================== SEG-Y Load for Demo ======================

@app.post("/api/demo/load-sgy")
async def demo_load_sgy(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """上传 SEG-Y 文件 → 提取 float32 → 后台压缩 → 返回 task_id 和形状"""
    import base64
    import json as json_module
    import zlib

    task_id = str(uuid.uuid4())[:8]
    task_dir = UPLOAD_DIR / task_id
    task_dir.mkdir(exist_ok=True)
    output_dir = str(OUTPUT_DIR / task_id)
    os.makedirs(output_dir, exist_ok=True)

    # Save uploaded SEG-Y
    sgy_path = task_dir / file.filename
    with open(sgy_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Extract headers and float32
    print(f"[INFO] Demo SEG-Y extraction: {file.filename}")
    sgy_headers = extract_sgy_headers(str(sgy_path))
    sgy_meta = sgy_headers["meta"]

    headers_serializable = {
        "text_header": base64.b64encode(sgy_headers["text_header"]).decode("ascii"),
        "binary_header": base64.b64encode(sgy_headers["binary_header"]).decode("ascii"),
        "trace_headers": [base64.b64encode(h).decode("ascii") for h in sgy_headers["trace_headers"]],
        "meta": sgy_meta,
    }
    sgy_headers_path = task_dir / "sgy_headers.json"
    with open(sgy_headers_path, "w", encoding="utf-8") as f:
        json_module.dump(headers_serializable, f, ensure_ascii=False, default=str)

    float32_2d = extract_sgy_float32(str(sgy_path), meta=sgy_meta, dtype=np.float32)
    profile_count = sgy_meta.get("profile_count") or 1
    traces_per_profile = sgy_meta.get("traces_per_profile") or sgy_meta["trace_count"]
    if profile_count * traces_per_profile == sgy_meta["trace_count"]:
        float32_3d = float32_2d.reshape(profile_count, traces_per_profile, sgy_meta["sample_count"])
    else:
        float32_3d = float32_2d.reshape(1, sgy_meta["trace_count"], sgy_meta["sample_count"])
        sgy_meta["profile_count"] = 1; sgy_meta["traces_per_profile"] = sgy_meta["trace_count"]

    dat_path = str(task_dir / (sgy_path.stem + ".dat"))
    float32_3d.tofile(dat_path)
    shape = list(float32_3d.shape)
    shape_path = dat_path + ".shape"
    with open(shape_path, "w") as f:
        json_module.dump(shape, f)

    print(f"[INFO] Demo extraction done: shape={shape}, {os.path.getsize(dat_path)/1024/1024:.1f}MB")

    # Build task and start background compression
    task_info = {
        "id": task_id, "status": "extracting", "filename": file.filename,
        "file_path": str(sgy_path), "compress_path": dat_path,
        "sgy_meta": sgy_meta, "float32_shape": shape, "is_sgy": True,
        "sgy_headers_path": str(sgy_headers_path),
        "original_sgy_path": str(sgy_path),
        "progress": 0, "created_at": datetime.now().isoformat(),
        "demo_source": True,
    }

    # Sign/mant extraction (like in run_compression_with_progress)
    volume = np.memmap(dat_path, dtype=np.float32, mode='r')
    u32 = volume.view(np.uint32)
    signs = ((u32 >> 31) & 0x1).astype(np.uint8)
    mants = (u32 & 0x7FFFFF).astype(np.uint32)
    volume._mmap.close()

    packed_signs = np.packbits(signs.reshape(-1))
    sign_path = os.path.join(output_dir, "sign.zlib")
    with open(sign_path, 'wb') as f:
        f.write(zlib.compress(packed_signs.tobytes(), level=1))
    mant_bytes = np.zeros((mants.size, 3), dtype=np.uint8)
    mant_bytes[:, 0] = (mants & 0xFF).astype(np.uint8)
    mant_bytes[:, 1] = ((mants >> 8) & 0xFF).astype(np.uint8)
    mant_bytes[:, 2] = ((mants >> 16) & 0xFF).astype(np.uint8)
    mant_path = os.path.join(output_dir, "mant.zlib")
    with open(mant_path, 'wb') as f:
        f.write(zlib.compress(mant_bytes.tobytes(), level=1))
    task_info["sign_path"] = sign_path
    task_info["mant_path"] = mant_path

    tasks[task_id] = task_info

    # Background: run CNN compression
    config = CompressConfig(feature_mode="diagonal_causal_edge", target_mode="residual",
                            patch_shape=[9, 17], inference_batch=DEFAULT_INFERENCE_BATCH, device=DEFAULT_DEVICE)
    task_info["config"] = config.model_dump()
    task_info["status"] = "compressing"
    background_tasks.add_task(run_compression_with_progress, task_id, config)

    return {
        "task_id": task_id,
        "filename": file.filename,
        "shape": shape,
        "dat_path": dat_path,
        "file_size_mb": round(os.path.getsize(sgy_path) / 1024 / 1024, 1),
        "status": "compressing"
    }


@app.get("/api/demo/stats/{task_id}")
async def demo_stats(task_id: str):
    """获取演示任务的压缩统计"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    t = tasks[task_id]
    output = t.get("output", {})
    return {
        "task_id": task_id,
        "status": t.get("status", "unknown"),
        "progress": t.get("progress", 0),
        "demo_ready": t.get("demo_ready", False),
        "shape": t.get("float32_shape", []),
        "filename": t.get("filename", ""),
        "dat_path": t.get("compress_path", t.get("file_path", "")),
        "original_size": output.get("original_size", 0),
        "compressed_size": output.get("total_compressed_bytes", output.get("compressed_size", 0)),
        "exponent_bytes": output.get("exponent_bytes", 0),
        "sign_bytes": output.get("sign_bytes", 0),
        "mant_bytes": output.get("mant_bytes", 0),
        "compression_ratio": output.get("total_compression_ratio", output.get("compression_ratio", 0)),
        "bits_per_voxel": output.get("bits_per_voxel", 0),
    }


class PredictRequest(BaseModel):
    task_id: Optional[str] = None
    coord: List[int] = [0, 100, 1000]
    feature_mode: str = "diagonal_causal_edge"
    target_mode: str = "residual"
    patch_shape: List[int] = [9, 17]
    file_path: Optional[str] = None

@app.post("/api/demo/predict")
async def demo_predict(req: PredictRequest):
    """CNN 概率预测 — 优先从缓存 demo trace 读取，无缓存时尝试实时预测，最后 fallback mock"""
    coord = tuple(req.coord)
    print(f"[API] /api/demo/predict task_id={req.task_id} coord={coord}")

    # 1) 优先从 demo trace 缓存读取（压缩后预计算的真实中间数据）
    if req.task_id and req.task_id in tasks:
        has_trace = "demo_trace" in tasks[req.task_id]
        print(f"[API] predict task found, demo_trace exists={has_trace}")
        if has_trace:
            trace = tasks[req.task_id]["demo_trace"]
            if coord in trace:
                print(f"[API] predict CACHE HIT for {coord}")
                result = dict(trace[coord])
                result["mock"] = False
                return result
            else:
                print(f"[API] predict cache miss: coord {coord} not in trace")
    else:
        print(f"[API] predict task not found or no task_id")

    # 2) 尝试实时预测（加载模型，较慢）
    print(f"[API] predict falling back to real-time inference")
    try:
        from core.stage4_bridge import predict_probabilities

        file_path = req.file_path
        if not file_path and req.task_id and req.task_id in tasks:
            file_path = tasks[req.task_id].get("compress_path") or tasks[req.task_id].get("file_path")

        if file_path and os.path.exists(file_path):
            checkpoint_candidates = list(ALGORITHM_DIR.rglob("*/checkpoint.pt"))
            if checkpoint_candidates:
                cp = str(checkpoint_candidates[0])
                print(f"[INFO] Demo predict using checkpoint: {cp}")
                result = predict_probabilities(
                    file_path=file_path,
                    coord=tuple(req.coord),
                    checkpoint_path=cp,
                    patch_shape=tuple(req.patch_shape),
                    feature_mode=req.feature_mode,
                    target_mode=req.target_mode,
                    device=DEFAULT_DEVICE
                )
                if "error" not in result:
                    result["mock"] = False
                    print(f"[API] predict real-time inference success")
                    return result
                else:
                    print(f"[API] predict real-time inference returned error: {result.get('error')}")
        else:
            print(f"[API] predict no file_path available")
    except Exception as e:
        print(f"[WARN] Real predict failed, falling back to mock: {e}")

    # Fallback: high-quality mock
    import math
    center = req.coord[2] % 256  # deterministic from coord
    probs = np.zeros(256, dtype=np.float32)
    for i in range(256):
        dist = abs(i - center)
        probs[i] = math.exp(-dist * dist / (2 * 15 * 15)) + np.random.random() * 0.02
    probs = probs / probs.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5 = [{"symbol": int(i), "prob": float(probs[i])} for i in top5_idx]

    return {
        "mock": True,
        "coord": req.coord,
        "probabilities": probs.tolist(),
        "predicted_symbol": int(np.argmax(probs)),
        "actual_symbol": center,
        "entropy": float(entropy),
        "top5": top5
    }


class FeaturesRequest(BaseModel):
    task_id: Optional[str] = None
    coord: List[int] = [0, 100, 1000]
    patch_shape: List[int] = [9, 17]
    feature_mode: str = "diagonal_causal_edge"
    target_mode: str = "residual"
    file_path: Optional[str] = None

@app.post("/api/features")
async def demo_features(req: FeaturesRequest):
    """6通道特征提取 — 优先真实提取，无数据时返回 mock"""
    # Try real extraction first
    try:
        from core.stage4_bridge import extract_feature_data

        file_path = req.file_path
        if not file_path and req.task_id and req.task_id in tasks:
            file_path = tasks[req.task_id].get("compress_path") or tasks[req.task_id].get("file_path")

        if file_path and os.path.exists(file_path):
            result = extract_feature_data(
                file_path=file_path,
                coord=tuple(req.coord),
                patch_shape=tuple(req.patch_shape),
                feature_mode=req.feature_mode,
                target_mode=req.target_mode
            )
            if "error" not in result:
                result["mock"] = False
                return result
    except Exception as e:
        print(f"[WARN] Real feature extract failed, falling back to mock: {e}")

    # Fallback: mock feature patch
    patch_h, patch_w = req.patch_shape
    channels = []
    names = ["像素值 (Values)", "可用掩码 (Valid)", "因果掩码 (Causal)", "映射掩码 (Mapped)", "预测值 (Predicted)", "残差值 (Residual)"]
    colors = ["#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#e74c3c", "#1abc9c"]
    rng = np.random.RandomState(req.coord[0] * 10000 + req.coord[1] * 100 + req.coord[2])

    for i in range(6):
        if i == 4:
            data = np.full((patch_h, patch_w), 127.0)
        elif i in [1, 2, 3]:
            data = rng.random((patch_h, patch_w)) > 0.3
            data = data.astype(np.float32)
        else:
            data = rng.random((patch_h, patch_w))
        ch_min, ch_max, ch_mean = float(data.min()), float(data.max()), float(data.mean())
        channels.append({
            "name": names[i],
            "data": (data / 255.0 if data.max() > 1.0 else data).tolist(),
            "color": colors[i],
            "index": i,
            "min": ch_min,
            "max": ch_max,
            "mean": ch_mean
        })

    return {
        "mock": True,
        "coord": req.coord,
        "patch_shape": req.patch_shape,
        "channels": channels,
        "predicted_value": 127,
        "actual_value": rng.randint(0, 256),
        "context_pixels": rng.randint(0, 256, (5, 5)).tolist(),
        "target_symbol": rng.randint(0, 256)
    }


# ====================== Bit Stats ======================

_bit_stats_cache = {}

@app.get("/api/sgy/bit-stats")
async def sgy_bit_stats(file_path: str = Query(...)):
    """全文件位统计（numpy 向量化扫描）"""
    import math
    import numpy as np

    if file_path in _bit_stats_cache:
        return _bit_stats_cache[file_path]

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    try:
        meta = _get_sgy_meta(file_path)
    except:
        raise HTTPException(status_code=500, detail="解析失败")

    profile_count = meta.get("profile_count", 0)
    traces_per = meta.get("traces_per_profile", 0)
    sample_count = meta.get("sample_count", 0)
    trace_total = meta.get("trace_total_bytes", 0)

    if not all([profile_count, traces_per, sample_count, trace_total]):
        raise HTTPException(status_code=500, detail="元数据不完整")

    exp_hist = np.zeros(256, dtype=np.int64)
    sign_pos = 0; sign_neg = 0; sign_zero = 0
    exp_grid = []; sign_grid = []
    mant_bit_counts = np.zeros(23, dtype=np.int64)
    total = 0

    with open(file_path, 'rb') as f:
        for p in range(profile_count):
            exp_row = []; sign_row = []
            for t in range(traces_per):
                off = 3600 + (p * traces_per + t) * trace_total + 240
                f.seek(off)
                raw = f.read(sample_count * 4)
                if len(raw) < sample_count * 4: break

                arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4)
                high = arr[:, 3].astype(np.int64)
                low3 = arr[:, :3]

                # 指数直方图 + 均值
                exp_hist += np.bincount(high, minlength=256).astype(np.int64)
                exp_row.append(round(float(np.mean(high)), 2))

                # 符号统计
                is_neg = (high & 0x80).astype(bool)
                is_zero = (high == 0)
                is_pos = ~is_neg & ~is_zero
                cp = int(np.count_nonzero(is_pos))
                cn = int(np.count_nonzero(is_neg))
                cz = int(np.count_nonzero(is_zero))
                sign_pos += cp; sign_neg += cn; sign_zero += cz
                sign_row.append({"pos": cp, "neg": cn, "zero": cz})

                # 尾数位: 3 字节 × 每 bit 位计数 = 向量化
                for bi in range(8):
                    mant_bit_counts[bi] += np.count_nonzero((low3[:, 0].astype(np.int64) >> bi) & 1)
                for bi in range(8):
                    mant_bit_counts[8 + bi] += np.count_nonzero((low3[:, 1].astype(np.int64) >> bi) & 1)
                for bi in range(7):
                    mant_bit_counts[16 + bi] += np.count_nonzero((low3[:, 2].astype(np.int64) >> bi) & 1)

                total += sample_count
            exp_grid.append(exp_row)
            sign_grid.append(sign_row)

    # 指数统计
    exp_hist = exp_hist.tolist()
    mant_bit_counts = mant_bit_counts.tolist()
    exp_entropy = sum(-c / total * math.log2(c / total) for c in exp_hist if c > 0)
    exp_nz_bins = sum(1 for c in exp_hist if c > 0)
    exp_theory_lb = math.log2(max(1, exp_nz_bins))
    exp_nz_ratio = (total - exp_hist[0]) / total if total > 0 else 0

    # 符号统计
    sign_total = sign_pos + sign_neg + sign_zero
    sign_pos_ratio = sign_pos / sign_total if sign_total > 0 else 0
    sign_neg_ratio = sign_neg / sign_total if sign_total > 0 else 0
    sign_zero_ratio = sign_zero / sign_total if sign_total > 0 else 0

    # 尾数位平面熵
    mant_total_eff = total
    mant_per_bit = []
    for bit in range(23):
        ones = mant_bit_counts[bit]
        zeros = mant_total_eff - ones
        p1 = ones / mant_total_eff if mant_total_eff > 0 else 0
        p0 = zeros / mant_total_eff if mant_total_eff > 0 else 0
        e = 0.0
        if p0 > 0: e -= p0 * math.log2(p0)
        if p1 > 0: e -= p1 * math.log2(p1)
        mant_per_bit.append(round(e, 6))
    mant_cumulative = []
    cum = 0
    for e in mant_per_bit:
        cum += e
        mant_cumulative.append(round(cum, 6))
    mant_total = round(sum(mant_per_bit), 4)
    mant_upper = 23.0
    mant_ratio = round(mant_total / mant_upper, 4)
    # 分组: 高位 bit22-16(7bits), 中位 bit15-8(8bits), 低位 bit7-0(8bits)
    g_hi = round(sum(mant_per_bit[16:23]), 4)  # bits 16-22
    g_mid = round(sum(mant_per_bit[8:16]), 4)   # bits 8-15
    g_lo = round(sum(mant_per_bit[0:8]), 4)     # bits 0-7

    result = {
        "total_samples": total,
        "profile_count": profile_count,
        "traces_per_profile": traces_per,
        "exponent": {
            "histogram": exp_hist,
            "spatial_grid": exp_grid,
            "entropy": round(exp_entropy, 4),
            "theoretical_lower_bound": round(exp_theory_lb, 4),
            "non_zero_ratio": round(exp_nz_ratio, 4),
        },
        "sign": {
            "positive_count": sign_pos, "negative_count": sign_neg, "zero_count": sign_zero,
            "positive_ratio": round(sign_pos_ratio, 4), "negative_ratio": round(sign_neg_ratio, 4),
            "zero_ratio": round(sign_zero_ratio, 4),
            "spatial_grid": sign_grid,
        },
        "mantissa": {
            "per_bit_entropy": mant_per_bit,
            "cumulative_entropy": mant_cumulative,
            "total_entropy": mant_total,
            "theoretical_upper_bound": mant_upper,
            "entropy_ratio": mant_ratio,
            "group_high": g_hi,
            "group_mid": g_mid,
            "group_low": g_lo,
            "group_high_ratio": round(g_hi / mant_upper, 4),
            "group_mid_ratio": round(g_mid / mant_upper, 4),
            "group_low_ratio": round(g_lo / mant_upper, 4),
        },
    }
    _bit_stats_cache[file_path] = result
    return result


# ====================== Small Volume Demo APIs ======================

class UploadSGYResponse(BaseModel):
    task_id: str
    shape: List[int]
    sample_count: int
    filename: str

@app.post("/api/demo/upload-sgy")
async def demo_upload_sgy(file: UploadFile = File(...)):
    """上传 SGY → 提取 2×2×100 small_volume → 预计算概率"""
    if SmallVolumeProcessor is None:
        raise HTTPException(status_code=503, detail="Demo pipeline not available")

    try:
        task_id = str(uuid.uuid4())[:8]
        task_dir = UPLOAD_DIR / task_id
        task_dir.mkdir(exist_ok=True)

        sgy_path = task_dir / file.filename
        with open(sgy_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        print(f"[API] /api/demo/upload-sgy: {file.filename} -> {sgy_path}")

        processor = SmallVolumeProcessor(
            str(sgy_path),
            n_samples=100,
            device=DEFAULT_DEVICE
        )

        demo_processors[task_id] = processor

        return {
            "task_id": task_id,
            "shape": list(processor.shape),
            "sample_count": processor.total_voxels,
            "filename": file.filename
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class SampleIndexRequest(BaseModel):
    task_id: str
    sample_index: int = 0

@app.post("/api/demo/decompose")
async def demo_decompose(req: SampleIndexRequest):
    """Bit 拆解"""
    if req.task_id not in demo_processors:
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        processor = demo_processors[req.task_id]
        result = processor.decompose(req.sample_index)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/demo/features")
async def demo_features(req: SampleIndexRequest):
    """特征提取"""
    if req.task_id not in demo_processors:
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        processor = demo_processors[req.task_id]
        result = processor.features(req.sample_index)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/demo/predict")
async def demo_predict(req: SampleIndexRequest):
    """CNN 概率预测"""
    if req.task_id not in demo_processors:
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        processor = demo_processors[req.task_id]
        result = processor.predict(req.sample_index)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/demo/encode")
async def demo_encode(req: SampleIndexRequest):
    """Range Coding"""
    if req.task_id not in demo_processors:
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        processor = demo_processors[req.task_id]
        result = processor.encode(req.sample_index)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/demo/stats/{task_id}")
async def demo_stats(task_id: str):
    """压缩统计"""
    if task_id not in demo_processors:
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        processor = demo_processors[task_id]
        result = processor.stats()
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
