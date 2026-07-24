#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import re
import secrets
import shutil
import sys
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.labelcloud.utils import (
    decode_label_name, read_pcd_as_xyzrgb, write_json_atomic)
from tools.infer_labelcloud import (
    InferenceAmpConfigError,
    MODEL_CHOICES,
    config_use_dim,
    inference_detector_with_amp,
    none_if_blank,
    prediction_payload,
    validate_amp_dtype,
    write_prediction_crops,
)


def load_env_file(path: Path | None = None) -> None:
    """加载 .env 到环境变量；不覆盖已存在的系统环境变量。"""
    candidates = [Path.cwd() / ".env", ROOT / ".env"] if path is None else [Path(path)]
    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or key in os.environ:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ[key] = value


load_env_file()


LOGGER = logging.getLogger(__name__)


def safe_token(value: str, default_token: str, max_len: int = 80) -> str:
    token = Path(value).name
    token = re.sub(r"[^0-9A-Za-z_.-]+", "_", token).strip("._-")
    return (token[:max_len] or default_token)


def safe_stem(name: str, default_token: str) -> str:
    return safe_token(Path(name).stem, default_token)


def cleanup_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def valid_bearer_token(authorization: str | None, token: str) -> bool:
    if authorization is None:
        return False
    scheme, separator, credential = authorization.strip().partition(" ")
    return bool(separator) and scheme.lower() == "bearer" and secrets.compare_digest(credential.strip(), token)


def validate_scene_upload(upload: UploadFile) -> None:
    suffix = Path(upload.filename or "scene.pcd").suffix.lower()
    if suffix and suffix != ".pcd":
        raise HTTPException(status_code=400, detail="scene_pcd 只支持 .pcd 文件。")


def validate_score_thresh(value: float) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise HTTPException(status_code=400, detail="score_thresh 必须是 0 到 1 之间的有限数值。")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class ModelCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str, str, str, bool, str], dict] = {}

    def get(
        self,
        model_name: str,
        cfg_file: str | None,
        ckpt: str | None,
        device: str,
        amp: bool,
        amp_dtype: str,
    ) -> dict:
        cfg_file = none_if_blank(cfg_file)
        if cfg_file is None:
            raise HTTPException(status_code=500, detail="服务配置错误：请配置 INFER_CFG_FILE 为明确配置文件。")
        ckpt = none_if_blank(ckpt)
        if ckpt is None:
            raise HTTPException(status_code=500, detail="服务配置错误：请配置 INFER_CKPT 为明确 checkpoint 路径。")
        if ckpt == "auto":
            raise HTTPException(status_code=500, detail="服务配置错误：HTTP 推理不支持 INFER_CKPT=auto，请使用明确 checkpoint 路径。")
        ckpt_path = Path(ckpt)
        key = (model_name, cfg_file, ckpt_path.as_posix(), device, amp, amp_dtype)

        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

            from mmdet3d.apis import inference_detector, init_model

            model = init_model(cfg_file, ckpt_path.as_posix(), device=device)
            class_names = [
                decode_label_name(name)
                for name in model.dataset_meta.get("classes", [])
            ]
            context = {
                "model": model,
                "inference_detector": inference_detector,
                "class_names": class_names,
                "cfg_file": cfg_file,
                "ckpt_path": ckpt_path,
                "use_dim": config_use_dim(cfg_file),
                "device": device,
                "amp": amp,
                "amp_dtype": amp_dtype,
                "lock": threading.Lock(),
            }
            self._cache[key] = context
            return context


MODEL_CACHE = ModelCache()

app = FastAPI(title="MMDetection3D labelCloud HTTP 推理服务", version="0.1.0")


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex
    if request.url.path == "/v1/infer":
        token = os.environ.get("HTTP_AUTH_TOKEN", "").strip()
        authorization = request.headers.get("authorization")
        if token and not valid_bearer_token(authorization, token):
            return error_response(401, "unauthorized", "缺少或无效的 Authorization Bearer token。", request.state.request_id)

    response = await call_next(request)
    request_id = getattr(request.state, "request_id", None)
    if request_id and "x-request-id" not in response.headers:
        response.headers["X-Request-ID"] = request_id
    return response


def error_response(status_code: int, code: str, message: str, request_id: str | None = None) -> JSONResponse:
    payload = {"ok": False, "error": {"code": code, "message": message}}
    if request_id:
        payload["request_id"] = request_id
    headers = {"X-Request-ID": request_id} if request_id else {}
    if status_code == 401:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(payload, status_code=status_code, headers=headers)


def http_error_code(status_code: int) -> str:
    if status_code == 401:
        return "unauthorized"
    if status_code == 403:
        return "forbidden"
    if status_code == 404:
        return "not_found"
    if status_code == 405:
        return "method_not_allowed"
    if status_code == 422:
        return "validation_error"
    return "bad_request" if status_code < 500 else "server_error"


def validation_error_message(exc: RequestValidationError) -> str:
    messages = []
    for error in exc.errors():
        loc = ".".join(str(item) for item in error.get("loc", []) if item != "body")
        msg = error.get("msg", "参数无效")
        messages.append(f"{loc}: {msg}" if loc else str(msg))
    return "；".join(messages) or "请求参数校验失败。"


@app.exception_handler(HTTPException)
def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    message = str(exc.detail)
    code = http_error_code(exc.status_code)
    return error_response(exc.status_code, code, message, request_id)


@app.exception_handler(RequestValidationError)
def handle_validation_exception(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    return error_response(422, "validation_error", validation_error_message(exc), request_id)


@app.exception_handler(Exception)
def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    if is_cuda_oom(exc):
        device = os.environ.get("DEVICE", "cuda:0")
        LOGGER.error(
            "MMDetection3D HTTP 请求处理失败：CUDA 显存不足 request_id=%s",
            request_id,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        cleanup_cuda_memory(device)
        return error_response(503, "cuda_oom", "当前 GPU 显存不足，请检查 GPU 占用或资源配置。", request_id)
    LOGGER.error("MMDetection3D HTTP 请求处理失败", exc_info=(type(exc), exc, exc.__traceback__))
    return error_response(500, "internal_error", "服务内部错误，请查看服务日志。", request_id)


def cleanup_cuda_memory(device_name: str) -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return

    if not torch.cuda.is_available():
        return

    try:
        device = torch.device(device_name)
    except Exception:
        device = None

    try:
        if device is not None and device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        else:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        LOGGER.warning("CUDA 显存清理失败", exc_info=True)


def iter_exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_cuda_oom(exc: Exception) -> bool:
    for current in iter_exception_chain(exc):
        if current.__class__.__name__ == "OutOfMemoryError" and current.__class__.__module__.startswith("torch"):
            return True
        if "CUDA out of memory" in str(current):
            return True
    return False


def configured_model() -> tuple[str, str | None, str | None, str, bool, str]:
    model_name = os.environ.get("MODEL", "tr3d")
    if model_name not in MODEL_CHOICES:
        raise ValueError(f"不支持的模型: {model_name}")
    ckpt = none_if_blank(os.environ.get("INFER_CKPT"))
    amp = env_bool("INFER_AMP", False)
    amp_dtype = os.environ.get("INFER_AMP_DTYPE", "fp16").strip().lower()
    if amp:
        try:
            validate_amp_dtype(amp_dtype)
        except InferenceAmpConfigError as exc:
            raise HTTPException(status_code=500, detail=f"服务配置错误：{exc}") from exc
    return (
        model_name,
        none_if_blank(os.environ.get("INFER_CFG_FILE")),
        ckpt,
        os.environ.get("DEVICE", "cuda:0"),
        amp,
        amp_dtype,
    )


def write_result_zip(zip_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir).as_posix())


def run_single_inference(
    input_path: Path,
    output_dir: Path,
    *,
    model_name: str,
    cfg_file: str | None,
    ckpt: str | None,
    score_thresh: float,
    device: str,
    amp: bool,
    amp_dtype: str,
    crop: bool,
) -> Path:
    context = MODEL_CACHE.get(model_name, cfg_file, ckpt, device, amp, amp_dtype)
    scene_dir = output_dir / input_path.stem
    pred_file = scene_dir / "predictions.json"

    points = read_pcd_as_xyzrgb(input_path)[:, context["use_dim"]]
    with context["lock"]:
        try:
            result, _ = inference_detector_with_amp(
                context["inference_detector"],
                context["model"],
                points,
                amp=context["amp"],
                amp_dtype=context["amp_dtype"],
                device=context["device"],
            )
        except InferenceAmpConfigError as exc:
            raise HTTPException(status_code=500, detail=f"服务配置错误：{exc}") from exc
        payload = prediction_payload(
            input_path.stem,
            result.pred_instances_3d,
            context["class_names"],
            score_thresh,
        )

    crop_dir = None
    if crop:
        crop_dir = write_prediction_crops(output_dir, input_path.stem, input_path, payload)

    write_json_atomic(pred_file, payload)
    summary_path = output_dir / "summary.json"
    write_json_atomic(summary_path, {
        "model": model_name,
        "cfg_file": context["cfg_file"],
        "ckpt": context["ckpt_path"].as_posix(),
        "use_dim": context["use_dim"],
        "class_names": list(context["class_names"]),
        "input": input_path.name,
        "score_thresh": score_thresh,
        "amp": bool(context["amp"]),
        "amp_dtype": context["amp_dtype"] if context["amp"] else None,
        "scenes": [{
            "frame_id": input_path.stem,
            "input_file": input_path.name,
            "num_predictions": len(payload["objects"]),
            "scene_dir": input_path.stem,
            "predictions_json": f"{input_path.stem}/predictions.json",
            "crop_dir": crop_dir,
        }],
    })

    try:
        import torch
    except ImportError:
        torch = None
    del points, result
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    zip_path = output_dir.parent / f"{input_path.stem}_mmdetection3d.zip"
    write_result_zip(zip_path, output_dir)
    return zip_path


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "mmdetection3d-labelcloud-http"}


@app.post("/v1/infer", response_model=None)
def infer(
    request: Request,
    background_tasks: BackgroundTasks,
    scene_pcd: UploadFile = File(...),
    score_thresh: float = Form(float(os.environ.get("SCORE_THRESH", "0.3"))),
    crop: bool = Form(env_bool("HTTP_CROP", False)),
    request_id: str | None = Form(None),
) -> Response:
    job_id = safe_token(request_id or getattr(request.state, "request_id", None) or uuid.uuid4().hex, "scene")
    request.state.request_id = job_id

    validate_scene_upload(scene_pcd)
    validate_score_thresh(score_thresh)

    work_dir = Path(tempfile.mkdtemp(prefix=f"mmd3d_{job_id}_"))
    cleanup_device = os.environ.get("DEVICE", "cuda:0")
    oom_response: JSONResponse | None = None
    try:
        input_dir = work_dir / "input"
        output_dir = work_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / f"{safe_stem(scene_pcd.filename or 'scene', 'scene')}.pcd"

        with input_path.open("wb") as fh:
            shutil.copyfileobj(scene_pcd.file, fh)

        model, cfg_file, ckpt, device, amp, amp_dtype = configured_model()
        cleanup_device = device
        zip_path = run_single_inference(
            input_path,
            output_dir,
            model_name=model,
            cfg_file=cfg_file,
            ckpt=ckpt,
            score_thresh=score_thresh,
            device=device,
            amp=amp,
            amp_dtype=amp_dtype,
            crop=crop,
        )
        response_zip_path = work_dir / f"{job_id}_mmdetection3d.zip"
        if zip_path != response_zip_path:
            zip_path.replace(response_zip_path)
            zip_path = response_zip_path
    except HTTPException:
        cleanup_dir(work_dir)
        raise
    except Exception as exc:
        cleanup_dir(work_dir)
        if is_cuda_oom(exc):
            LOGGER.error(
                "MMDetection3D HTTP 请求处理失败：CUDA 显存不足 request_id=%s",
                job_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            oom_response = error_response(503, "cuda_oom", "当前 GPU 显存不足，请检查 GPU 占用或资源配置。", job_id)
        else:
            raise

    if oom_response is not None:
        cleanup_cuda_memory(cleanup_device)
        return oom_response

    background_tasks.add_task(cleanup_dir, work_dir)
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=zip_path.name,
        headers={"X-Request-ID": job_id},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 MMDetection3D labelCloud HTTP 推理服务。")
    parser.add_argument("--host", default=os.environ.get("HTTP_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("HTTP_PORT", "8011")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import uvicorn

    uvicorn.run(
        "tools.http_infer_labelcloud:app",
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
