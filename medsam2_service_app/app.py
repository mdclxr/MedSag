import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from medsam2_service_app.predictor import MedSAM2GPUService


logger = logging.getLogger(__name__)


def _pick_first_existing(*paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return paths[0] if paths else ''


def _service_settings() -> dict:
    repo_path = os.path.abspath(os.environ.get('MEDSAM2_REPO', '/home/mdc/MedSAM2'))
    checkpoint_path = os.environ.get(
        'MEDSAM2_CHECKPOINT',
        _pick_first_existing(
            os.path.join(repo_path, 'checkpoints', 'MedSAM2_latest.pt'),
            os.path.join(repo_path, 'checkpoints', 'MedSAM2_2411.pt'),
        ),
    )
    config_path = os.environ.get(
        'MEDSAM2_CONFIG',
        _pick_first_existing(
            os.path.join(repo_path, 'sam2', 'configs', 'sam2.1_hiera_t512.yaml'),
            os.path.join(repo_path, 'efficient_track_anything', 'configs', 'efficienttam_s_512x512.yaml'),
        ),
    )
    device = os.environ.get('MEDSAM2_DEVICE', 'cuda:0')
    return {
        'repo_path': repo_path,
        'checkpoint_path': os.path.abspath(checkpoint_path),
        'config_path': os.path.abspath(config_path),
        'device': device,
    }


class SegmentRequest(BaseModel):
    volume_path: str
    output_path: str
    points: list[list[float]] = Field(default_factory=list)
    boxes: list[list[float]] = Field(default_factory=list)
    prompt_mask_path: str | None = None
    prompt_mask_axis: str = 'z'
    prompt_mask_slice_idx: int | None = None
    prompt_mask_label: int | None = None
    label_value: int = 1
    ww: float = 1500.0
    wl: float = -600.0


class TaskStore:
    def __init__(self):
        self._tasks = {}
        self._lock = threading.Lock()

    def create(self, payload: dict) -> str:
        task_id = uuid.uuid4().hex
        with self._lock:
            self._tasks[task_id] = {
                'task_id': task_id,
                'status': 'pending',
                'progress': 0,
                'message': 'Task created',
                'payload': payload,
                'nii_path': None,
                'error': None,
                'created_at': time.time(),
            }
        return task_id

    def update(self, task_id: str, **fields):
        with self._lock:
            if task_id not in self._tasks:
                return
            self._tasks[task_id].update(fields)

    def get(self, task_id: str) -> dict | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return dict(task) if task else None


def create_app() -> FastAPI:
    app = FastAPI(title='VisioFirm MedSAM2 GPU Service')
    tasks = TaskStore()
    executor = ThreadPoolExecutor(max_workers=1)
    service = MedSAM2GPUService(**_service_settings())

    def _run_task(task_id: str, payload: SegmentRequest):
        tasks.update(task_id, status='running', progress=10, message='Loading CT volume on GPU')
        try:
            output_path = service.segment_volume(
                volume_path=payload.volume_path,
                output_path=payload.output_path,
                points=payload.points,
                boxes=payload.boxes,
                prompt_mask_path=payload.prompt_mask_path,
                prompt_mask_axis=payload.prompt_mask_axis,
                prompt_mask_slice_idx=payload.prompt_mask_slice_idx,
                prompt_mask_label=payload.prompt_mask_label,
                label_value=payload.label_value,
                ww=payload.ww,
                wl=payload.wl,
            )
            tasks.update(
                task_id,
                status='done',
                progress=100,
                message='Segmentation completed',
                nii_path=output_path,
            )
        except Exception as exc:
            logger.exception("MedSAM2 segmentation failed for task %s", task_id)
            tasks.update(
                task_id,
                status='error',
                progress=100,
                message=str(exc),
                error=str(exc),
            )

    @app.get('/health')
    def health():
        state = service.health()
        return {'available': state['ready'], **state}

    @app.post('/segment')
    def segment(req: SegmentRequest):
        if not os.path.isfile(req.volume_path):
            raise HTTPException(status_code=404, detail=f"Volume not found: {req.volume_path}")
        task_id = tasks.create(req.model_dump())
        executor.submit(_run_task, task_id, req)
        return {'success': True, 'task_id': task_id, 'status': 'pending'}

    @app.get('/tasks/{task_id}')
    def task_status(task_id: str):
        task = tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail='Task not found')
        return task

    return app


app = create_app()
