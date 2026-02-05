"""
Celery application configuration.
"""

from celery import Celery

from app.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "deepfake_detection",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.tasks"],
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=600,  # 10 minutes
    task_soft_time_limit=540,  # 9 minutes
    # Result settings
    result_expires=86400,  # 24 hours
    # Beat schedule (if needed)
    beat_schedule={},
)
