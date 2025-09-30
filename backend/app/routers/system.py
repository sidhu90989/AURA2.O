from fastapi import APIRouter


try:  # attempt to import root-level module
    from system_monitor import SystemMetricsCollector  # type: ignore
    collector = SystemMetricsCollector()
except Exception:  # pragma: no cover
    collector = None

router = APIRouter(prefix="/system", tags=["system"])

@router.get("/snapshot")
async def snapshot():
    if not collector:
        return {"error": "collector unavailable"}
    return {
        "cpu": collector.collect_cpu_usage(),
        "memory": collector.collect_memory_usage(),
        "disk": collector.collect_disk_usage(),
    }
