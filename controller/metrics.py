from prometheus_client import Counter, Gauge

SATURATION = Gauge("adaptive_saturation_score", "Current saturation score")
PENDING = Gauge("adaptive_pending_jobs", "Pending jobs")
RUNNING = Gauge("adaptive_running_jobs", "Running jobs")
MAX_JOBS = Gauge("adaptive_target_max_jobs", "Target MaxJobsPU")
PRIORITY = Gauge("adaptive_priority_weight_fs", "PriorityWeightFairshare proxy")
ACTIONS = Gauge("adaptive_action_changed", "1 when a config action changed state")
LOOP_ERRORS = Counter("adaptive_loop_errors_total", "Total number of control-loop iterations that raised")
