from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total number of inference requests"
)

INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency"
)

DETECTION_COUNT = Counter(
    "detections_total",
    "Total number of detected objects"
)
