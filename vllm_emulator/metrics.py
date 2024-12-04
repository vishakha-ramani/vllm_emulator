import prometheus_client
from typing import List

class Metrics:

    def __init__(self, labelnames: List[str]):
        self.gauge_scheduler_running = prometheus_client.Gauge(
            name="vllm:num_requests_running",
            documentation="Number of requests currently running on GPU.",
            labelnames=labelnames)
        self.gauge_scheduler_waiting = prometheus_client.Gauge(
            name="vllm:num_requests_waiting",
            documentation="Number of requests waiting to be processed.",
            labelnames=labelnames)
        #   KV Cache Usage in %
        self.gauge_gpu_cache_usage = prometheus_client.Gauge(
            name="vllm:gpu_cache_usage_perc",
            documentation="GPU KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames)
        
        # request rate (req per min) for a server (deployment)
        # avg num of tokens per request (input + output)
    
        """
        Counter for request rate
        A counter is a cumulative metric that represents a single monotonically increasing counter 
        whose value can only increase or be reset to zero on restart. 
        For example, you can use a counter to represent the number of requests served, tasks completed, or errors.
        """
        self.counter_scheduler_total = prometheus_client.Counter(
            name="vllm:requests_count", # prometheus adds a _total suffix at the end
            documentation="Total number of requests received.",
            labelnames=labelnames)

        """
        Counter for total number of tokens generated (input + output)
        """
        self.counter_tokens_total = prometheus_client.Counter(
            name="vllm:tokens_count", # prometheus adds a _total suffix at the end
            documentation="Total number of tokens generated.",
            labelnames=labelnames)