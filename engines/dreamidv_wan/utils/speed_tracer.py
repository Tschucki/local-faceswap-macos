import time
import logging
import torch

class SpeedTracer:
    def __init__(self):
        self.timers = {}
        self.step_timers = {}
        self.start_times = {}

    def start(self, name):
        self.start_times[name] = time.perf_counter()

    def stop(self, name):
        if name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[name]
            self.timers[name] = self.timers.get(name, 0.0) + elapsed
            return elapsed
        return 0.0

    def start_step(self, step_idx):
        self.step_timers[step_idx] = {
            'start': time.perf_counter(),
            'dit': 0.0,
            'offload': 0.0,
            'scheduler': 0.0
        }

    def log_step_metric(self, step_idx, metric, duration):
        if step_idx in self.step_timers:
            self.step_timers[step_idx][metric] = duration

    def stop_step(self, step_idx):
        if step_idx in self.step_timers:
            elapsed = time.perf_counter() - self.step_timers[step_idx]['start']
            self.step_timers[step_idx]['total'] = elapsed
            
            metrics = self.step_timers[step_idx]
            logging.info(
                f"[SPEED_STEP:{step_idx}] total={elapsed:.3f}s | "
                f"dit={metrics['dit']:.3f}s | "
                f"offload={metrics['offload']:.3f}s | "
                f"scheduler={metrics['scheduler']:.3f}s"
            )

    def report(self):
        logging.info("================ SPEED PROFILE REPORT ================")
        for name, duration in self.timers.items():
            logging.info(f"  Phase [{name}]: {duration:.3f} seconds")
        
        step_totals = [s['total'] for s in self.step_timers.values() if 'total' in s]
        if step_totals:
            avg_step = sum(step_totals) / len(step_totals)
            logging.info(f"  Average Denoising Step: {avg_step:.3f} seconds/iteration")
        logging.info("======================================================")

tracer = SpeedTracer()
