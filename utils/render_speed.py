import time


def format_render_speed_summary(label, frames, render_ms_total, total_seconds):
    if frames <= 0:
        return f"[Speed][{label}] frames=0"

    render_ms_per_frame = render_ms_total / frames
    render_fps = 1000.0 / render_ms_per_frame if render_ms_per_frame > 0 else float("inf")
    total_ms_per_frame = total_seconds * 1000.0 / frames
    total_fps = frames / total_seconds if total_seconds > 0 else float("inf")

    return (
        f"[Speed][{label}] frames={frames}\n"
        f"  render: {render_ms_per_frame:.2f} ms/frame, {render_fps:.2f} FPS\n"
        f"  total : {total_ms_per_frame:.2f} ms/frame, {total_fps:.2f} FPS"
    )


class RenderSpeedMeter:
    def __init__(self, label):
        self.label = label
        self.frames = 0
        self.render_ms_total = 0.0
        self.total_start = None

    def start(self):
        self._cuda_synchronize()
        self.total_start = time.perf_counter()

    def render(self, render_func, *args, **kwargs):
        torch = self._get_torch()
        if torch is not None and torch.cuda.is_available():
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            result = render_func(*args, **kwargs)
            ender.record()
            ender.synchronize()
            render_ms = starter.elapsed_time(ender)
        else:
            start = time.perf_counter()
            result = render_func(*args, **kwargs)
            render_ms = (time.perf_counter() - start) * 1000.0

        self.frames += 1
        self.render_ms_total += render_ms
        return result

    def update_progress(self, progress):
        if self.frames <= 0:
            return
        render_ms_per_frame = self.render_ms_total / self.frames
        render_fps = 1000.0 / render_ms_per_frame if render_ms_per_frame > 0 else float("inf")
        progress.set_postfix({"render_fps": f"{render_fps:.2f}"})

    def finish(self):
        self._cuda_synchronize()
        total_seconds = time.perf_counter() - self.total_start if self.total_start is not None else 0.0
        print(format_render_speed_summary(self.label, self.frames, self.render_ms_total, total_seconds))

    @staticmethod
    def _get_torch():
        try:
            import torch
            return torch
        except ImportError:
            return None

    def _cuda_synchronize(self):
        torch = self._get_torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
