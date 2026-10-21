"""Human collection with complete physical landing and native-resolution previews."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
import numpy as np
from .recording import EpisodeRecorder


class DemonstrationSession:
    def __init__(self, environment, writer, seed=0, max_steps=3000):
        if environment.name not in ("pusht", "pushbox2d", "pushbox3d"):
            raise ValueError("Unknown push environment.")
        if max_steps < 1:
            raise ValueError("max_steps must be positive.")
        self.environment, self.writer = environment, writer
        self.seed, self.max_steps = seed, max_steps
        self.saved = 0
        self.active = False
        self.controller = environment.teleop_agent()
        self.last_result = None
        self._save_executor = self._save_future = self._save_context = None
        self._save_started = None
        self.reset()

    @property
    def spatial(self):
        return self.environment.name == "pushbox3d"

    @property
    def saving(self):
        return self._save_future is not None

    @property
    def saving_elapsed(self):
        return perf_counter() - self._save_started if self.saving else 0.0

    def enable_background_saving(self):
        """Keep encoding and disk writes off the GUI thread; one episode at a time."""
        if self._save_executor is None:
            self._save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pushenv-save")

    def reset(self):
        if self.saving:
            raise RuntimeError("Cannot reset while an episode is being saved.")
        if hasattr(self, "recorder"):
            self.recorder.close()
        # Retire the old episode before a potentially slow/reset-interrupted
        # physics call. Ctrl+C must never save the committed rows again.
        self.recorder = EpisodeRecorder(self.environment, buffer_dir=getattr(self.writer, "root", None))
        self.active = self.finishing = self.discarding = False
        self.environment.reset(self.seed)
        self.last_info = self.environment.environment._info() if self.spatial else {}

    def toggle(self):
        if self.saving:
            return
        self.active = True if self.finishing else not self.active

    def action(self, mouse=None, direction=(0, 0), yaw=0.):
        if self.spatial:
            return self.controller.act(mouse=mouse, direction=direction, yaw=yaw)
        return self.controller.act(mouse=mouse, direction=direction)

    def step(self, action):
        if not self.active or self.saving:
            return None
        transition = self.recorder.step(action)
        self.last_info = transition.info
        if transition.done or transition.terminated or transition.truncated:
            if self.discarding:
                self.reset()
                return None
            return self._save("environment")
        if self.spatial and transition.info.get("pending_finish"):
            self.finishing = True
        if self.recorder.frames >= self.max_steps and not self.finishing:
            return self.save("max_steps")
        return None

    def discard(self):
        """Land a started 3-D attempt before discarding it; reuse its seed."""
        if self.saving:
            return
        if self.spatial and self.recorder.frames:
            self.environment.request_finish("discard")
            self.discarding = self.finishing = self.active = True
        else:
            self.reset()

    def save(self, reason="manual_save"):
        """Finish a 3-D episode physically before committing its data."""
        if self.saving or not self.recorder.frames:
            return None
        if self.spatial:
            self.environment.request_finish(reason)
            self.finishing = self.active = True
            return None
        return self._save(reason)

    def _save(self, reason):
        context = {"episode_index": self.saved, "seed": self.seed}
        if self.spatial:
            context.update(finish_reason=self.last_info.get("finish_reason"),
                           xy_error_m=self.last_info.get("xy_error"),
                           yaw_error_rad=self.last_info.get("yaw_error"),
                           final_height_m=self.last_info.get("height"),
                           failure_reason=self.last_info.get("failure_reason"))
        if self._save_executor is not None:
            self.active = False
            self._save_context = context
            self._save_started = perf_counter()
            self._save_future = self._save_executor.submit(self._write_episode, self.recorder, reason)
            return None
        return self._complete_save(self._write_episode(self.recorder, reason), context)

    def _write_episode(self, recorder, reason):
        started = perf_counter()
        result = recorder.save(self.writer, reason)
        # Disk-buffered recorders already checkpoint before unmapping RGB.
        if getattr(self.writer, "root", None) is None:
            self.writer.checkpoint()
        result["save_seconds"] = perf_counter() - started
        return result

    def poll_save(self, *, wait=False):
        """Publish a completed save and reset physics only on the caller thread."""
        future = self._save_future
        if future is None or (not wait and not future.done()):
            return None
        try:
            result = future.result()
        except BaseException:
            if future.done():
                self._save_future = self._save_context = None
                self.active = self.finishing = False
            raise
        context = self._save_context
        self._save_future = self._save_context = None
        return self._complete_save(result, context)

    def _complete_save(self, result, context):
        self.saved += 1
        self.last_result = {**context, **result}
        self.seed += 1
        self.reset()
        print(json.dumps(self.last_result), flush=True)
        return self.last_result

    def finish_and_save(self, reason="user_exit"):
        if self.saving:
            return self.poll_save(wait=True)
        result = self.save(reason)
        if self.saving:
            return self.poll_save(wait=True)
        if self.spatial and self.finishing:
            for _ in range(60 * self.environment.fps):
                result = self.step(np.zeros(self.environment.action_dim, dtype=np.float32))
                if self.saving:
                    return self.poll_save(wait=True)
                if not self.finishing:
                    return result
            raise RuntimeError("Physical landing did not finish within 60 seconds of simulation.")
        return result

    def close(self):
        try:
            if self.saving:
                self.poll_save(wait=True)
        finally:
            if self._save_executor is not None:
                self._save_executor.shutdown(wait=True)
                self._save_executor = None
            if hasattr(self, "recorder"):
                self.recorder.close()


def lift_readiness(session):
    """Separate the automatic opening lift from permission to use XY keys."""
    phase = session.last_info.get("phase", "lifting")
    if session.saving:
        return "本条已结束；正在统计、编码和保存，XY / yaw 操作已锁定"
    if phase == "control":
        return ("抬升已完成；可以开始 XY / yaw 操作" if session.active else
                "抬升已完成；当前暂停，按空格开始 XY / yaw 操作")
    if phase == "lifting":
        return ("抬升未完成；请等待，暂不能操作 XY / yaw" if session.active else
                "抬升未完成；按空格开始抬升，暂不能操作 XY / yaw")
    if phase in ("lowering", "settling", "complete"):
        return "抬升已完成；正在落放或落稳，XY / yaw 操作已锁定"
    return "本条操作失败；XY / yaw 操作已锁定"


def run_window(session, episode_limit=0):
    import pygame

    pygame.init()
    session.enable_background_saving()
    spatial = session.spatial
    screen = pygame.display.set_mode((1024, 720) if spatial else (512, 660))
    pygame.display.set_caption(f"{session.environment.name} | Human demonstrations | LeRobot")
    font = pygame.font.Font(None, 23)
    # Avoid Windows font-registry enumeration: some installed font entries
    # contain numeric values that older Pygame versions cannot parse.
    chinese_font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/msyh.ttc"
    readiness_font = pygame.font.Font(str(chinese_font), 20) if spatial and chinese_font.is_file() else font
    clock = pygame.time.Clock()
    period_ms = 1000 / session.environment.fps
    next_tick = pygame.time.get_ticks() + period_ms
    next_preview = 0
    exiting = False
    cached_views = None
    try:
        while True:
            session.poll_save()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    session.save("user_exit")
                    exiting = True
                elif event.type == pygame.KEYDOWN and not exiting:
                    if event.key == pygame.K_SPACE:
                        session.toggle()
                        next_tick = pygame.time.get_ticks() + period_ms
                    elif event.key == pygame.K_RETURN:
                        session.save()
                    elif event.key == pygame.K_r and not session.finishing and not session.saving:
                        session.discard()
            if exiting and not session.finishing and not session.saving:
                break
            if episode_limit and session.saved >= episode_limit:
                break
            now = pygame.time.get_ticks()
            if session.active and now >= next_tick:
                automatic = spatial and session.environment.environment.phase != "control"
                if not pygame.key.get_focused() and not automatic and not session.finishing:
                    session.active = False
                else:
                    keys = pygame.key.get_pressed()
                    mouse = None
                    yaw = float(int(keys[pygame.K_e]) - int(keys[pygame.K_q])) if spatial else 0.
                    if session.environment.name in ("pushbox2d", "pushbox3d"):
                        direction = (int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT]),
                                     (int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN])) if spatial else
                                     (int(keys[pygame.K_DOWN]) - int(keys[pygame.K_UP])))
                    else:
                        direction = (int(keys[pygame.K_d]) - int(keys[pygame.K_a]),
                                     int(keys[pygame.K_s]) - int(keys[pygame.K_w]))
                        x, y = pygame.mouse.get_pos()
                        if pygame.mouse.get_pressed()[0] and 0 <= x < 512 and 0 <= y < 512:
                            mouse = (x, y)
                    session.step(session.action(mouse=mouse, direction=direction, yaw=yaw))
                next_tick += period_ms
                if next_tick < now - period_ms:
                    next_tick = now + period_ms
            if now < next_preview:
                clock.tick(max(60, session.environment.fps * 2))
                continue
            next_preview = now + 33
            if not session.saving or cached_views is None:
                if spatial:
                    cached_views = session.environment.render_preview()
                else:
                    native = session.environment.environment
                    size = native.render_size
                    try:
                        native.render_size = 512
                        cached_views = {"Overview": session.environment.render()}
                    finally:
                        native.render_size = size
            screen.fill((28, 30, 36))
            for index, (label, pixels) in enumerate(cached_views.items()):
                # Both 3-D previews are rendered directly at width 512. Keep
                # camera2's aspect ratio; never stretch a 96-pixel image.
                surface = pygame.surfarray.make_surface(pixels.transpose(1, 0, 2))
                y = 24 + (512 - pixels.shape[0]) // 2
                screen.blit(surface, (index * 512, y))
                screen.blit(font.render(label, True, (240, 240, 240)), (index * 512 + 8, 3))
            status = (f"SAVING ({session.saving_elapsed:.1f} s)" if session.saving else
                      "RECORDING" if session.active else "PAUSED - Space to start/resume")
            controls = ("Arrows: XY | Q: CCW (-) | E: CW (+) | Z automatic: lift/land 20 mm" if spatial else
                        "Grip center | Arrow keys: XY" if session.environment.name == "pushbox2d" else
                        "Agent | hold left mouse or WASD")
            lines = [f"{status} | saved {session.saved} | frames {session.recorder.frames}", controls,
                     "Space: start/pause | Enter: land + save" if spatial else
                     "Space: start/pause | Enter: save | R: discard/reset",
                     "Esc: land + save + quit | R: discard/reset | aligned: auto-land + save" if spatial else
                     "Esc: save partial + quit | success: auto-save + pause"]
            if spatial:
                info = session.last_info
                lines.append(f"Phase: {info.get('phase', 'lifting')} | XY error: {info.get('xy_error', 0) * 1000:.2f} mm"
                             f" | yaw: {np.degrees(info.get('yaw', 0)):+.2f} deg"
                             f" | Lift: {info.get('lifted_height', 0) * 1000:.2f}/20 mm | {info.get('failure_reason') or ''}")
                lines.append(lift_readiness(session))
            if session.last_result:
                lines.append(f"Last episode: {'SUCCESS' if session.last_result['success'] else 'INCOMPLETE/FAILED'}"
                             f" | {session.last_result['frames']} frames")
            for index, line in enumerate(lines):
                line_font = readiness_font if spatial and index == 5 else font
                screen.blit(line_font.render(line, True, (240, 240, 240)), (8, 542 + index * 23))
            pygame.display.flip()
            clock.tick(max(60, session.environment.fps * 2))
    finally:
        pygame.quit()
