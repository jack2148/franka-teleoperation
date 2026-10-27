"""Best-effort viewer copy in a separate process; never clocks physics."""
import multiprocessing as mp
import queue
import time

import mujoco
import numpy as np
from geomagic.teleop.sync_capture import get_sync_capture


class ViewerBridge:
    def __init__(self, model, xml_path, *, visible=True):
        self.ctx = mp.get_context('spawn')
        self.length = model.nq + model.nv + model.nu + 1
        self.values = self.ctx.RawArray('d', self.length)
        self.sync_values = self.ctx.RawArray('d', 2) if get_sync_capture() else None
        self.frame_id = 0
        self.lock = self.ctx.Lock()
        self.closed = self.ctx.Event()
        self.ready = self.ctx.Event()
        self.keys = self.ctx.Queue(maxsize=32)
        self.errors = self.ctx.Queue(maxsize=1)
        self.process = self.ctx.Process(target=_viewer_main,
            args=(str(xml_path), self.values, self.lock, self.closed, self.ready, self.keys, self.errors, visible, self.sync_values),
            name='teleop_geo_viewer')
        self.started = False

    def publish(self, data):
        if not self.lock.acquire(False):
            return False  # Drop a display snapshot, never wait for rendering.
        try:
            out = np.frombuffer(self.values)
            cursor = 0
            for source in (data.qpos, data.qvel, data.ctrl):
                out[cursor:cursor + len(source)] = source
                cursor += len(source)
            out[-1] = data.time
            if self.sync_values is not None:
                self.frame_id += 1
                self.sync_values[:] = (self.frame_id, time.monotonic_ns())
        finally:
            self.lock.release()
        return True

    def start(self, data):
        self.publish(data)
        self.process.start()
        self.started = True

    def running(self):
        return not self.closed.is_set() and self.process.is_alive()

    def raise_if_failed(self):
        try:
            message = self.errors.get_nowait()
        except queue.Empty:
            message = None
        if message:
            raise RuntimeError(f'MuJoCo viewer failed: {message}')
        if self.started and self.process.exitcode not in (None, 0):
            raise RuntimeError(f'MuJoCo viewer exited (code {self.process.exitcode}); check the GLX/GLFW error above')

    def drain_keys(self):
        result = []
        for _ in range(32):
            try:
                result.append(self.keys.get_nowait())
            except queue.Empty:
                break
        return result

    def close(self):
        self.closed.set()
        if self.started:
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()  # Only our visualization child.
                self.process.join(timeout=2)
        self.keys.close()
        self.errors.close()


class GlfwRenderer:
    """All window/context/event/render operations stay on this process's main thread.

    No passive-viewer render thread, simulation stepping or native mode-key
    shortcuts. ``visible=False`` permits device-free graphics verification.
    """
    def __init__(self, model, key_callback, *, visible=True):
        import glfw
        self.glfw = glfw
        self.window = self.context = None
        self.model = model
        self.last_cursor = None
        if not glfw.init():
            raise RuntimeError('GLFW initialization failed; check DISPLAY and GLX')
        try:
            glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)
            self.window = glfw.create_window(1024, 768, 'teleop_geo — Geomagic / FR3', None, None)
            if not self.window:
                raise RuntimeError('GLFW window/context creation failed')
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)
            self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self.scene = mujoco.MjvScene(model, maxgeom=10000)
            self.options = mujoco.MjvOption()
            self.camera = mujoco.MjvCamera()
            self._reset_camera()
            def on_key(window, key, scancode, action, mods):
                if action != glfw.PRESS: return
                if key == glfw.KEY_ESCAPE:
                    glfw.set_window_should_close(window, True)
                elif key == glfw.KEY_R:
                    self._reset_camera()
                else:
                    key_callback(key)
            glfw.set_key_callback(self.window, on_key)
            glfw.set_cursor_pos_callback(self.window, self._move_camera)
            glfw.set_mouse_button_callback(self.window, self._mouse_button)
            glfw.set_window_focus_callback(self.window, self._focus)
            glfw.set_scroll_callback(self.window, self._scroll_camera)
        except BaseException:
            self.close()
            raise

    def _reset_camera(self):
        # Model-framed free view once at startup, or on an explicit R press.
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)

    def _mouse_button(self, window, button, action, mods):
        # Anchor each drag locally; mouse presses/releases never request close.
        self.last_cursor = self.glfw.get_cursor_pos(window)

    def _focus(self, window, focused):
        # Re-anchor on the next motion to avoid a jump across focus changes.
        self.last_cursor = None

    def _scroll_camera(self, window, x, y):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                             0, -.05*y, self.camera)

    def _move_camera(self, window, x, y):
        glfw = self.glfw
        previous, self.last_cursor = self.last_cursor, (x, y)
        if previous is None: return
        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        if not (left or right or middle): return
        height = max(1, glfw.get_window_size(window)[1])
        shift = any(glfw.get_key(window, k) == glfw.PRESS for k in (glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT))
        if right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(self.model, action, (x-previous[0])/height,
                             (y-previous[1])/height, self.camera)

    def draw(self, data):
        glfw = self.glfw
        glfw.poll_events()
        if glfw.window_should_close(self.window): return False
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0: return False  # Minimized: no invalid drawable swap.
        mujoco.mjv_updateScene(self.model, data, self.options, None, self.camera,
            mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), self.scene, self.context)
        glfw.swap_buffers(self.window)
        return True

    def running(self):
        return not self.glfw.window_should_close(self.window)

    def close(self):
        # Free GL resources with the context still current; destroy window last.
        try:
            if self.context is not None:
                self.glfw.make_context_current(self.window)
                self.context.free()
                self.context = None
        finally:
            if self.window is not None:
                self.glfw.destroy_window(self.window)
                self.window = None
            self.glfw.terminate()


def _viewer_main(xml_path, values, lock, closed, ready, keys, errors, visible=True, sync_values=None):
    renderer = None
    sync = get_sync_capture()
    if sync:
        sync.role = 'viewer'
    frame_id, submitted_ns = 0, None
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        def key_callback(code):
            try:
                keys.put_nowait(code)
            except queue.Full:
                pass
        renderer = GlfwRenderer(model, key_callback, visible=visible)
        while not closed.is_set() and renderer.running():
            started = time.monotonic()
            if lock.acquire(False):
                try:
                    snapshot = np.frombuffer(values).copy()
                    if sync_values is not None:
                        frame_id, submitted_ns = sync_values[:]
                finally:
                    lock.release()
                cursor = 0
                for dest in (data.qpos, data.qvel, data.ctrl):
                    dest[:] = snapshot[cursor:cursor + len(dest)]
                    cursor += len(dest)
                data.time = snapshot[-1]
                mujoco.mj_forward(model, data)  # Visualization only.
            if renderer.draw(data):
                ready.set()  # First successful render + swap.
                if sync:
                    rendered_ns = time.monotonic_ns()
                    sync.emit('preview', float(data.time), timestamp_ns=rendered_ns,
                              T6_ns=rendered_ns, frame=int(frame_id), submitted_ns=submitted_ns,
                              visual_kind='glfw_swap_completed')
            closed.wait(max(0, 1 / 60 - (time.monotonic() - started)))
    except BaseException as error:
        errors.put_nowait(f'{type(error).__name__}: {error}')
        raise
    finally:
        closed.set()  # Closing/error is a force-off request to the owner.
        if renderer is not None: renderer.close()
        if sync:
            sync.write()  # Serialize only after requesting force-off and GL cleanup.
