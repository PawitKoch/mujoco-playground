"""10 - MuJoCo Studio: the new Python viewer, with your own plugins.

MuJoCo 4 ships `mujoco-studio` (installed by the `mujoco` metapackage): a
plugin-based viewer with a native window (Filament) and a WEB viewer that
renders in your browser, so it also works on a headless box over ssh.

This script adds three custom plugins to the stock Studio UI:
  EnergyProbe    (SimPlugin, sim thread)    reads mjData.energy after every step and
                                            sends an EnergySnapshot to the viewer
  EnergyPanel    (ViewerPlugin, viewer thread) Dear ImGui window: time, energy, a gravity
                                            slider that sends a SetGravity event back
  TipVelocityArrow (ViewerPlugin)           3D overlay arrow at the pendulum tip (extra_geoms)

    uv run python examples/10_studio_viewer/run.py            # web viewer, open the printed URL
    uv run python examples/10_studio_viewer/run.py --native   # native Filament window
    uv run python examples/10_studio_viewer/run.py --smoke    # start, run 3 s, check it serves, exit

Remote machine? `ssh -L 8080:localhost:8080 <host>` then open http://localhost:8080/.
"""

import argparse
import dataclasses
import os
import socket
import sys
import time
import urllib.request
from pathlib import Path

import mujoco
import numpy as np
from mujoco.studio import app as studio_app
from mujoco.studio import launch_web, messages, plugin, viewer_protocol
from mujoco.studio.dear_imgui import dear_imgui as imgui

HERE = Path(__file__).parent


# ---------------------------------------------------------------- messages
# Snapshot = latest-wins (fine to drop stale values); Event = reliable, ordered.
@dataclasses.dataclass(frozen=True)
class EnergySnapshot(messages.Snapshot):
    time: float
    potential: float
    kinetic: float


@dataclasses.dataclass(frozen=True)
class SetGravity(messages.Event):
    g: float


# ---------------------------------------------------------------- sim side
class EnergyProbe(plugin.SimPlugin):
    """Runs on the simulation thread: owns the authoritative model/data."""

    def __init__(self) -> None:
        self.pending_gravity: float | None = None

    @messages.handler
    def on_set_gravity(self, event: SetGravity) -> None:
        self.pending_gravity = event.g

    @messages.handler
    def on_pre_step(self, event: messages.PreStepEvent) -> None:
        # Model mutations belong in step handlers so Studio's diff-sync forwards them.
        if self.pending_gravity is not None:
            event.model.opt.gravity[2] = -self.pending_gravity
            self.pending_gravity = None

    @messages.handler
    def on_post_step(self, event: messages.PostStepEvent) -> None:
        d = event.data
        self.handle.send_to_viewer(EnergySnapshot(d.time, float(d.energy[0]), float(d.energy[1])))


# ---------------------------------------------------------------- viewer side
class EnergyPanel(plugin.ViewerPlugin):
    def __init__(self) -> None:
        self.latest: EnergySnapshot | None = None
        self.e0: float | None = None
        self.gravity = 9.81

    @messages.handler
    def on_energy(self, snap: EnergySnapshot) -> None:
        self.latest = snap
        if self.e0 is None:
            self.e0 = snap.potential + snap.kinetic

    @messages.handler
    def on_build_gui(self, _: messages.BuildGuiEvent) -> None:
        if imgui.Begin("Energy (custom plugin)"):
            s = self.latest
            if s is None:
                imgui.TextUnformatted("waiting for the first step... (press Space to run)")
            else:
                total = s.potential + s.kinetic
                imgui.Text(f"t = {s.time:7.3f} s")
                imgui.Text(f"potential = {s.potential:+8.3f} J")
                imgui.Text(f"kinetic   = {s.kinetic:+8.3f} J")
                imgui.Text(f"total     = {total:+8.3f} J  (drift {total - (self.e0 or total):+.2e})")
                imgui.ProgressBar(s.kinetic / max(1e-9, abs(total) + s.kinetic))
            imgui.Separator()
            changed, self.gravity = imgui.SliderFloat("gravity [m/s^2]", self.gravity, 0.0, 30.0)
            if changed:
                self.viewer.send_to_sim(SetGravity(self.gravity))
                self.e0 = None  # energy reference changes with gravity
        imgui.End()


class TipVelocityArrow(plugin.ViewerPlugin):
    """3D overlay: an arrow along the tip velocity, drawn without touching mjModel."""

    def __init__(self) -> None:
        self.prev: tuple[float, np.ndarray] | None = None

    @messages.handler
    def on_update(self, _: messages.UpdateEvent) -> None:
        self.viewer.extra_geoms.clear()
        m, d = self.viewer.model, self.viewer.data
        if d is None or m is None or m.nsite == 0:
            return
        tip = d.site_xpos[0].copy()
        if self.prev is not None and d.time > self.prev[0]:
            vel = (tip - self.prev[1]) / (d.time - self.prev[0])  # finite difference on the mirrored state
            geom = mujoco.MjvGeom()
            mujoco.mjv_initGeom(geom, int(mujoco.mjtGeom.mjGEOM_ARROW), np.zeros(3), np.zeros(3),
                                np.eye(3).flatten(), np.array([1.0, 0.9, 0.2, 0.9], np.float32))
            mujoco.mjv_connector(geom, int(mujoco.mjtGeom.mjGEOM_ARROW), 0.02, tip, tip + 0.1 * vel)
            self.viewer.extra_geoms.append(geom)
        self.prev = (d.time, tip)


def load():
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("chaos").id)
    return model, data


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def smoke() -> None:
    """Non-blocking launch + our own sim loop, then check the HTTP endpoint and exit."""
    model, data = load()
    port = free_port()
    config = viewer_protocol.ViewerConfig(title="smoke", gfx="web", http_port=port)
    probe, panel = EnergyProbe(), EnergyPanel()
    with launch_web.launch(config, plugins=[probe, panel, TipVelocityArrow()]) as handle:
        t_end = time.time() + 3.0
        while handle.is_running() and time.time() < t_end:
            model, data = handle.sync(model, data)  # always rebind: Studio may swap the model
            tick = time.perf_counter()
            mujoco.mj_step(model, data)
            time.sleep(max(0.0, model.opt.timestep - (time.perf_counter() - tick)))
        html = urllib.request.urlopen(f"http://localhost:{port}/", timeout=5).read()
    print(f"web viewer served {len(html)} bytes on port {port}; sim reached t={data.time:.2f} s")
    print(f"energy at the end: potential={data.energy[0]:+.3f} J kinetic={data.energy[1]:+.3f} J")
    print(f"EnergyPanel (viewer thread) last received snapshot: {panel.latest}")
    assert html and data.time > 0.5
    assert panel.latest is not None, "sim -> viewer EnergySnapshot never arrived"
    print("OK: Studio web viewer started, served HTML and stepped the sim")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--native", action="store_true",
                   help="native Filament window (on Linux, opens the web viewer in your browser instead; see README)")
    p.add_argument("--force-native", action="store_true", help="try the native Filament window even on Linux")
    p.add_argument("--smoke", action="store_true", help="start the web viewer for 3 s and exit (for run_all)")
    p.add_argument("--port", type=int, default=0, help="web viewer port (0 = first free port >= 8080)")
    args = p.parse_args()

    if args.smoke:
        smoke()
        return

    model, data = load()
    # Stock Studio UI + stepping/history, plus our three plugins.
    plugins = [studio_app.App(), *studio_app.SIM_PLUGINS, EnergyProbe(), EnergyPanel(), TipVelocityArrow()]
    open_browser = False
    if args.native and not args.force_native and sys.platform.startswith("linux"):
        print("Studio's native window crashes on Linux in mujoco 4.0.1 "
              "(https://github.com/google-deepmind/mujoco/issues/3941); "
              "using the web viewer instead. Pass --force-native to try it anyway.")
        args.native = False
        # Only auto-open a browser on a desktop session: over SSH, webbrowser may start a text browser in the terminal.
        open_browser = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if args.native or args.force_native:
        from mujoco.studio import launch_native

        launch_native.run(viewer_protocol.ViewerConfig(title="mujoco-playground: Studio"),
                          model=model, data=data, plugins=plugins)
    else:
        config = viewer_protocol.ViewerConfig(title="mujoco-playground: Studio", gfx="web",
                                              http_port=args.port, open_browser=open_browser)
        launch_web.run(config, model=model, data=data, plugins=plugins)


if __name__ == "__main__":
    main()
