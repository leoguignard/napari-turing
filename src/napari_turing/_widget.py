"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
import time
from ._TuringPattern import TuringPattern
from napari import Viewer
import numpy as np
from qtpy.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QPushButton
from magicgui import widgets
from napari.qt.threading import thread_worker
from functools import partial

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

from enum import Enum


class Boundaries(Enum):
    Closed = "Closed"
    Left_Right_Tube = "LR-Tube"
    Top_Down_Tube = "TD-Tube"
    Inifinite = "Infinite"


class DiffusionDirection(Enum):
    Isotrope = [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]
    Left = [
        [0, 1, 0],
        [2, 0, 0],
        [0, 1, 0],
    ]
    Right = [
        [0, 1, 0],
        [0, 0, 2],
        [0, 1, 0],
    ]
    Top = [
        [0, 2, 0],
        [1, 0, 1],
        [0, 0, 0],
    ]
    Bottom = [
        [0, 0, 0],
        [1, 0, 1],
        [0, 2, 0],
    ]


class TuringViewer(QWidget):
    def update_layer(self, data):
        for l, arr in zip(self.tr_layers, data):
            l.data = arr
            l.refresh()

    @thread_worker
    def play_click_worker(self):
        while True:
            time.sleep(0.1)
            self.tr.compute_turing(self.increment.value)
            to_yield = []
            if self.A_show:
                to_yield.append(self.tr.A)
            if self.I_show:
                to_yield.append(self.tr.I)
            yield to_yield

    def clear_tr(self):
        del self.worker
        self.play.clicked.connect(self.play_click)
        self.create_tr()

    def pause_tr(self):
        self.play.clicked.connect(self.play_click)

    def play_click(self):
        if 1 < len(self.viewer.layers):
            self.create_tr()
        self.play.clicked.disconnect()
        self.worker = self.play_click_worker()
        self.worker.yielded.connect(self.update_layer)
        self.worker.finished.connect(self.clear_tr)
        self.worker.paused.connect(self.pause_tr)
        self.worker.start()

    def pause_click(self):
        if hasattr(self, "worker"):
            self.worker.pause()
        else:
            self.play.clicked.connect(self.play_click)

    def stop_click(self):
        self.randomize = False
        if hasattr(self, "worker"):
            self.worker.quit()
        else:
            self.play.clicked.connect(self.play_click)

    def new_run(self):
        if hasattr(self, "worker"):
            self.worker.quit()
        else:
            self.play.clicked.connect(self.play_click)
            self.create_tr()

    def reset_all_values_click(self):
        self.mu_a.value = 2.8
        self.mu_i.value = 5
        self.tau.value = 0.1
        self.k.value = -5
        self.increment.value = 100

    @staticmethod
    def reset_value_click(value, slider):
        slider.value = value

    def update_values(self):
        self.tr.mu_a = self.mu_a.value * 1e-4
        self.tr.mu_i = self.mu_i.value * 1e-3
        self.tr.tau = self.tau.value
        self.tr.k = self.k.value * 1e-3
        self.tr.boundaries = self.boundaries.value.value
        self.tr.kernel = self.direction.value.value

    def change_I(self):
        if not self.I_show.value:
            self.viewer.grid.enabled = False
            if "Inhibitor" in self.viewer.layers:
                l = self.viewer.layers["Inhibitor"]
                self.tr_layers.remove(l)
                self.viewer.layers.remove("Inhibitor")
            if not self.A_show.value:
                self.A_show.value = True
        else:
            if not "Inhibitor" in self.viewer.layers:
                self.tr_layers.append(
                    self.viewer.add_image(
                        self.tr.I,
                        cache=False,
                        name="Inhibitor",
                        colormap="viridis",
                        interpolation="Spline36",
                    )
                )
                self.viewer.grid.enabled = 1 < len(self.tr_layers)

    def change_A(self):
        if not self.A_show.value:
            self.viewer.grid.enabled = False
            if "Activator" in self.viewer.layers:
                l = self.viewer.layers["Activator"]
                self.tr_layers.remove(l)
                self.viewer.layers.remove("Activator")
            if not self.I_show.value:
                self.I_show.value = True
        else:
            if not "Activator" in self.viewer.layers:
                self.tr_layers.append(
                    self.viewer.add_image(
                        self.tr.I,
                        cache=False,
                        name="Activator",
                        colormap="viridis",
                        interpolation="Spline36",
                    )
                )
                self.viewer.grid.enabled = 1 < len(self.tr_layers)

    @staticmethod
    def create_button(button_name):
        btn = QPushButton(button_name)
        btn.native = btn
        btn.name = button_name
        btn.label = button_name
        btn._explicitly_hidden = False
        btn.tooltip = ""
        btn.label_changed = None
        return btn

    def create_slider(
        self, name, value, min, max, change_connect=None, float=True
    ):
        label = widgets.Label(value=name)
        if float:
            slider = widgets.FloatSlider(value=value, min=min, max=max)
        else:
            slider = widgets.Slider(value=value, min=min, max=max)
        btn = widgets.PushButton(name="Reset")
        btn.changed.connect(
            partial(self.reset_value_click, value=value, slider=slider)
        )
        s_container = widgets.Container(
            widgets=[slider, btn], layout="horizontal", labels=False
        )
        container = widgets.Container(
            widgets=[label, s_container], labels=False
        )
        if change_connect:
            slider.changed.connect(change_connect)
        return slider, container

    def create_tr(self):
        if hasattr(self, "tr_layers") and not self.tr_layers is None:
            for l in self.tr_layers:
                if l in self.viewer.layers:
                    self.viewer.layers.remove(l)
        if self.randomize:
            if 0 < len(self.viewer.layers):
                if self.viewer.layers.selection.active:
                    l = self.viewer.layers.selection.active
                    A = l.data
                else:
                    l = self.viewer.layers[0]
                    A = l.data
                self.viewer.layers.remove(l)
            else:
                A = None
            self.tr = TuringPattern(
                mu_a=self.mu_a.value * 1e-4,
                mu_i=self.mu_i.value * 1e-3,
                tau=self.tau.value,
                dt=0.001,
                k=self.k.value * 1e-3,
                A=A,
            )
        else:
            self.tr.A = self.tr.init_A
            self.tr.I = self.tr.init_I
        self.randomize = True
        self.tr_layers = []
        if self.A_show.value:
            self.tr_layers.append(
                self.viewer.add_image(
                    self.tr.A,
                    cache=False,
                    name="Activator",
                    colormap="viridis",
                    interpolation="Spline36",
                )
            )
        if self.I_show.value:
            self.tr_layers.append(
                self.viewer.add_image(
                    self.tr.I,
                    cache=False,
                    name="Inhibitor",
                    colormap="viridis",
                    interpolation="Spline36",
                )
            )
        self.tr.boundaries = self.boundaries.value.value
        self.tr.kernel = self.direction.value.value
        if 1 < len(self.tr_layers):
            self.viewer.grid.enabled = True
        for l in self.tr_layers:
            l.refresh()

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        np.random.seed(0)
        self.continue_playing = False

        self.play = self.create_button("Play")
        self.play.clicked.connect(self.play_click)
        pause = self.create_button("Pause")
        pause.clicked.connect(self.pause_click)
        stop = self.create_button("Restart")
        stop.clicked.connect(self.stop_click)
        new_run = self.create_button("New Run")
        new_run.clicked.connect(self.new_run)
        reset_values = self.create_button("Reset values")
        reset_values.clicked.connect(self.reset_all_values_click)
        control_w = widgets.Container(
            widgets=[self.play, pause, stop, new_run],
            labels=False,
            layout="horizontal",
        )

        self.mu_a, mu_a_w = self.create_slider(
            "Activator diffusion coefficient (10^-4)",
            value=2.8,
            min=1,
            max=5,
            change_connect=self.update_values,
        )

        self.mu_i, mu_i_w = self.create_slider(
            "Inhibitor diffusion coefficient (10^-3)",
            value=5,
            min=2,
            max=7,
            change_connect=self.update_values,
        )

        self.tau, tau_w = self.create_slider(
            "Reaction time ration between\nActivator and inhibitor",
            value=0.1,
            min=0.01,
            max=2,
            change_connect=self.update_values,
        )

        self.k, k_w = self.create_slider(
            "Is the activator a source (>0), a sink (<0)\nor neutral (0), (10^-3)",
            value=-5,
            min=-10,
            max=10,
            change_connect=self.update_values,
        )

        label_b = widgets.Label(value="Boundary conditions")
        self.boundaries = widgets.RadioButtons(
            value=Boundaries.Closed,
            choices=Boundaries,
        )
        self.boundaries.changed.connect(self.update_values)

        label_d = widgets.Label(value="Diffusion direction")
        self.direction = widgets.ComboBox(
            value=DiffusionDirection.Isotrope,
            choices=DiffusionDirection,
        )
        self.direction.changed.connect(self.update_values)

        self.increment, increment_w = self.create_slider(
            "Number of steps per frame",
            value=100,
            min=10,
            max=1000,
            float=False,
        )

        label_display = widgets.Label(value="Concentration to display")
        self.A_show = widgets.CheckBox(value=True, name="Activator")
        self.A_show.changed.connect(self.change_A)

        self.I_show = widgets.CheckBox(value=False, name="Inhibitor")
        self.I_show.changed.connect(self.change_I)

        self.randomize = True
        self.create_tr()

        widget_AI = widgets.Container(
            widgets=[self.A_show, self.I_show], layout="horizontal"
        )
        widget_display = widgets.Container(
            widgets=[label_display, widget_AI], labels=False
        )
        widget_b = widgets.Container(
            widgets=[label_b, self.boundaries], labels=False
        )
        widget_d = widgets.Container(
            widgets=[label_d, self.direction], labels=False
        )
        geometry_widget = widgets.Container(
            widgets=[widget_display, increment_w, widget_b, widget_d],
            layout="vertical",
            labels=False,
        )

        w = widgets.Container(
            widgets=[
                mu_a_w,
                mu_i_w,
                tau_w,
                k_w,
                reset_values,
            ],
            labels=False,
        )

        tab_controls = QTabWidget()
        tab_controls.addTab(w.native, "Parameters")
        tab_controls.addTab(geometry_widget.native, "Output and Geometry")
        tab_controls.native = tab_controls
        tab_controls.adjustSize()

        w.native.layout().addStretch(1)
        layout = QVBoxLayout()
        layout.addStretch(1)
        self.setLayout(layout)
        self.layout().addWidget(control_w.native)
        self.layout().addWidget(tab_controls)
        w.native.adjustSize()
