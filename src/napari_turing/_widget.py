"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
import time
from .Models._TuringPattern import Boundaries, DiffusionDirection
from .Models._model_list import AvailableModels
from qtpy.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QPushButton
from magicgui import widgets
from napari.qt.threading import thread_worker
from functools import partial

class ModelControler(QWidget):
    def update_layer(self, data):
        self.image_layer.data = data
        self.image_layer.refresh()

    @thread_worker
    def play_click_worker(self):
        while True:
            time.sleep(0.1)
            self.tr.compute_turing(self.increment.value)
            yield self.tr[self.concentration_show.value]

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
        for (val, _, default_val) in self.params.values():
            val.value = default_val
        self.increment.value = self.current_model.increment.value

    @staticmethod
    def reset_value_click(value, slider):
        slider.value = value

    def update_values(self):
        for name, (val, exp, _) in self.params.items():
            self.tr[name] = val.value * exp
        self.tr.boundaries = self.boundaries.value
        self.tr.kernel = self.direction.value

    def change_display_concentration(self):
        if "Concentration" in self.viewer.layers:
            self.viewer.layers.remove("Concentration")
        self.image_layer = self.viewer.add_image(
            self.tr[self.concentration_show.value],
            cache=False,
            name="Concentration",
            colormap=self.current_model.default_color_map,
            interpolation=self.current_model.default_interpolation,
            contrast_limits = self.current_model.default_contrast_limits
        )
        self.image_layer.refresh()

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
        self, name, value, min, max, change_connect=None, is_float=True, dtype=float
    ):
        label = widgets.Label(value=name)
        if dtype is not float or not is_float:
            slider = widgets.Slider(value=value, min=min, max=max)
        else:
            slider = widgets.FloatSlider(value=value, min=min, max=max)
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
        if "Concentration" in self.viewer.layers:
            self.viewer.layers.remove("Concentration")
        concentrations = {c: None for c in self.possible_concentrations}
        if self.randomize:
            if 0 < len(self.viewer.layers):
                if self.viewer.layers.selection.active:
                    l = self.viewer.layers.selection.active
                    concentrations[self.possible_concentrations[0]] = l.data
                else:
                    l = self.viewer.layers[0]
                    concentrations[self.possible_concentrations[0]] = l.data
                self.viewer.layers.remove(l)
            params = {
                name: v[0].value * v[1] for name, v in self.params.items()
            }
            self.tr = self.current_model(
                concentrations=concentrations, **params
            )
        else:
            self.tr.reset()
        self.randomize = True
        self.image_layer = self.viewer.add_image(
            self.tr[self.concentration_show.value],
            cache=False,
            name="Concentration",
            colormap=self.current_model.default_color_map,
            interpolation=self.current_model.default_interpolation,
            contrast_limits = self.current_model.default_contrast_limits
        )
        self.tr.boundaries = self.boundaries.value
        self.tr.kernel = self.direction.value
        for l in self.viewer.layers:
            l.refresh()

    def __init__(self, napari_viewer, current_model):
        super().__init__()
        self.viewer = napari_viewer
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

        self.current_model = current_model

        self.params = {}
        widget_params = []
        for parameter in self.current_model._tunable_parameters:
            p_value, w = self.create_slider(
                parameter.description,
                value=parameter.value,
                min=parameter.min,
                max=parameter.max,
                change_connect=self.update_values,
                dtype=parameter.dtype
            )
            self.params[parameter.name] = (p_value, parameter.exponent, parameter.value)
            widget_params.append(w)

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
            self.current_model.increment.description,
            value=self.current_model.increment.value,
            min=self.current_model.increment.min,
            max=self.current_model.increment.max,
            is_float=False,
        )

        label_display = widgets.Label(value="Concentration to display")
        self.possible_concentrations = self.current_model._concentration_names
        self.concentration_show = widgets.ComboBox(
            value=self.possible_concentrations[0],
            choices=self.possible_concentrations,
        )
        self.concentration_show.changed.connect(
            self.change_display_concentration
        )

        self.randomize = True
        self.create_tr()

        w_label = widgets.Label(value=str(self.tr))

        widget_display = widgets.Container(
            widgets=[label_display, self.concentration_show], labels=False
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
                w_label,
            ]
            + widget_params
            + [
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
        w.native.setContentsMargins(0,0,0,0)
        self.layout().setContentsMargins(0,0,0,0)


class TuringViewer(QWidget):
    def change_model(self):
        if hasattr(self, "controler"):
            self.viewer.window.remove_dock_widget(self.controler)
        self.controler = ModelControler(
            self.viewer, self.model_selection.value.value
        )
        self.viewer.window.add_dock_widget(
            self.controler, name=f"{self.model_selection.value.name} Controler"
        )

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        model_selection_label = widgets.Label(value="Choose the model to run")
        self.model_selection = widgets.ComboBox(
            value=list(AvailableModels)[0], choices=AvailableModels
        )
        self.model_selection.changed.connect(self.change_model)
        self.widget = widgets.Container(
            widgets=[model_selection_label, self.model_selection], labels=False
        )
        layout = QVBoxLayout()
        layout.addStretch(1)
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)
        self.layout().addWidget(self.widget.native)
        
        self.change_model()