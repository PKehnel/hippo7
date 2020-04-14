from typing import List

import numpy as np
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.properties import NumericProperty
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget

from hippo7_app.hippo7_frontend.graph_manager import InputInterface
from hippo7_app.hippo7_frontend.util import BeatDetector


class GlobalBPMWidget(InputInterface, BoxLayout):
    bpm_text = StringProperty("")

    def __init__(self, observable_node, graph_manager, bpm_manager, **kwargs):
        """
        Schedule the update event 60 times a second. Initialize connection to the bpm_manager.
        They both work on the same node. The BPMWidget is responsible for displaying the bpm and
        offering functionality to manipulate it.

        Args:
            observable_node: Bhe BPM node, storing the current bpm value as float.
            graph_manager: Project graph manager.
            bpm_manager: Checking if the frame is on a beat.
            **kwargs:
        """
        super(GlobalBPMWidget, self).__init__(graph_manager, **kwargs)
        self.bpm_manager = bpm_manager
        self.bpm_observable_node = observable_node
        self.bpm_observable_node.subscribe(self.update_observable)
        self.set_bpm(self.bpm_manager.bpm)
        Clock.schedule_interval(self.update, 1.0 / 60.0)
        self.beat_detector = BeatDetector()
        self.bpm_btn_state = True
        self.update_observable()

        self.time = 0
        self.tap_times = []
        self.tapping_done = 0

        self.window = Window
        self.window.bind(on_keyboard=self._on_keyboard)

    def _on_keyboard(self, instance, key, scancode, codepoint, modifiers):
        """
        Allowing tap detection via space bar.
        Not a long term solution, since keyboard inputs should be handled in a own class.
        """
        if key == 32:
            self.ids["tap_detect"].trigger_action()

    def update(self, dt):
        """
        Update is triggered by a scheduled, 60 times a second (see init).
        It checks if there are enough samples to allow for beat detection and
        checks if the tapping stopped.
        Args:
            dt: Delta time.
        """
        self.time += dt
        if self.bpm_btn_state:
            self.ids.bpm_detect.set_disabled(
                not self.beat_detector.has_enough_samples()
            )
        self.bpm_manager(dt)
        self.ids.bpm_animation.beat_num = self.bpm_manager.counter % 4
        if self.tapping_done < self.time:
            if len(self.tap_times) >= 4:
                bpm = self.get_bpm_from_tap(self.tap_times)
                self.set_bpm(bpm)
            self.tap_times = []

    def detect_bpm(self, *args):
        """
        Call the beat detection from the Beat Manager.
        Args:
            *args: Kivy internal, (contains the caller (Button) of the function)
        """
        bpm = self.beat_detector.detect_beat()
        if bpm > 0:
            self.set_bpm(bpm)
            print("BPM detected: ", bpm)

    def auto_detect(self):
        """
        Toggles the "detect_bpm" button and call the detect_bpm function every 8 seconds.
        The interval needs to be at least a long as the Beat Detections needs for enough samples.
        """
        self.ids.bpm_detect.set_disabled(self.bpm_btn_state)
        if self.bpm_btn_state:
            self.detect_bpm()
            self.auto_update = Clock.schedule_interval(self.detect_bpm, 8.0)
        else:
            self.auto_update.cancel()
        self.bpm_btn_state = not self.bpm_btn_state

    def tap_detect(self):
        """
        Appends the current time to a list o timestamps.
        Called in hippo7.kv
        """
        self.tap_times.append(self.time)
        self.tapping_done = self.time + 2.0

    @staticmethod
    def get_bpm_from_tap(tap_times: List) -> float:
        """
        Compute the BPM based on a list of timestamps.
        Args:
            tap_times: List of time values.

        Returns: The computed bpm value.

        """
        start = tap_times[0]
        tap_times = np.array(tap_times)
        tap_times -= start
        times_between = tap_times[1:] - tap_times[:-1]
        avg_time = times_between.mean()
        bpm = 60.0 / avg_time
        return bpm

    def set_bpm(self, value):
        self.set(
            param_name=self.bpm_observable_node.params["name"],
            node_name=self.bpm_observable_node.params["node_name"],
            value=float(value),
        )

    def update_observable(self, *args, **kwargs):
        self.bpm_text = format(self.bpm_manager.bpm, ".2f")

    def get_bpm(self):
        """
        Function to use at runtime in kv files.
        Returns: BPM (float)

        """
        return self.bpm_manager.bpm


class BPMAnimation(Widget):
    col = NumericProperty(1)
    beat_num = NumericProperty(0)


class BPMManager(InputInterface, Widget):
    bpm = NumericProperty(120.0)
    time = NumericProperty(0)

    def __init__(self, observable_node, graph_manager, **kwargs):
        """
        The main purpose of the BPMManager is to tell if the current frame lies on a beat.
        Args:
            observable_node: The BPM node, storing the current bpm value as float.
            graph_manager: Project graph manager.
            **kwargs:
        """
        super(BPMManager, self).__init__(graph_manager, **kwargs)
        self.counter = 0
        self.bpm_observable_node = observable_node
        self.bpm_observable_node.subscribe(self.update_observable)
        self.bpm = self.bpm_observable_node.params["value"]

    def __call__(self, dt):
        """
        Computes the values for:
          - isBeat: Tells if the current frame lies on a beat.
          - counter: Beat Counter.
        Args:
            dt: Delta Time.
        """
        divby = 60.0 / self.bpm
        dt_next_beat = dt / divby
        self.time += dt_next_beat
        if self.time > 1:
            self.isBeat = True
            self.time = self.time % 1
            self.counter += 1
        else:
            self.isBeat = False

    def update_observable(self, **kwargs):
        self.bpm = float(kwargs["value"])

    def set_bpm(self, bpm):
        self.bpm = bpm
        self.set(
            self.bpm_observable_node.params["name"],
            self.bpm_observable_node.params["node_name"],
            bpm,
        )
