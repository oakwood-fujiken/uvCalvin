"""Simulator-agnostic interactive object logic (Button, Switch, Light, Door).

These classes extract the logical behavior from CALVIN's PyBullet-specific
implementations. They read joint states via the SimHandler interface and
maintain the same toggle/threshold logic.
"""

from __future__ import annotations

import itertools
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from calvin_metasim.env.handler import SimHandler

MAX_FORCE = 4


# ---------------------------------------------------------------------------
# Light
# ---------------------------------------------------------------------------


class LightState(Enum):
    ON = 1
    OFF = 0


class SimLight:
    """Simulator-agnostic light."""

    def __init__(
        self,
        name: str,
        link_name: str,
        color_on: List[float],
        parent_body_id: int,
        handler: SimHandler,
    ):
        self.name = name
        self.handler = handler
        self.parent_body_id = parent_body_id
        self.link_name = link_name
        self.link_id = handler.get_link_index_by_name(parent_body_id, link_name)
        self.color_on = color_on
        self.color_off = [1.0, 1.0, 1.0, 1.0]
        self.state = LightState.OFF

    def reset(self, state: Optional[int] = None):
        if state is None:
            self.turn_off()
        elif state == LightState.ON.value:
            self.turn_on()
        elif state == LightState.OFF.value:
            self.turn_off()
        else:
            raise ValueError("Light state must be 0 or 1.")

    def get_state(self) -> int:
        return self.state.value

    def get_info(self) -> Dict:
        return {"logical_state": self.get_state()}

    def turn_on(self):
        self.state = LightState.ON
        self.handler.change_visual_color(self.parent_body_id, self.link_id, self.color_on)

    def turn_off(self):
        self.state = LightState.OFF
        self.handler.change_visual_color(self.parent_body_id, self.link_id, self.color_off)


# ---------------------------------------------------------------------------
# Button
# ---------------------------------------------------------------------------


class ButtonState(Enum):
    ON = 1
    OFF = 0


class SimButton:
    """Simulator-agnostic button with toggle-on-press logic."""

    def __init__(
        self,
        name: str,
        initial_state: float,
        effect: str,
        parent_body_id: int,
        handler: SimHandler,
    ):
        self.name = name
        self.handler = handler
        self.parent_body_id = parent_body_id
        self.joint_index = handler.get_joint_index_by_name(parent_body_id, name)
        self.initial_state = initial_state
        self.effect = effect

        joint_info = handler.get_joint_info(parent_body_id, self.joint_index)
        self.ll = joint_info["lower_limit"]
        self.ul = joint_info["upper_limit"]
        self.trigger_threshold = (self.ll + self.ul) / 2.0

        # Set position control spring-back
        handler.set_joint_motor_position(
            parent_body_id,
            self.joint_index,
            target_position=initial_state,
            force=MAX_FORCE,
        )

        self.state = ButtonState.OFF
        self.prev_is_pressed = self._is_pressed
        self.light: Optional[SimLight] = None

    def reset(self, state: Optional[float] = None):
        _state = self.initial_state if state is None else state
        self.handler.reset_joint_state(self.parent_body_id, self.joint_index, _state)
        self.state = ButtonState.OFF

    def step(self):
        if self.state == ButtonState.OFF and not self.prev_is_pressed and self._is_pressed:
            self.state = ButtonState.ON
            if self.light is not None:
                self.light.turn_on()
        elif self.state == ButtonState.ON and not self.prev_is_pressed and self._is_pressed:
            self.state = ButtonState.OFF
            if self.light is not None:
                self.light.turn_off()
        self.prev_is_pressed = self._is_pressed

    @property
    def _is_pressed(self) -> bool:
        current = self.get_state()
        if self.initial_state <= self.trigger_threshold:
            return current > self.trigger_threshold
        else:
            return current < self.trigger_threshold

    def get_state(self) -> float:
        return self.handler.get_joint_state(self.parent_body_id, self.joint_index)

    def get_info(self) -> Dict:
        return {"joint_state": self.get_state(), "logical_state": self.state.value}

    def add_effect(self, light: SimLight):
        self.light = light


# ---------------------------------------------------------------------------
# Switch
# ---------------------------------------------------------------------------


class SimSwitch:
    """Simulator-agnostic switch (stays ON while pressed, OFF when released)."""

    def __init__(
        self,
        name: str,
        initial_state: float,
        effect: str,
        parent_body_id: int,
        handler: SimHandler,
    ):
        self.name = name
        self.handler = handler
        self.parent_body_id = parent_body_id
        self.joint_index = handler.get_joint_index_by_name(parent_body_id, name)
        self.initial_state = initial_state
        self.effect = effect

        joint_info = handler.get_joint_info(parent_body_id, self.joint_index)
        self.ll = joint_info["lower_limit"]
        self.ul = joint_info["upper_limit"]
        self.trigger_threshold = (self.ll + self.ul) / 2.0

        # Velocity control for friction
        handler.set_joint_motor_velocity(parent_body_id, self.joint_index, force=MAX_FORCE)

        self.state = ButtonState.OFF
        self.light: Optional[SimLight] = None

    def reset(self, state: Optional[float] = None):
        _state = self.initial_state if state is None else state
        self.handler.reset_joint_state(self.parent_body_id, self.joint_index, _state)
        self.state = ButtonState.OFF

    def step(self):
        if self.is_pressed:
            if self.light is not None and self.state == ButtonState.OFF:
                self.light.turn_on()
            self.state = ButtonState.ON
        else:
            if self.light is not None and self.state == ButtonState.ON:
                self.light.turn_off()
            self.state = ButtonState.OFF

    @property
    def is_pressed(self) -> bool:
        current = self.get_state()
        if self.initial_state <= self.trigger_threshold:
            return current > self.trigger_threshold
        else:
            return current < self.trigger_threshold

    def get_state(self) -> float:
        return self.handler.get_joint_state(self.parent_body_id, self.joint_index)

    def get_info(self) -> Dict:
        return {"joint_state": self.get_state(), "logical_state": self.state.value}

    def add_effect(self, light: SimLight):
        self.light = light


# ---------------------------------------------------------------------------
# Door (drawer / slider)
# ---------------------------------------------------------------------------


class SimDoor:
    """Simulator-agnostic door/drawer/slider joint."""

    def __init__(
        self,
        name: str,
        initial_state: float,
        parent_body_id: int,
        handler: SimHandler,
    ):
        self.name = name
        self.handler = handler
        self.parent_body_id = parent_body_id
        self.joint_index = handler.get_joint_index_by_name(parent_body_id, name)
        self.initial_state = initial_state

        # Velocity control for friction
        handler.set_joint_motor_velocity(parent_body_id, self.joint_index, force=MAX_FORCE)

    def reset(self, state: Optional[float] = None):
        _state = self.initial_state if state is None else state
        self.handler.reset_joint_state(self.parent_body_id, self.joint_index, _state)

    def get_state(self) -> float:
        return self.handler.get_joint_state(self.parent_body_id, self.joint_index)

    def get_info(self) -> Dict:
        return {"current_state": self.get_state()}


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class InteractiveObjectManager:
    """Orchestrates all interactive objects in the scene."""

    def __init__(self):
        self.doors: List[SimDoor] = []
        self.buttons: List[SimButton] = []
        self.switches: List[SimSwitch] = []
        self.lights: List[SimLight] = []

    def link_effects(self):
        """Link buttons/switches to their controlled lights."""
        for light in self.lights:
            for trigger in itertools.chain(self.buttons, self.switches):
                if trigger.effect == light.name:
                    trigger.add_effect(light)

    def step(self):
        """Call step() on all buttons and switches (light toggling)."""
        for trigger in itertools.chain(self.buttons, self.switches):
            trigger.step()

    def reset(self, scene_obs_parts=None):
        """Reset all interactive objects.

        Args:
            scene_obs_parts: If provided, a tuple of (door_info, button_info, switch_info, light_info)
                             arrays from parse_scene_obs.
        """
        if scene_obs_parts is None:
            for obj in itertools.chain(self.doors, self.buttons, self.switches, self.lights):
                obj.reset()
        else:
            door_info, button_info, switch_info, light_info = scene_obs_parts
            for door, state in zip(self.doors, door_info):
                door.reset(float(state))
            for button, state in zip(self.buttons, button_info):
                button.reset(float(state))
            for switch, state in zip(self.switches, switch_info):
                switch.reset(float(state))
            for light, state in zip(self.lights, light_info):
                light.reset(int(state))

    def get_obs(self) -> list:
        """Return state values in CALVIN's scene_obs order: doors, buttons, switches, lights."""
        door_states = [d.get_state() for d in self.doors]
        button_states = [b.get_state() for b in self.buttons]
        switch_states = [s.get_state() for s in self.switches]
        light_states = [l.get_state() for l in self.lights]
        return door_states + button_states + switch_states + light_states

    def get_info(self) -> Dict:
        """Return info dict compatible with CALVIN's PlayTableScene.get_info()."""
        return {
            "doors": {d.name: d.get_info() for d in self.doors},
            "buttons": {b.name: b.get_info() for b in self.buttons},
            "switches": {s.name: s.get_info() for s in self.switches},
            "lights": {l.name: l.get_info() for l in self.lights},
        }
