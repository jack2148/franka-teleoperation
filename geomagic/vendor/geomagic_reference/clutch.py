"""Edge-aware clutch state machine."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ClutchEvent:
    held: bool
    pressed: bool
    released: bool


class Clutch:
    def __init__(self) -> None:
        self._held = False

    @property
    def held(self) -> bool:
        return self._held

    def update(self, pressed: bool) -> ClutchEvent:
        pressed = bool(pressed)
        event = ClutchEvent(
            held=pressed,
            pressed=pressed and not self._held,
            released=self._held and not pressed,
        )
        self._held = pressed
        return event

