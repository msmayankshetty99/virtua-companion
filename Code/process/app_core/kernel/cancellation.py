"""One vocabulary for work that stopped on purpose, whichever provider or package stopped it."""


class Deferred(Exception):
    """Stopped on purpose (a Stop, a live turn, shutdown), not failed: drop or retry it, never report it as an error."""


class TurnCancelled(Deferred):
    """Expected user interruption, not a provider failure."""


class BackgroundPreempted(Deferred, RuntimeError):
    """Background inference (initiative, reflection) gave way to a live turn, a cancel or shutdown; retry it later."""
