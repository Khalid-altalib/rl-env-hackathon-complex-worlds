from catanatron import Player, RandomPlayer


class OpenRewardPlayer(Player):
    """Catanatron Player whose action is supplied externally by a tool call.

    The CatanEnv sets `pending_action` before invoking `game.play_tick()`.
    `decide()` returns it (and clears it) so the engine can advance one ply.
    `last_playable` is captured so the env's tools can show the agent its options.
    """

    def __init__(self, color):
        super().__init__(color)
        self.pending_action = None
        self.last_playable: list = []

    def decide(self, game, playable_actions):
        self.last_playable = list(playable_actions)
        if self.pending_action is None:
            raise RuntimeError(
                "OpenRewardPlayer.decide called with no pending_action set. "
                "The env should set pending_action before calling play_tick()."
            )
        action = self.pending_action
        self.pending_action = None
        return action

    def reset_state(self):
        self.pending_action = None
        self.last_playable = []

    def __reduce__(self):
        # Serialize as a plain RandomPlayer so consumers without our package
        # (e.g. the catanatron web server in Docker) can unpickle the game.
        return (RandomPlayer, (self.color,))
