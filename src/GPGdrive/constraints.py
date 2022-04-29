
class StageConstraints(object):
    """
    A class used to represent stage constraints
    """
    def __init__(self, constraints):
        self.stage_constraints = constraints
        self.id_list = ()
        self.length = None

    def __call__(self, *args):
        return self.stage_constraints(*args)

    def __add__(self, constraints):
        return StageConstraints(lambda *args: self(*args)+constraints(*args))

    def __radd__(self, constraints):
        return StageConstraints(lambda *args: constraints(*args)+self(*args))

    def __pos__(self):
        return self

    def __neg__(self):
        return StageConstraints(lambda *args: -self(*args))

    # def __len__(self):  # Todo
    #     return len(self.stage_constraints({0: [0, 0, 0, 0], 1: [0, 0, 0, 0]}))


def stageconstraints(constraints):
    """ Decorator function """
    return StageConstraints(constraints)


def empty_stage_constraints():
    """ Initialize empty constraints """
    @StageConstraints
    def empty(x):
        return []
    empty.length = 0
    return empty

# class CoupledConstraints(object):
#     def __init__(self, players, constraint):
#         self.constraint = constraint
#         self.players = players
#         return

#     def is_player_involved(player_id):
#         return True if player_id in [player.id for player in self.players] else False




