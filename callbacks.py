# mixup/callbacks.py


class EarlyStopping:
    """
    Arrête l'entraînement si la métrique principale
    ne s'améliore pas après `patience` validations.
    """

    def __init__(self, patience: int, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.num_bad_epochs = 0

    def step(self, current: float) -> bool:
        """
        Appeler après chaque évaluation.
        Si return True → on doit early-stop.
        """
        if self.best is None:
            self.best = current
            return False

        improved = (self.mode == "max" and current > self.best) or (
            self.mode == "min" and current < self.best
        )
        if improved:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        return self.num_bad_epochs >= self.patience
