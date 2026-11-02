"""Optional EMA and SWA copies of one training trajectory."""

from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn


class WeightAveraging:
    def __init__(self, model, options):
        self.options = options
        methods = options.get("methods", ["ema", "swa"])
        assert methods and len(set(methods)) == len(methods) and set(methods) <= {"ema", "swa"}
        self.starts = {method: options[f"{method}_start_epoch"] for method in methods}
        self.models = {method: AveragedModel(model, avg_fn=get_ema_avg_fn(options["ema_decay"]) if method == "ema" else None)
                       for method in methods}

    def update_step(self, epoch, model):
        if "ema" in self.models and epoch >= self.starts["ema"]:
            self.models["ema"].update_parameters(model)

    def update_epoch(self, epoch, model):
        if "swa" in self.models and epoch >= self.starts["swa"]:
            self.models["swa"].update_parameters(model)

    def active_models(self):
        return {method: averaged.module for method, averaged in self.models.items()
                if averaged.n_averaged.item() > 0}

    def metadata(self, method):
        return {
            "averaging_method": method,
            "averaging_start_epoch": self.starts[method],
            "averaging_decay": self.options["ema_decay"] if method == "ema" else None,
            "averaging_updates": int(self.models[method].n_averaged.item()),
        }
