import numpy as np

class GaussianStatStopper:
    def __init__(self, patience=500, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.history = []

    def update(self, gaussians):
        pos = gaussians["positions"]
        sca = gaussians["scales"]
        opa = gaussians["opacities"]

        # 전역 통계량 (mean, std)
        stats = np.concatenate([
            pos.mean(0), pos.std(0),
            sca.mean(0), sca.std(0),
            [opa.mean(), opa.std()]
        ])
        self.history.append(stats)

        if len(self.history) < self.patience:
            return False
        
        recent = np.stack(self.history[-self.patience:])
        diff = np.mean(np.abs(recent[1:] - recent[:-1]))
        if diff < self.min_delta:
            print(f"[AdaptiveStop] Global Gaussian stats stabilized (Δ={diff:.6e}) → stopping.")
            return True
        return False