from TRITON_SWMM_toolkit.experiment import TRITONSWMM_experiment


class TRITONSWMM_sim_post_processing:
    def __init__(self, exp: TRITONSWMM_experiment) -> None:
        self.log = exp.log
