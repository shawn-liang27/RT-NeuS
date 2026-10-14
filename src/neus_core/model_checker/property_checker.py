from neus_core.model_checker.stormpy import StormModelChecker


class PropertyChecker:
    """Frame validation, temporal-stage lookup and Storm model checking for one specification.

    ``frame_validator_cls`` picks the frame-level filter: NeuS-QA passes the upstream
    ``neusqa.frame_validator.FrameValidator``, RT-NeuS passes ``rtneus.frame_validator.StageFrameValidator``.
    """

    def __init__(self, proposition, specification, model_type, tl_satisfaction_threshold, detection_threshold,
                 frame_validator_cls):
        self.proposition = proposition
        self.tl_satisfaction_threshold = tl_satisfaction_threshold
        self.specification = self.generate_specification(specification)
        self.model_type = model_type
        self.detection_threshold = detection_threshold

        self.model_checker = StormModelChecker(
            proposition_set=self.proposition,
            ltl_formula=self.specification
        )
        self.frame_validator = frame_validator_cls(
            ltl_formula=self.specification,
            threshold_of_probability=self.detection_threshold
        )

    def generate_specification(self, specification_raw):
        return f"P>={self.tl_satisfaction_threshold:.2f} [ {specification_raw} ]"

    def validate_frame(self, frame_of_interest):
        return self.frame_validator.validate_frame(frame_of_interest)

    def check_automaton(self, automaton):
        return self.model_checker.check_automaton(
            transitions=automaton.transitions,
            states=automaton.states,
            model_type=self.model_type
        )

    def validate_tl_specification(self, specification):
        return self.model_checker.validate_tl_specification(specification)

    @property
    def num_stages(self):
        """Number of temporal stages (sides of top-level Until) detections are split into."""
        if hasattr(self.frame_validator, "stages"):
            return max(1, len(self.frame_validator.stages))
        return 2  # upstream NeuS-QA: always a P set and a Q set

    def check_split(self, prop):
        """Stage index a detected proposition belongs to, or None to ignore it."""
        if hasattr(self.frame_validator, "stage_of"):
            return self.frame_validator.stage_of(prop)
        splits = self.specification.split(" U ")
        return 0 if prop in splits[0] else 1 # accounts for len(splits) == 1
