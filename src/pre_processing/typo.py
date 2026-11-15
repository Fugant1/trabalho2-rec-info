class TypoHandler:
    def __init__(self, typo_correction_model):
        self.typo_correction_model = typo_correction_model

    def correct_typos(self, text: str) -> str:
        # Use the typo correction model to correct typos in the input text
        corrected_text = self.typo_correction_model.correct(text)
        return corrected_text