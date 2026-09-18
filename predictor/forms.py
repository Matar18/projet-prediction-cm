from django import forms
from .ml import team_names


class PredictionForm(forms.Form):
    home_team = forms.ChoiceField(label="Équipe à domicile")
    away_team = forms.ChoiceField(label="Équipe adverse")
    neutral = forms.BooleanField(label="Terrain neutre", required=False, initial=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        choices = [(t, t) for t in team_names()]
        self.fields["home_team"].choices = choices
        self.fields["away_team"].choices = choices

    def clean(self):
        cleaned = super().clean()
        if cleaned.get("home_team") and cleaned.get("home_team") == cleaned.get("away_team"):
            raise forms.ValidationError("Choisissez deux équipes différentes.")
        return cleaned
