from django.shortcuts import render

from .forms import PredictionForm
from .ml import predict_match


def predict_view(request):
    result = None

    if request.method == "POST":
        form = PredictionForm(request.POST)
        if form.is_valid():
            result = predict_match(
                home=form.cleaned_data["home_team"],
                away=form.cleaned_data["away_team"],
                neutral=form.cleaned_data["neutral"],
            )
    else:
        initial = {"home_team": "Portugal", "away_team": "Spain", "neutral": True}
        form = PredictionForm(initial=initial)
        # Pré-calcule une prédiction par défaut à l'affichage initial
        try:
            result = predict_match("Portugal", "Spain", neutral=True)
        except KeyError:
            result = None

    return render(request, "predictor/predict.html", {"form": form, "result": result})
