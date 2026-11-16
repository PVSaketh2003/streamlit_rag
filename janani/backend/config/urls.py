from django.urls import include, path, re_path

from studio.views import spa

urlpatterns = [
    path("api/", include("studio.urls")),
    # Everything that is not the API (and not a static file WhiteNoise already served) is the React app.
    re_path(r"^(?!api/).*$", spa),
]
