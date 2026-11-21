from django.urls import path

from . import auth, views

urlpatterns = [
    path("health/", views.health),
    path("auth/start/", auth.auth_start),
    path("auth/verify/", auth.auth_verify),
    path("auth/resend/", auth.auth_resend),
    path("auth/me/", auth.auth_me),
    path("auth/logout/", auth.auth_logout),
    path("config/", views.config),
    path("settings/", views.app_settings),
    path("video/inspect/", views.inspect_video),
    path("sources/", views.sources),
    path("sources/index/", views.index_source),
    path("sources/manual/", views.index_manual),
    path("sources/<int:pk>/", views.source_detail),
    path("conversations/", views.conversations),
    path("conversations/<uuid:pk>/", views.conversation_detail),
    path("conversations/<uuid:pk>/clear/", views.conversation_clear),
    path("conversations/<uuid:pk>/ask/", views.ask),
    path("insights/", views.insights),
    path("models/", views.models_info),
    path("models/llm/check/", views.llm_check),
    path("models/embedding/check/", views.embed_check),
    path("cache/clear/", views.cache_clear),
    path("diagnostics/", views.diagnostics),
]
