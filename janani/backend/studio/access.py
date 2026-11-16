import hmac

from django.conf import settings
from django.http import JsonResponse

OPEN_PATHS = ("/api/health/",)


class AccessTokenMiddleware:
    """If APP_ACCESS_TOKEN is set, every /api/ request must carry `X-Access-Token: <token>`.
    Constant-time comparison; /api/health/ stays open for uptime checks."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        token = settings.APP_ACCESS_TOKEN
        if token and request.path.startswith("/api/") and request.path not in OPEN_PATHS:
            sent = request.headers.get("X-Access-Token", "")
            if not hmac.compare_digest(sent.encode(), token.encode()):
                return JsonResponse({"error": {"code": "access_required",
                                               "message": "This workspace is private. Enter the access code."}},
                                    status=401)
        return self.get_response(request)
