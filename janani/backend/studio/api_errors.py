import logging

from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_handler

log = logging.getLogger("studio.api")


class ApiError(Exception):
    def __init__(self, message, status=400, code="bad_request", extra=None):
        super().__init__(message)
        self.message, self.status, self.code, self.extra = message, status, code, extra or {}


def exception_handler(exc, context):
    if isinstance(exc, ApiError):
        return Response({"error": {"code": exc.code, "message": exc.message, **exc.extra}}, status=exc.status)
    response = drf_handler(exc, context)
    if response is not None:
        detail = response.data.get("detail") if isinstance(response.data, dict) else None
        response.data = {"error": {"code": "invalid", "message": str(detail or "Invalid request."),
                                   "fields": response.data if not detail else None}}
        return response
    log.exception("Unhandled API error")
    return Response({"error": {"code": "server_error", "message": f"{type(exc).__name__}: {exc}"}}, status=500)


from rest_framework.negotiation import DefaultContentNegotiation


class AnyAcceptNegotiation(DefaultContentNegotiation):
    """Always answer with the first renderer. The SSE endpoints stream raw text and browsers
    send `Accept: text/event-stream`, which a JSON-only API would otherwise reject with 406."""

    def select_renderer(self, request, renderers, format_suffix=None):
        return renderers[0], renderers[0].media_type
