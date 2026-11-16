import json

from django.http import StreamingHttpResponse


def sse_event(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def sse_response(generator):
    resp = StreamingHttpResponse(generator, content_type="text/event-stream; charset=utf-8")
    resp["Cache-Control"] = "no-cache, no-transform"
    resp["X-Accel-Buffering"] = "no"
    return resp
