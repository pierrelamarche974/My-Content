from __future__ import annotations

import json
import logging

import azure.functions as func

from recommender import ArtifactConfigurationError, get_recommender, parse_clicks


app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


def json_response(payload: dict, status_code: int) -> func.HttpResponse:
    return func.HttpResponse(
        body=json.dumps(payload, ensure_ascii=False),
        status_code=status_code,
        mimetype="application/json",
    )


@app.route(route="recommend", methods=["GET", "POST"])
def recommend(req: func.HttpRequest) -> func.HttpResponse:
    user_id = req.params.get("user_id")
    body: dict = {}
    if req.method == "POST":
        try:
            parsed = req.get_json()
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            body = parsed
        elif user_id is None:
            return json_response(
                {"error": "The request body must be valid JSON."}, 400
            )
    if user_id is None:
        user_id = body.get("user_id")

    try:
        user_id = int(user_id)
        if user_id < 0:
            raise ValueError
    except (TypeError, ValueError):
        return json_response(
            {"error": "user_id must be a non-negative integer."}, 400
        )

    try:
        clicks = parse_clicks(body.get("clicks"))
    except ValueError as error:
        return json_response({"error": str(error)}, 400)

    try:
        service = get_recommender()
        result = service.recommend(user_id, clicks=clicks)
    except ArtifactConfigurationError as error:
        logging.exception("Model artifacts are unavailable")
        return json_response({"error": str(error)}, 503)
    except Exception:
        logging.exception("Recommendation failed")
        return json_response({"error": "Recommendation failed."}, 500)

    return json_response(
        {
            "user_id": result.user_id,
            "recommendations": result.article_ids,
            "scores": result.scores,
            "strategy": result.strategy,
            "model": service.info(),
        },
        200,
    )
