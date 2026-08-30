from __future__ import annotations


PRODUCTION_FRONTEND_ORIGIN = "https://ai-data-analyst-wine-mu.vercel.app"


def _health_preflight(client, origin: str):
    return client.options(
        "/api/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )


def test_production_frontend_origin_can_call_health(client):
    response = _health_preflight(client, PRODUCTION_FRONTEND_ORIGIN)

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == PRODUCTION_FRONTEND_ORIGIN
    assert response.headers["access-control-allow-credentials"] == "true"


def test_production_frontend_origin_can_make_actual_browser_upload_request(client):
    response = client.post(
        "/api/datasets/upload",
        headers={"Origin": PRODUCTION_FRONTEND_ORIGIN},
        files={"file": ("sales.csv", b"revenue\n10\n20\n", "text/csv")},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == PRODUCTION_FRONTEND_ORIGIN
    assert response.headers["access-control-allow-credentials"] == "true"


def test_untrusted_origin_is_not_allowed_to_call_health(client):
    response = _health_preflight(client, "https://untrusted.example")

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
