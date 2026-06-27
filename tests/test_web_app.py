"""Smoke tests for the Dash web app (no server started)."""

import dash

from lorenz_attractor.web.app import create_app


def test_create_app_returns_dash_instance():
    app = create_app()
    assert isinstance(app, dash.Dash)


def test_app_has_a_layout():
    app = create_app()
    assert app.layout is not None
