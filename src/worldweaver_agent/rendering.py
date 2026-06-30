from __future__ import annotations

from worldweaver_agent.backends import RenderingBackend
from worldweaver_agent.schemas import RenderRequest, RenderResult, WorldState


class RenderingAgent:
    def __init__(self, backend: RenderingBackend):
        self.backend = backend

    def render(self, render_request: RenderRequest, state: WorldState) -> RenderResult:
        return self.backend.render(render_request=render_request, state=state)
