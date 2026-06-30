from __future__ import annotations

from worldweaver_agent.backends import RetrievalBackend
from worldweaver_agent.schemas import RetrievedFact, WorldState


class RetrievalAgent:
    def __init__(self, backend: RetrievalBackend):
        self.backend = backend

    def search(self, queries: list[str], state: WorldState) -> list[RetrievedFact]:
        if not queries:
            return []
        return self.backend.search(queries=queries, state=state)
