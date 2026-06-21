"""Neo4j connection wrapper for JARVIS Phase 4.

Provides a thin client around the official ``neo4j`` Python driver with:

    1. Graceful degradation — if Neo4j is unreachable the rest of JARVIS
       still works (graph features are silently disabled).
    2. Standardised read/write transaction helpers.
    3. A single place to manage the driver lifecycle.

The driver is imported lazily so JARVIS boots without the ``neo4j`` package
when graph features are disabled in ``config.yaml``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class Neo4jConnectionError(RuntimeError):
    """Raised when JARVIS cannot reach the local Neo4j instance."""


@dataclass
class Neo4jClient:
    """Manages the Neo4j Bolt driver lifecycle and query execution.

    Attributes:
        uri:       Bolt URI, e.g. ``bolt://localhost:7687``.
        username:  Neo4j username.
        password:  Neo4j password.
        database:  Target database name (default ``neo4j``).
    """

    uri: str
    username: str
    password: str
    database: str = "neo4j"

    # Private — set after connect()
    _driver: Any = field(default=None, init=False, repr=False)

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def connect(self) -> None:
        """Open a driver connection to Neo4j.

        Raises ``Neo4jConnectionError`` if the ``neo4j`` package is missing
        or the server is unreachable.
        """
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise Neo4jConnectionError(
                "Graph features require the 'neo4j' Python package. "
                "Install it with: pip install neo4j"
            ) from exc

        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
            )
            # Verify the connection is live.
            self._driver.verify_connectivity()
            logger.info("Neo4j connected: %s", self.uri)
        except Exception as exc:
            self._driver = None
            raise Neo4jConnectionError(
                f"Cannot reach Neo4j at {self.uri}. "
                "Start Neo4j with: docker compose up neo4j -d"
            ) from exc

    def close(self) -> None:
        """Close the driver connection gracefully."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed.")

    def is_connected(self) -> bool:
        """Return True if the driver is open and responsive."""
        if self._driver is None:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    # -----------------------------------------------------------------------
    # Query execution
    # -----------------------------------------------------------------------

    def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a write transaction and return all result records as dicts.

        Raises ``Neo4jConnectionError`` if the driver is not connected.
        """
        self._ensure_connected()
        with self._driver.session(database=self.database) as session:
            result = session.execute_write(
                lambda tx: list(tx.run(query, parameters or {}).data())
            )
        return result

    def execute_read(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a read transaction and return all result records as dicts."""
        self._ensure_connected()
        with self._driver.session(database=self.database) as session:
            result = session.execute_read(
                lambda tx: list(tx.run(query, parameters or {}).data())
            )
        return result

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        if self._driver is None:
            raise Neo4jConnectionError(
                "Neo4j client is not connected. Call connect() first."
            )
