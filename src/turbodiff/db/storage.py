from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg.types.json import Json


@dataclass(frozen=True)
class SessionCreatePayload:
    session_id: str
    user_id: str
    session_type: str
    parameters: Dict[str, Any]
    cst_weights_upper: List[float]
    cst_weights_lower: List[float]
    chord_length: float
    angle_of_attack: Optional[float]


@dataclass(frozen=True)
class SimulationMetricsUpdate:
    session_id: str
    user_id: str
    cl: Optional[float]
    cd: Optional[float]
    lift: Optional[float]
    drag: Optional[float]
    angle_of_attack: Optional[float]


@dataclass(frozen=True)
class OptimizedAirfoilPayload:
    session_id: str
    user_id: str
    cst_weights_upper: List[float]
    cst_weights_lower: List[float]
    chord_length: float
    angle_of_attack: Optional[float]
    cl: Optional[float]
    cd: Optional[float]
    lift: Optional[float]
    drag: Optional[float]


@dataclass(frozen=True)
class SessionRecord:
    id: str
    user_id: str
    session_type: str
    parameters: Dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class CstRecord:
    id: str
    weights_upper: List[float]
    weights_lower: List[float]
    chord_length: float
    created_at: datetime


@dataclass(frozen=True)
class AirfoilRecord:
    id: str
    cst_id: str
    session_id: Optional[str]
    is_optimized: bool
    cl: Optional[float]
    cd: Optional[float]
    lift: Optional[float]
    drag: Optional[float]
    angle_of_attack: Optional[float]
    created_by_user_id: str
    created_at: datetime


@dataclass(frozen=True)
class SessionCreateResult:
    session_id: str
    cst_id: str
    airfoil_id: str


class StorageRepository:
    def create_session_with_airfoil(
        self, payload: SessionCreatePayload
    ) -> SessionCreateResult:
        raise NotImplementedError

    def update_simulation_metrics(
        self, payload: SimulationMetricsUpdate
    ) -> AirfoilRecord:
        raise NotImplementedError

    def save_optimized_airfoil(self, payload: OptimizedAirfoilPayload) -> AirfoilRecord:
        raise NotImplementedError

    def get_latest_airfoil(
        self, session_id: str, is_optimized: bool
    ) -> Optional[AirfoilRecord]:
        raise NotImplementedError


class InMemoryStorageRepository(StorageRepository):
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionRecord] = {}
        self._cst: Dict[str, CstRecord] = {}
        self._airfoils: Dict[str, AirfoilRecord] = {}
        self._airfoils_by_session: Dict[str, List[str]] = {}

    def create_session_with_airfoil(
        self, payload: SessionCreatePayload
    ) -> SessionCreateResult:
        if payload.session_id in self._sessions:
            raise ValueError("session_id already exists")
        if payload.session_type not in {"optimize", "simulate"}:
            raise ValueError("invalid session_type")

        now = datetime.now(tz=timezone.utc)
        session = SessionRecord(
            id=payload.session_id,
            user_id=payload.user_id,
            session_type=payload.session_type,
            parameters=payload.parameters,
            created_at=now,
        )
        self._sessions[payload.session_id] = session

        cst_id = str(uuid4())
        cst = CstRecord(
            id=cst_id,
            weights_upper=list(payload.cst_weights_upper),
            weights_lower=list(payload.cst_weights_lower),
            chord_length=payload.chord_length,
            created_at=now,
        )
        self._cst[cst_id] = cst

        airfoil_id = str(uuid4())
        airfoil = AirfoilRecord(
            id=airfoil_id,
            cst_id=cst_id,
            session_id=payload.session_id,
            is_optimized=False,
            cl=None,
            cd=None,
            lift=None,
            drag=None,
            angle_of_attack=payload.angle_of_attack,
            created_by_user_id=payload.user_id,
            created_at=now,
        )
        self._airfoils[airfoil_id] = airfoil
        self._airfoils_by_session.setdefault(payload.session_id, []).append(airfoil_id)

        return SessionCreateResult(
            session_id=payload.session_id,
            cst_id=cst_id,
            airfoil_id=airfoil_id,
        )

    def update_simulation_metrics(
        self, payload: SimulationMetricsUpdate
    ) -> AirfoilRecord:
        airfoil_id = self._find_latest_airfoil_id(
            payload.session_id, is_optimized=False
        )
        if airfoil_id is None:
            raise ValueError("no matching airfoil for session")
        existing = self._airfoils[airfoil_id]
        if existing.created_by_user_id != payload.user_id:
            raise ValueError("user_id mismatch for airfoil")

        updated = AirfoilRecord(
            id=existing.id,
            cst_id=existing.cst_id,
            session_id=existing.session_id,
            is_optimized=existing.is_optimized,
            cl=payload.cl,
            cd=payload.cd,
            lift=payload.lift,
            drag=payload.drag,
            angle_of_attack=payload.angle_of_attack,
            created_by_user_id=existing.created_by_user_id,
            created_at=existing.created_at,
        )
        self._airfoils[airfoil_id] = updated
        return updated

    def save_optimized_airfoil(self, payload: OptimizedAirfoilPayload) -> AirfoilRecord:
        if payload.session_id not in self._sessions:
            raise ValueError("unknown session_id")

        now = datetime.now(tz=timezone.utc)
        cst_id = str(uuid4())
        cst = CstRecord(
            id=cst_id,
            weights_upper=list(payload.cst_weights_upper),
            weights_lower=list(payload.cst_weights_lower),
            chord_length=payload.chord_length,
            created_at=now,
        )
        self._cst[cst_id] = cst

        airfoil_id = str(uuid4())
        airfoil = AirfoilRecord(
            id=airfoil_id,
            cst_id=cst_id,
            session_id=payload.session_id,
            is_optimized=True,
            cl=payload.cl,
            cd=payload.cd,
            lift=payload.lift,
            drag=payload.drag,
            angle_of_attack=payload.angle_of_attack,
            created_by_user_id=payload.user_id,
            created_at=now,
        )
        self._airfoils[airfoil_id] = airfoil
        self._airfoils_by_session.setdefault(payload.session_id, []).append(airfoil_id)
        return airfoil

    def get_latest_airfoil(
        self, session_id: str, is_optimized: bool
    ) -> Optional[AirfoilRecord]:
        airfoil_id = self._find_latest_airfoil_id(session_id, is_optimized=is_optimized)
        if airfoil_id is None:
            return None
        return self._airfoils[airfoil_id]

    def _find_latest_airfoil_id(
        self, session_id: str, is_optimized: bool
    ) -> Optional[str]:
        airfoil_ids = self._airfoils_by_session.get(session_id, [])
        for airfoil_id in reversed(airfoil_ids):
            airfoil = self._airfoils[airfoil_id]
            if airfoil.is_optimized == is_optimized:
                return airfoil_id
        return None


class PostgresStorageRepository(StorageRepository):
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def create_session_with_airfoil(
        self, payload: SessionCreatePayload
    ) -> SessionCreateResult:
        if payload.session_type not in {"optimize", "simulate"}:
            raise ValueError("invalid session_type")

        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                self._ensure_user(cur, payload.user_id)

                cur.execute(
                    """
                    INSERT INTO sessions (id, user_id, session_type, parameters)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        payload.session_id,
                        payload.user_id,
                        payload.session_type,
                        Json(payload.parameters),
                    ),
                )

                cur.execute(
                    """
                    INSERT INTO cst (weights_upper, weights_lower, chord_length)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (
                        Json(payload.cst_weights_upper),
                        Json(payload.cst_weights_lower),
                        payload.chord_length,
                    ),
                )
                cst_id = cur.fetchone()["id"]

                cur.execute(
                    """
                    INSERT INTO airfoils (
                        cst_id,
                        session_id,
                        is_optimized,
                        cl,
                        cd,
                        lift,
                        drag,
                        angle_of_attack,
                        created_by_user_id
                    )
                    VALUES (%s, %s, FALSE, NULL, NULL, NULL, NULL, %s, %s)
                    RETURNING id
                    """,
                    (
                        cst_id,
                        payload.session_id,
                        payload.angle_of_attack,
                        payload.user_id,
                    ),
                )
                airfoil_id = cur.fetchone()["id"]

            conn.commit()

        return SessionCreateResult(
            session_id=payload.session_id,
            cst_id=str(cst_id),
            airfoil_id=str(airfoil_id),
        )

    def update_simulation_metrics(
        self, payload: SimulationMetricsUpdate
    ) -> AirfoilRecord:
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM airfoils
                    WHERE session_id = %s AND is_optimized = FALSE
                    ORDER BY created_at DESC
                    LIMIT 1
                    FOR UPDATE
                    """,
                    (payload.session_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError("no matching airfoil for session")
                if str(row["created_by_user_id"]) != payload.user_id:
                    raise ValueError("user_id mismatch for airfoil")

                cur.execute(
                    """
                    UPDATE airfoils
                    SET cl = %s,
                        cd = %s,
                        lift = %s,
                        drag = %s,
                        angle_of_attack = %s
                    WHERE id = %s
                    RETURNING *
                    """,
                    (
                        payload.cl,
                        payload.cd,
                        payload.lift,
                        payload.drag,
                        payload.angle_of_attack,
                        row["id"],
                    ),
                )
                updated = cur.fetchone()
            conn.commit()

        return _airfoil_from_row(updated)

    def save_optimized_airfoil(self, payload: OptimizedAirfoilPayload) -> AirfoilRecord:
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM sessions WHERE id = %s",
                    (payload.session_id,),
                )
                if cur.fetchone() is None:
                    raise ValueError("unknown session_id")

                self._ensure_user(cur, payload.user_id)

                cur.execute(
                    """
                    INSERT INTO cst (weights_upper, weights_lower, chord_length)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (
                        Json(payload.cst_weights_upper),
                        Json(payload.cst_weights_lower),
                        payload.chord_length,
                    ),
                )
                cst_id = cur.fetchone()["id"]

                cur.execute(
                    """
                    INSERT INTO airfoils (
                        cst_id,
                        session_id,
                        is_optimized,
                        cl,
                        cd,
                        lift,
                        drag,
                        angle_of_attack,
                        created_by_user_id
                    )
                    VALUES (%s, %s, TRUE, %s, %s, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (
                        cst_id,
                        payload.session_id,
                        payload.cl,
                        payload.cd,
                        payload.lift,
                        payload.drag,
                        payload.angle_of_attack,
                        payload.user_id,
                    ),
                )
                row = cur.fetchone()
            conn.commit()

        return _airfoil_from_row(row)

    def get_latest_airfoil(
        self, session_id: str, is_optimized: bool
    ) -> Optional[AirfoilRecord]:
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM airfoils
                    WHERE session_id = %s AND is_optimized = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (session_id, is_optimized),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return _airfoil_from_row(row)

    @staticmethod
    def _ensure_user(cursor: psycopg.Cursor, user_id: str) -> None:
        cursor.execute(
            """
            INSERT INTO users (id)
            VALUES (%s)
            ON CONFLICT (id) DO NOTHING
            """,
            (user_id,),
        )


def _airfoil_from_row(row: Dict[str, Any]) -> AirfoilRecord:
    return AirfoilRecord(
        id=str(row["id"]),
        cst_id=str(row["cst_id"]),
        session_id=str(row["session_id"]) if row["session_id"] is not None else None,
        is_optimized=bool(row["is_optimized"]),
        cl=row["cl"],
        cd=row["cd"],
        lift=row["lift"],
        drag=row["drag"],
        angle_of_attack=row["angle_of_attack"],
        created_by_user_id=str(row["created_by_user_id"]),
        created_at=row["created_at"],
    )


_STORAGE_REPO: StorageRepository = InMemoryStorageRepository()


def get_storage_repository() -> StorageRepository:
    return _STORAGE_REPO


def set_storage_repository(repo: StorageRepository) -> None:
    global _STORAGE_REPO
    _STORAGE_REPO = repo


def configure_storage_from_env() -> None:
    load_dotenv()
    database_url = os.environ.get("TURBODIFF_DATABASE_URL")
    if not database_url:
        return
    apply_patches(database_url)
    set_storage_repository(PostgresStorageRepository(database_url))


def apply_patches(database_url: str) -> None:
    patch_dir = Path(__file__).resolve().parent / "patches"
    if not patch_dir.exists():
        return

    patch_files = sorted(patch_dir.glob("*.sql"))
    if not patch_files:
        return

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

            cur.execute("SELECT id FROM schema_migrations")
            applied = {row["id"] for row in cur.fetchall()}

            for patch_path in patch_files:
                patch_id = patch_path.name
                if patch_id in applied:
                    continue
                sql = patch_path.read_text(encoding="utf-8")
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (id) VALUES (%s)",
                    (patch_id,),
                )

        conn.commit()
