import os

import pytest

from turbodiff.db.storage import (
    PostgresStorageRepository,
    SessionCreatePayload,
    SimulationMetricsUpdate,
    OptimizedAirfoilPayload,
    apply_patches,
)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("TURBODIFF_DATABASE_URL"),
    reason="TURBODIFF_DATABASE_URL not set",
)
def test_postgres_storage_round_trip():
    database_url = os.environ["TURBODIFF_DATABASE_URL"]
    apply_patches(database_url)
    repo = PostgresStorageRepository(database_url)

    session_id = "00000000-0000-0000-0000-000000000001"
    user_id = "00000000-0000-0000-0000-000000000002"

    repo.create_session_with_airfoil(
        SessionCreatePayload(
            session_id=session_id,
            user_id=user_id,
            session_type="simulate",
            parameters={"request": {"fidelity": "low"}},
            cst_weights_upper=[0.1, 0.2],
            cst_weights_lower=[-0.1, -0.2],
            chord_length=0.25,
            angle_of_attack=3.0,
        )
    )

    updated = repo.update_simulation_metrics(
        SimulationMetricsUpdate(
            session_id=session_id,
            user_id=user_id,
            cl=1.0,
            cd=0.1,
            lift=2.0,
            drag=0.2,
            angle_of_attack=4.0,
        )
    )
    assert updated.cl == 1.0
    assert updated.cd == 0.1
    assert updated.is_optimized is False

    optimized = repo.save_optimized_airfoil(
        OptimizedAirfoilPayload(
            session_id=session_id,
            user_id=user_id,
            cst_weights_upper=[0.15, 0.25],
            cst_weights_lower=[-0.12, -0.22],
            chord_length=0.25,
            angle_of_attack=2.0,
            cl=1.3,
            cd=0.08,
            lift=2.6,
            drag=0.25,
        )
    )
    assert optimized.is_optimized is True
    assert optimized.cl == 1.3

    csts = repo.list_cst_for_user(user_id)
    assert len(csts) >= 2
