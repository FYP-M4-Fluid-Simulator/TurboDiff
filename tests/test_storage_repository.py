from turbodiff.db.storage import (
    InMemoryStorageRepository,
    OptimizedAirfoilPayload,
    SessionCreatePayload,
    SimulationMetricsUpdate,
)


def test_create_session_with_airfoil_sets_null_metrics():
    repo = InMemoryStorageRepository()
    result = repo.create_session_with_airfoil(
        SessionCreatePayload(
            session_id="session-1",
            user_id="user-1",
            session_type="simulate",
            parameters={"request": {"fidelity": "medium"}},
            cst_weights_upper=[0.1, 0.2],
            cst_weights_lower=[-0.1, -0.2],
            chord_length=0.25,
            angle_of_attack=5.0,
        )
    )

    assert result.session_id == "session-1"
    airfoil = repo.get_latest_airfoil("session-1", is_optimized=False)
    assert airfoil is not None
    assert airfoil.is_optimized is False
    assert airfoil.cl is None
    assert airfoil.cd is None


def test_update_simulation_metrics_updates_existing_airfoil():
    repo = InMemoryStorageRepository()
    repo.create_session_with_airfoil(
        SessionCreatePayload(
            session_id="session-2",
            user_id="user-2",
            session_type="simulate",
            parameters={},
            cst_weights_upper=[0.1, 0.2],
            cst_weights_lower=[-0.1, -0.2],
            chord_length=0.3,
            angle_of_attack=None,
        )
    )

    updated = repo.update_simulation_metrics(
        SimulationMetricsUpdate(
            session_id="session-2",
            user_id="user-2",
            cl=0.9,
            cd=0.1,
            lift=2.1,
            drag=0.3,
            angle_of_attack=4.0,
        )
    )

    assert updated.cl == 0.9
    assert updated.cd == 0.1
    assert updated.lift == 2.1
    assert updated.drag == 0.3
    assert updated.is_optimized is False


def test_save_optimized_airfoil_creates_new_record():
    repo = InMemoryStorageRepository()
    repo.create_session_with_airfoil(
        SessionCreatePayload(
            session_id="session-3",
            user_id="user-3",
            session_type="optimize",
            parameters={},
            cst_weights_upper=[0.1, 0.2],
            cst_weights_lower=[-0.1, -0.2],
            chord_length=0.25,
            angle_of_attack=None,
        )
    )

    optimized = repo.save_optimized_airfoil(
        OptimizedAirfoilPayload(
            session_id="session-3",
            user_id="user-3",
            cst_weights_upper=[0.15, 0.25],
            cst_weights_lower=[-0.12, -0.22],
            chord_length=0.25,
            angle_of_attack=3.0,
            cl=1.2,
            cd=0.08,
            lift=2.5,
            drag=0.2,
        )
    )

    assert optimized.is_optimized is True
    assert optimized.cl == 1.2
    assert optimized.cd == 0.08


def test_list_cst_for_user_returns_unique_items():
    repo = InMemoryStorageRepository()
    repo.create_session_with_airfoil(
        SessionCreatePayload(
            session_id="session-4",
            user_id="user-4",
            session_type="optimize",
            parameters={},
            cst_weights_upper=[0.1, 0.2],
            cst_weights_lower=[-0.1, -0.2],
            chord_length=0.25,
            angle_of_attack=None,
        )
    )
    repo.save_optimized_airfoil(
        OptimizedAirfoilPayload(
            session_id="session-4",
            user_id="user-4",
            cst_weights_upper=[0.15, 0.25],
            cst_weights_lower=[-0.12, -0.22],
            chord_length=0.25,
            angle_of_attack=2.0,
            cl=1.1,
            cd=0.09,
            lift=2.4,
            drag=0.22,
        )
    )

    items = repo.list_cst_for_user("user-4")
    assert len(items) == 2
    weights = {(tuple(item.weights_upper), tuple(item.weights_lower)) for item in items}
    assert ((0.1, 0.2), (-0.1, -0.2)) in weights
    assert ((0.15, 0.25), (-0.12, -0.22)) in weights
