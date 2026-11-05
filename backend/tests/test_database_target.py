"""The schema-reset fixture must reject the application database before connecting."""
import pytest
from conftest import ensure_test_database


@pytest.mark.parametrize("database", ["studio_test", "rag_matrix_20261105", "TEST_rag", "integration_test"])
def test_dedicated_targets_are_allowed(database):
    ensure_test_database(f"postgresql+psycopg://local:local@localhost/{database}")


@pytest.mark.parametrize("database", ["studio", "postgres", "production", ""])
def test_application_targets_are_rejected(database):
    with pytest.raises(RuntimeError, match="dedicated"):
        ensure_test_database(f"postgresql+psycopg://local:local@localhost/{database}")
