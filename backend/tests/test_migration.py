from pathlib import Path

from app.database import _alembic_config


def test_alembic_configuration_and_initial_revision_exist():
    config = _alembic_config()
    backend = Path(config.config_file_name).parent
    revision = backend / "migrations" / "versions" / "0001_initial_schema.py"
    assert revision.is_file()
    assert config.get_main_option("script_location") == "migrations"
