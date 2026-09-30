from unittest.mock import MagicMock

import pytest

import backend.database.schema as schema


def test_database_dependency_rolls_back_and_closes(monkeypatch):
    session = MagicMock()
    monkeypatch.setattr(schema, "SessionLocal", lambda: session)
    dependency = schema.get_db()
    assert next(dependency) is session
    with pytest.raises(RuntimeError):
        dependency.throw(RuntimeError("boom"))
    session.rollback.assert_called_once()
    session.close.assert_called_once()


def test_database_dependency_commits_and_closes(monkeypatch):
    session = MagicMock()
    monkeypatch.setattr(schema, "SessionLocal", lambda: session)
    dependency = schema.get_db()
    assert next(dependency) is session
    with pytest.raises(StopIteration):
        next(dependency)
    session.commit.assert_called_once()
    session.close.assert_called_once()
