from uuid import uuid4
from unittest.mock import MagicMock

import pytest


class MockRedis:
    def __init__(self):
        self._data = {}
        self._sets = {}

    def smembers(self, key):
        # Return bytes to match real Redis behavior
        result = self._sets.get(key, set())
        return {k.encode() if isinstance(k, str) else k for k in result}

    def sadd(self, key, *values):
        if key not in self._sets:
            self._sets[key] = set()
        self._sets[key].update(values)
        return len(values)

    def srem(self, key, *values):
        if key in self._sets:
            self._sets[key].difference_update(values)
            return len(values)
        return 0

    def hset(self, key, field=None, value=None, mapping=None):
        if key not in self._data:
            self._data[key] = {}
        # Handle hset(key, field, value)
        if field is not None and value is not None:
            self._data[key][field.encode() if isinstance(field, str) else field] = value.encode() if isinstance(value, str) else value
        # Handle hset(key, mapping={...})
        if mapping:
            self._data[key].update({k.encode() if isinstance(k, str) else k: v.encode() if isinstance(v, str) else v for k, v in mapping.items()})
        return 1

    def hgetall(self, key):
        # Return bytes keys to match real Redis
        raw = self._data.get(key, {})
        return {k.encode() if isinstance(k, str) else k: v.encode() if isinstance(v, str) else v for k, v in raw.items()}

    def hget(self, key, field):
        return self._data.get(key, {}).get(field)

    def delete(self, *keys):
        count = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                count += 1
            if key in self._sets:
                del self._sets[key]
                count += 1
        return count

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value):
        self._data[key] = value
        return True

    def publish(self, channel, message):
        return 0

    def pubsub(self):
        return MockPubSub()


class MockPubSub:
    def __init__(self):
        self._subscriptions = {}

    def subscribe(self, channel):
        self._subscriptions[channel] = None

    def get_message(self, ignore_subscribe_messages=True, timeout=0.0):
        return None


@pytest.fixture
def mock_redis():
    return MockRedis()


@pytest.fixture
def room_service(mock_redis, monkeypatch):
    from app.services.room import RoomService

    monkeypatch.setattr("app.db.rooms_and_clients.redis.Redis", lambda **kwargs: mock_redis)
    service = RoomService()
    return service


@pytest.fixture
def client():
    from app.services.room import Client

    return Client(id_=uuid4(), name="test_client")
