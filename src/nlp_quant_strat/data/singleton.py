import threading
from typing import Any, Dict, TypeVar, Callable, cast

T = TypeVar("T")


class SingletonMeta(type):
    """
    Thread-safe singleton metaclass.

    A single instance is maintained per subclass using this metaclass.
    """

    _instances: Dict[type, Any] = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        """
        Return the unique instance for `cls`.

        Uses double-checked locking to avoid creating multiple instances
        in concurrent contexts.
        """
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)

        return cls._instances[cls]

    def clear_instances(cls) -> None:
        """
        Clear all singleton instances.

        Useful in tests.
        """
        with cls._lock:
            cls._instances.clear()

    def clear_instance(cls, target_class: type) -> None:
        """
        Clear the singleton instance for a specific class.

        Useful in tests.
        """
        with cls._lock:
            cls._instances.pop(target_class, None)


class Singleton(metaclass=SingletonMeta):
    """
    Base class for singleton objects.

    Subclasses should guard their own initialization if they want `__init__`
    logic to run only once.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass


def singleton(cls: type[T]) -> Callable[..., T]:
    """
    Decorator-based singleton implementation.

    Usage:
        @singleton
        class MyClass:
            ...

        instance_1 = MyClass()
        instance_2 = MyClass()
        assert instance_1 is instance_2
    """
    instance: T | None = None
    lock = threading.Lock()

    def get_instance(*args, **kwargs) -> T:
        nonlocal instance
        if instance is None:
            with lock:
                if instance is None:
                    instance = cls(*args, **kwargs)
        return cast(T, instance)

    return get_instance