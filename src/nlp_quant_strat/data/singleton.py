import threading
from typing import Dict, Any


class SingletonMeta(type):
    """
    Metaclass that creates a singleton instance of a class.

    This implementation is thread-safe and ensures that only one instance
    of each class exists throughout the application lifecycle.
    """

    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        """
        Control the creation of class instances.

        Returns the existing instance if it exists, otherwise creates a new one.
        This method is thread-safe using double-checked locking pattern.
        """
        # Double-checked locking pattern for thread safety
        if cls not in cls._instances:
            with cls._lock:
                # Check again in case another thread created the instance
                # while we were waiting for the lock
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance

        return cls._instances[cls]

    def clear_instances(cls):
        """
        Clear all singleton instances. Useful for testing.
        """
        with cls._lock:
            cls._instances.clear()

    def clear_instance(cls, target_class: type):
        """
        Clear a specific singleton instance. Useful for testing.

        Args:
            target_class: The class whose instance should be cleared
        """
        with cls._lock:
            cls._instances.pop(target_class, None)


class Singleton(metaclass=SingletonMeta):
    """
    Base singleton class that other classes can inherit from.

    Example usage:
        class MyClass(Singleton):
            def __init__(self, value=None):
                # Only initialize once
                if not hasattr(self, 'initialized'):
                    self.value = value
                    self.initialized = True

        # Both instances will be the same object
        instance1 = MyClass("first")
        instance2 = MyClass("second")
        print(instance1 is instance2)  # True
        print(instance1.value)  # "first"
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the singleton instance.

        Note: This will only be called once per class, even if the constructor
        is called multiple times. Subclasses should implement their own logic
        to handle multiple initialization attempts if needed.
        """
        pass


# Decorator version for those who prefer decorator syntax
def singleton(cls):
    """
    Decorator version of the singleton pattern.

    Usage:
        @singleton
        class MyClass:
            def __init__(self, value):
                self.value = value
    """
    instances = {}
    lock = threading.Lock()

    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance