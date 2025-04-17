import os
import json
import hashlib
import functools
import time as import_time
from typing import Any, Callable, TypeVar, cast

T = TypeVar("T")


class JsonEncoder(json.JSONEncoder):
    """Extended JSON encoder to handle more Python types."""

    def default(self, o):
        if hasattr(o, "__dict__"):
            return o.__dict__
        if isinstance(o, (set, frozenset)):
            return list(o)
        return str(o)


def hash_arg(arg) -> str:
    """Create a hash string for a single argument."""
    if isinstance(arg, (int, float, bool, str, type(None))):
        # Simple types are converted to string and hashed
        arg_str = str(arg)
    elif isinstance(arg, (list, tuple)):
        # For sequences, hash each element and join
        arg_str = "[" + ",".join(hash_arg(x) for x in arg) + "]"
    elif isinstance(arg, dict):
        # For dictionaries, hash each key-value pair and join
        arg_str = (
            "{" + ",".join(f"{k}:{hash_arg(v)}" for k, v in sorted(arg.items())) + "}"
        )
    else:
        # For complex objects, use their string representation
        arg_str = str(arg)

    # Create a hash for the string representation
    return hashlib.md5(arg_str.encode()).hexdigest()[:8]


def cache(cache_dir: str = "cache"):
    """
    A decorator that caches function results in files, one file per set of arguments.
    Files are named as: func_name-hash(arg1)-hash(arg2)...

    Args:
        cache_dir: Directory to store cache files (default: "cache")

    Returns:
        Decorated function that uses file-based caching
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        os.makedirs(cache_dir, exist_ok=True)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Generate a unique filename based on function name and arguments
            func_name = func.__name__
            arg_hashes = [hash_arg(arg) for arg in args]

            # Add keyword arguments as well, sorted by key
            for key, value in sorted(kwargs.items()):
                arg_hashes.append(f"{key}-{hash_arg(value)}")

            # Create the cache file path with hyphens between components
            filename_parts = [func_name] + arg_hashes
            cache_file = os.path.join(cache_dir, "-".join(filename_parts) + ".json")

            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r") as f:
                        cached_data = json.load(f)
                        # Return just the result part of the cached data
                        return cast(T, cached_data["result"])
                except (json.JSONDecodeError, KeyError) as e:
                    # Handle corrupted cache file or missing result key
                    os.remove(cache_file)

            # Cache miss or corrupted cache, call the function
            result = func(*args, **kwargs)

            # Save both arguments and result to cache
            try:
                cache_data = {
                    "args": args,
                    "kwargs": kwargs,
                    "result": result,
                    "cached_at": import_time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                with open(cache_file, "w") as f:
                    json.dump(cache_data, f, cls=JsonEncoder)
            except Exception as e:
                print(f"Warning: failed to cache result: {e}")

            return result

        return wrapper

    return decorator
