from collections import abc


def map_nested_dicts(ob, func):
    if isinstance(ob, abc.Mapping):
        return {k: map_nested_dicts(v, func) for k, v in ob.items()}
    else:
        try:
            return func(ob)
        except ValueError:
            return ob