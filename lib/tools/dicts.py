"""

helpers with dictionaries

"""

import json
from typing import Any

# region ObjectReflection


def from_object(obj: object, wanted_types=(int, float), round_float=6) -> dict:
    """
    Extract selected attributes from an object into a dictionary.

    Parameters
    ----------
    obj : object
        The object whose attributes will be inspected.
    wanted_types : tuple of types, optional
        A tuple of types to include (default: (int, float)).

    Returns
    -------
    dict
        A dictionary mapping attribute names to their values,
        only including attributes of the specified types.
    """
    result = {}
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        value = getattr(obj, attr)
        if isinstance(value, wanted_types):

            if round_float and isinstance(value, float):  # round floats
                value = round(value, round_float)

            result[attr] = value
    return result


def set_object(obj: object, d: dict) -> None:
    """
    Set attributes on an object from a dictionary.

    Parameters
    ----------
    obj : object
        The object whose attributes will be updated.
    d : dict
        A dictionary mapping attribute names to values.
        Keys must match existing attribute names on the object.

    Returns
    -------
    None
        The object is modified in place.
    """
    for attr, value in d.items():
        if hasattr(obj, attr):
            setattr(obj, attr, value)


def changes(old: dict, new: dict) -> dict:
    """
    Compare two dictionaries and return the keys whose values
    differ in the second dictionary.

    Parameters
    ----------
    old : dict
        The baseline dictionary (original settings).
    new : dict
        The updated dictionary (new settings).

    Returns
    -------
    dict
        A dictionary of {key: (old_value, new_value)} pairs
        for keys where the value in `new` differs from `old`.
        Keys present only in `new` are also included, with
        old_value set to None.
    """
    changes = {}
    for key, new_val in new.items():
        old_val = old.get(key, None)
        if old_val != new_val:
            changes[key] = new_val
    return changes


def to_string(pars: dict[str, Any], format_string: str = "{key} = {value}\n") -> str:
    result = ""
    for key, value in pars.items():
        result += format_string.format(key=key, value=value)
    return result


# endregion

# region JSON


def to_json(d: dict, filename: str) -> None:
    """
    Save a dictionary to a JSON file.

    Parameters
    ----------
    d : dict
        The dictionary to save.
    filename : str
        Path to the JSON file to write.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=4)


def from_json(filename: str) -> dict:
    """
    Load a dictionary from a JSON file.

    Parameters
    ----------
    filename : str
        Path to the JSON file to read.

    Returns
    -------
    dict
        The dictionary loaded from the file.
    """
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

# endregion
