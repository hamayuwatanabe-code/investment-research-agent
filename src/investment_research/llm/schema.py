"""Minimal JSON-Schema validation for LLM output (requirement P2).

Deliberately hand-rolled and dependency-free, matching the rest of the core. It
covers exactly the subset the agent schemas use: objects, arrays, strings with
enums, numbers with bounds, booleans, required keys and
``additionalProperties: false``.

The rule this enforces is the one that matters: **a model response that does not
validate is a failed agent run, not a partial result.** Silently accepting a
malformed structure is how an unchecked hallucination reaches a report.
"""

from __future__ import annotations

from typing import Any


class SchemaValidationError(ValueError):
    """Raised when an LLM response does not match its declared schema."""


def validate(instance: Any, schema: dict[str, Any], path: str = "$") -> None:
    kind = schema.get("type")

    if kind == "object":
        if not isinstance(instance, dict):
            raise SchemaValidationError(f"{path}: expected object, got {type(instance).__name__}")
        properties: dict[str, Any] = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                raise SchemaValidationError(f"{path}: missing required property {key!r}")
        if schema.get("additionalProperties") is False:
            extra = set(instance) - set(properties)
            if extra:
                raise SchemaValidationError(
                    f"{path}: unexpected properties {sorted(extra)}"
                )
        for key, value in instance.items():
            if key in properties:
                validate(value, properties[key], f"{path}.{key}")
        return

    if kind == "array":
        if not isinstance(instance, list):
            raise SchemaValidationError(f"{path}: expected array, got {type(instance).__name__}")
        max_items = schema.get("maxItems")
        if max_items is not None and len(instance) > max_items:
            raise SchemaValidationError(
                f"{path}: {len(instance)} items exceeds maxItems {max_items}"
            )
        min_items = schema.get("minItems")
        if min_items is not None and len(instance) < min_items:
            raise SchemaValidationError(
                f"{path}: {len(instance)} items below minItems {min_items}"
            )
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(instance):
                validate(item, item_schema, f"{path}[{index}]")
        return

    if kind == "string":
        if not isinstance(instance, str):
            raise SchemaValidationError(f"{path}: expected string, got {type(instance).__name__}")
        choices = schema.get("enum")
        if choices is not None and instance not in choices:
            raise SchemaValidationError(
                f"{path}: {instance!r} is not one of {choices}"
            )
        max_length = schema.get("maxLength")
        if max_length is not None and len(instance) > max_length:
            raise SchemaValidationError(f"{path}: string longer than {max_length}")
        return

    if kind in ("number", "integer"):
        if isinstance(instance, bool) or not isinstance(instance, (int, float)):
            raise SchemaValidationError(f"{path}: expected {kind}, got {type(instance).__name__}")
        if kind == "integer" and not float(instance).is_integer():
            raise SchemaValidationError(f"{path}: expected integer, got {instance}")
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and instance < minimum:
            raise SchemaValidationError(f"{path}: {instance} below minimum {minimum}")
        if maximum is not None and instance > maximum:
            raise SchemaValidationError(f"{path}: {instance} above maximum {maximum}")
        return

    if kind == "boolean":
        if not isinstance(instance, bool):
            raise SchemaValidationError(f"{path}: expected boolean, got {type(instance).__name__}")
        return

    if kind is None:
        return

    raise SchemaValidationError(f"{path}: unsupported schema type {kind!r}")


def as_strict_tool(name: str, description: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap a schema as a strict tool definition.

    ``strict: true`` makes the API guarantee the arguments validate against the
    schema, which removes the most common class of parse failure. Local
    validation still runs afterwards: the guarantee covers shape, not content.
    """
    return {
        "name": name,
        "description": description,
        "input_schema": schema,
        "strict": True,
    }
