"""Versioned model capabilities and conservative compatibility helpers."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, cast

from packaging.version import InvalidVersion, Version

EOAI_VERSION = Version(
    json.loads((Path(__file__).parent / "manifest.json").read_text("utf-8"))["version"]
)

MAX_CATALOG_BYTES = 256 * 1024
SUPPORTED_SCHEMA_VERSIONS = frozenset({5, 6})
CURRENT_SCHEMA_VERSION = 6
COMPATIBILITY_MESSAGE = (
    "A newer model catalogue is available, but it requires a newer version "
    "of Extended OpenAI Conversation."
)


class IncompatibleCatalogError(ValueError):
    """A well-formed catalogue envelope requires newer consumer code."""

    def __init__(
        self,
        *,
        schema_version: int,
        catalog_version: int,
        minimum_eoai_version: str | None = None,
    ) -> None:
        super().__init__(COMPATIBILITY_MESSAGE)
        self.schema_version = schema_version
        self.catalog_version = catalog_version
        self.minimum_eoai_version = minimum_eoai_version


_ID = re.compile(r"[a-z0-9][a-z0-9._:-]{0,127}\Z")
_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
_SUPPORT = {"always", "conditional", "never", "undocumented"}
_SEND_POLICIES = {"omit", "omit_unless_configured"}
_STATUSES = {"current", "deprecated", "unknown"}
_APIS = {"responses", "chat_completions"}
_METADATA_REQUIRED = {
    "status",
    "api",
    "function_calling",
    "reasoning",
    "temperature",
    "top_p",
    "limits",
    "streaming",
    "output_tokens",
    "recommended_profile",
    "service_tiers",
    "explicit_prompt_cache",
    "structured_outputs",
    "responses_web_search",
    "tools",
}
_MODEL_WRAPPER_KEYS = {"id", "display_name", "kind"}
_METADATA_OPTIONAL = {"alias_of", "lifecycle_note", "auto_api"}


class _PreparedCatalog(dict[str, Any]):
    """Raw catalogue document with resolved records kept out of persistence."""

    def __init__(
        self, raw: dict[str, Any], resolved: dict[str, dict[str, Any]]
    ) -> None:
        super().__init__(raw)
        self.resolved = resolved
        self.merged: dict[str, Any] | None = None

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        # Copies are editable candidate documents and must be validated again.
        return deepcopy(dict(self), memo)


def _keys(value: Any, required: set[str], optional: set[str] | None = None) -> None:
    if (
        not isinstance(value, dict)
        or not required <= value.keys()
        or value.keys() - required - (optional or set())
    ):
        raise ValueError("Unexpected or missing catalogue fields")


def _bool_map(value: Any, keys: set[str], label: str) -> None:
    _keys(value, keys)
    if any(type(value[key]) is not bool for key in keys):
        raise ValueError(f"{label} values must be boolean")


def _validate_sampling(value: Any, efforts: list[str], label: str) -> None:
    _keys(value, {"support", "allowed_reasoning_efforts", "send_policy"})
    support = value["support"]
    allowed = value["allowed_reasoning_efforts"]
    if support not in _SUPPORT or value["send_policy"] not in _SEND_POLICIES:
        raise ValueError(f"Invalid {label} capability")
    if support == "conditional":
        if (
            not isinstance(allowed, list)
            or not allowed
            or any(not isinstance(item, str) for item in allowed)
            or len(set(allowed)) != len(allowed)
            or any(item not in efforts for item in allowed)
        ):
            raise ValueError(f"Invalid conditional {label} reasoning efforts")
    elif allowed is not None:
        raise ValueError(f"{label} allowed_reasoning_efforts must be null")
    if support in {"never", "undocumented"} and value["send_policy"] != "omit":
        raise ValueError(f"{label} must be omitted when unsupported or undocumented")


def _validate_tool_rule(rule: Any, efforts: list[str]) -> None:
    _keys(rule, {"support"}, {"requires", "excludes"})
    if rule["support"] not in {"always", "conditional", "never"}:
        raise ValueError("Invalid tool support")
    conditions = 0
    for operator in ("requires", "excludes"):
        if operator not in rule:
            continue
        conditions += 1
        dimensions = rule[operator]
        if (
            not isinstance(dimensions, dict)
            or not dimensions
            or dimensions.keys() - {"reasoning_effort"}
        ):
            raise ValueError("Invalid tool condition")
        for allowed in dimensions.values():
            domain = set(efforts)
            if (
                not isinstance(allowed, list)
                or not allowed
                or any(
                    not isinstance(item, str) or item not in domain for item in allowed
                )
                or len(set(allowed)) != len(allowed)
            ):
                raise ValueError("Invalid tool condition values")
    if (rule["support"] == "conditional") != bool(conditions):
        raise ValueError("Conditional tool support requires conditions")


def _validate_metadata(value: dict[str, Any], *, model_entry: bool = False) -> None:
    optional = set(_METADATA_OPTIONAL)
    if model_entry:
        optional |= _MODEL_WRAPPER_KEYS
    _keys(value, _METADATA_REQUIRED, optional)
    if value["status"] not in _STATUSES:
        raise ValueError("Invalid model lifecycle status")

    _bool_map(
        value["api"],
        {"responses", "chat_completions", "completions_legacy"},
        "API capability",
    )
    auto_api = value.get("auto_api")
    if auto_api is not None and (auto_api not in _APIS or not value["api"][auto_api]):
        raise ValueError("Invalid Auto API preference")

    functions = value["function_calling"]
    _keys(functions, {"responses", "chat_completions", "preferred_api"})
    if (
        functions["preferred_api"] not in _APIS
        or not value["api"][functions["preferred_api"]]
    ):
        raise ValueError("Invalid preferred function-calling API")

    reasoning = value["reasoning"]
    _keys(reasoning, {"supported", "efforts", "by_api", "openai_default"})
    efforts = reasoning["efforts"]
    if type(reasoning["supported"]) is not bool or not isinstance(efforts, list):
        raise ValueError("Invalid reasoning capability")
    if (
        len(efforts) > len(_EFFORTS)
        or len(set(efforts)) != len(efforts)
        or any(not isinstance(item, str) or item not in _EFFORTS for item in efforts)
        or reasoning["supported"] != bool(efforts)
    ):
        raise ValueError("Invalid reasoning efforts")
    if (
        reasoning["openai_default"] is not None
        and reasoning["openai_default"] not in efforts
    ):
        raise ValueError("Invalid OpenAI reasoning default")
    _keys(reasoning["by_api"], _APIS)
    for api in _APIS:
        _keys(reasoning["by_api"][api], {"efforts"})
        api_efforts = reasoning["by_api"][api]["efforts"]
        if (
            not isinstance(api_efforts, list)
            or any(not isinstance(item, str) for item in api_efforts)
            or len(set(api_efforts)) != len(api_efforts)
            or any(item not in efforts for item in api_efforts)
            or (api_efforts and not value["api"][api])
        ):
            raise ValueError("Invalid API-specific reasoning efforts")
    if set(efforts) != set().union(
        *(set(reasoning["by_api"][api]["efforts"]) for api in _APIS)
    ):
        raise ValueError("Model-wide reasoning efforts must equal the API union")
    _keys(value["tools"], {"function", "web_search"})
    for name in ("function", "web_search"):
        _keys(value["tools"][name], _APIS)
        for api in _APIS:
            _validate_tool_rule(
                value["tools"][name][api], reasoning["by_api"][api]["efforts"]
            )
            if (
                not value["api"][api]
                and value["tools"][name][api]["support"] != "never"
            ):
                raise ValueError("Tool support requires an available API")
    for api in _APIS:
        support = functions[api]
        if type(support) is bool:
            continue
        _keys(support, {"support", "allowed_reasoning_efforts"})
        if support["support"] != "conditional":
            raise ValueError("Invalid function-calling support")
        allowed = support["allowed_reasoning_efforts"]
        if (
            not isinstance(allowed, list)
            or not allowed
            or any(not isinstance(item, str) for item in allowed)
            or len(set(allowed)) != len(allowed)
            or any(item not in efforts for item in allowed)
        ):
            raise ValueError("Invalid conditional function-calling efforts")

    _validate_sampling(value["temperature"], efforts, "temperature")
    _validate_sampling(value["top_p"], efforts, "top_p")

    limits = value["limits"]
    _keys(limits, {"context_tokens", "max_output_tokens"})
    if any(type(limits[key]) is not int or limits[key] <= 0 for key in limits):
        raise ValueError("Model token limits must be positive integers")
    if type(value["streaming"]) is not bool:
        raise ValueError("Streaming capability must be boolean")

    output_tokens = value["output_tokens"]
    _keys(output_tokens, {"responses", "chat_completions", "legacy_max_tokens"})
    if output_tokens != {
        "responses": "max_output_tokens",
        "chat_completions": "max_completion_tokens",
        "legacy_max_tokens": "never_send",
    }:
        raise ValueError("Invalid output-token parameter mapping")

    profile = value["recommended_profile"]
    _keys(profile, {"api", "reasoning_effort", "temperature", "top_p"})
    if profile["api"] not in _APIS or not value["api"][profile["api"]]:
        raise ValueError("Invalid recommended API")
    if (
        profile["reasoning_effort"] is not None
        and profile["reasoning_effort"] not in efforts
    ):
        raise ValueError("Invalid recommended reasoning effort")
    if (
        profile["reasoning_effort"] is not None
        and profile["reasoning_effort"]
        not in reasoning["by_api"][profile["api"]]["efforts"]
    ):
        raise ValueError("Invalid recommended reasoning effort for selected API")
    if profile["temperature"] != "omit" or profile["top_p"] != "omit":
        raise ValueError("Recommended sampling profile must omit temperature/top_p")
    service_tiers = value["service_tiers"]
    allowed_tiers = {"auto", "default", "flex", "fast", "priority", "ultrafast"}
    if (
        not isinstance(service_tiers, list)
        or len(set(service_tiers)) != len(service_tiers)
        or any(
            not isinstance(item, str) or item not in allowed_tiers
            for item in service_tiers
        )
        or type(value["explicit_prompt_cache"]) is not bool
        or type(value["structured_outputs"]) is not bool
        or type(value["responses_web_search"]) is not bool
    ):
        raise ValueError("Invalid service-tier or compatibility metadata")

    alias_of = value.get("alias_of")
    if alias_of is not None and (
        not isinstance(alias_of, str) or not _ID.fullmatch(alias_of)
    ):
        raise ValueError("Invalid alias target")
    note = value.get("lifecycle_note")
    if note is not None and (not isinstance(note, str) or len(note) > 512):
        raise ValueError("Invalid lifecycle note")


def _merge_snapshot(parent: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    """Copy inherited metadata and apply only explicitly declared overrides."""
    result = deepcopy(parent)
    for key, override in snapshot.items():
        if isinstance(override, dict) and isinstance(result.get(key), dict):
            result[key] = {**result[key], **deepcopy(override)}
        else:
            result[key] = deepcopy(override)
    return result


def validate_catalog(value: Any) -> _PreparedCatalog:
    """Validate raw catalogue data and prepare exact snapshot capabilities."""
    if not isinstance(value, dict):
        raise ValueError("Invalid model catalogue envelope")
    schema = value.get("schema_version")
    version = value.get("catalog_version")
    if type(schema) is not int or type(version) is not int or version < 1:
        raise ValueError("Invalid model catalogue version")
    compatibility = value.get("compatibility")
    minimum = None
    if compatibility is not None:
        _keys(compatibility, {"minimum_eoai_version"})
        minimum = compatibility["minimum_eoai_version"]
        if not isinstance(minimum, str) or len(minimum) > 64:
            raise ValueError("Invalid minimum EOAI version")
        try:
            required_version = Version(minimum)
        except InvalidVersion as exc:
            raise ValueError("Invalid minimum EOAI version") from exc
    if schema not in SUPPORTED_SCHEMA_VERSIONS or (
        minimum is not None and required_version > EOAI_VERSION
    ):
        # Do not inspect unknown semantics or project any of their fields.
        if not isinstance(value.get("defaults"), dict) or not isinstance(
            value.get("models"), list
        ):
            raise ValueError("Invalid model catalogue envelope")
        raise IncompatibleCatalogError(
            schema_version=schema,
            catalog_version=version,
            minimum_eoai_version=minimum,
        )
    _keys(
        value,
        {"schema_version", "catalog_version", "defaults", "models"}
        | ({"compatibility"} if schema == 6 else set()),
        {"compatibility"} if schema == 5 else set(),
    )
    if value["catalog_version"] < 7:
        raise ValueError("catalog_version must be at least 7")
    _validate_metadata(value["defaults"])
    if value["defaults"]["status"] != "unknown":
        raise ValueError("Catalogue defaults must describe unknown models")

    models = value["models"]
    if not isinstance(models, list) or not 1 <= len(models) <= 512:
        raise ValueError("Invalid model list")
    ids: set[str] = set()
    by_id: dict[str, dict[str, Any]] = {}
    for model in models:
        _keys(model, _MODEL_WRAPPER_KEYS, _METADATA_OPTIONAL | _METADATA_REQUIRED)
        model_id = model["id"]
        if (
            not isinstance(model_id, str)
            or not _ID.fullmatch(model_id)
            or model_id in ids
        ):
            raise ValueError("Invalid or duplicate model ID")
        ids.add(model_id)
        by_id[model_id] = model
        if (
            not isinstance(model["display_name"], str)
            or not 1 <= len(model["display_name"]) <= 128
        ):
            raise ValueError("Invalid display name")
        if model["kind"] not in {"alias", "snapshot"}:
            raise ValueError("Invalid model kind")
        if model["kind"] != "snapshot" or "alias_of" not in model:
            _keys(model, _MODEL_WRAPPER_KEYS | _METADATA_REQUIRED, _METADATA_OPTIONAL)

    bundled = globals().get("BUNDLED_CATALOG")
    bundled_by_id = bundled.resolved if isinstance(bundled, _PreparedCatalog) else {}
    resolved: dict[str, dict[str, Any]] = {}
    resolving: set[str] = set()

    def resolve(model_id: str) -> dict[str, Any]:
        if model_id in resolved:
            return resolved[model_id]
        if model_id in resolving:
            raise ValueError("Snapshot inheritance cycle")
        model = by_id[model_id]
        resolving.add(model_id)
        parent_id = model.get("alias_of") if model["kind"] == "snapshot" else None
        if parent_id is not None:
            if not isinstance(parent_id, str) or not _ID.fullmatch(parent_id):
                raise ValueError("Invalid snapshot parent")
            if parent_id in by_id:
                parent = resolve(parent_id)
            elif parent_id in bundled_by_id:
                parent = bundled_by_id[parent_id]
            else:
                raise ValueError("Snapshot parent is missing")
            if (
                parent["kind"] not in {"alias", "snapshot"}
                or parent["status"] == "unknown"
            ):
                raise ValueError("Invalid snapshot parent kind or status")
            effective = _merge_snapshot(parent, model)
            if effective["status"] == "current" and parent["status"] == "deprecated":
                raise ValueError("Current snapshot cannot inherit a deprecated model")
        else:
            effective = deepcopy(model)
        _validate_metadata(effective, model_entry=True)
        if model["kind"] == "alias" and model.get("alias_of") is not None:
            alias_of = model["alias_of"]
            if alias_of == model_id or (
                alias_of not in by_id and alias_of not in bundled_by_id
            ):
                raise ValueError("Alias target must reference another catalogue model")
        resolved[model_id] = effective
        resolving.remove(model_id)
        return effective

    for model_id in by_id:
        resolve(model_id)
    return _PreparedCatalog(deepcopy(dict(value)), resolved)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def parse_catalog(raw: bytes) -> _PreparedCatalog:
    if len(raw) > MAX_CATALOG_BYTES:
        raise ValueError("Model catalogue too large")
    return validate_catalog(json.loads(raw, object_pairs_hook=_unique_object))


BUNDLED_CATALOG = parse_catalog(Path(__file__).with_suffix(".json").read_bytes())
_bundled_effective = {
    **BUNDLED_CATALOG,
    "models": list(BUNDLED_CATALOG.resolved.values()),
}
_active = _bundled_effective
_active_by_id = BUNDLED_CATALOG.resolved


def migrate_catalog_v1(value: Any) -> dict[str, Any]:
    """Replace ambiguous v1 booleans with authoritative bundled metadata."""
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Not a model catalogue v1 document")
    migrated = deepcopy(BUNDLED_CATALOG)
    old_version = value.get("catalog_version")
    if type(old_version) is int:
        migrated["catalog_version"] = max(migrated["catalog_version"], old_version + 1)
    return migrated


def migrate_catalog_v2(value: Any) -> dict[str, Any]:
    """Preserve v2 metadata while replacing service-tier booleans with exact lists."""
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise ValueError("Not a model catalogue v2 document")
    migrated = deepcopy(value)
    migrated["schema_version"] = 3
    migrated["catalog_version"] = max(3, int(migrated.get("catalog_version", 0)))
    bundled_tiers = {
        item["id"]: list(item["service_tiers"])
        for item in BUNDLED_CATALOG.resolved.values()
    }

    def migrate_metadata(metadata: dict[str, Any], model_id: str | None = None) -> None:
        supported = metadata.pop("service_tier", False)
        metadata["service_tiers"] = (
            bundled_tiers.get(model_id, ["auto", "default"]) if supported else []
        )

    migrate_metadata(migrated["defaults"])
    for model in migrated["models"]:
        migrate_metadata(model, model.get("id"))
    return migrate_catalog_v3(migrated)


def migrate_catalog_v3(value: Any) -> dict[str, Any]:
    """Retain v3 function booleans and prior hosted/structured behavior."""
    if not isinstance(value, dict) or value.get("schema_version") != 3:
        raise ValueError("Not a model catalogue v3 document")
    migrated = deepcopy(value)
    migrated["schema_version"] = 4
    migrated["catalog_version"] = max(4, migrated["catalog_version"])
    migrated["defaults"]["structured_outputs"] = False
    migrated["defaults"]["responses_web_search"] = False
    for model in migrated["models"]:
        model.setdefault("structured_outputs", True)
        model.setdefault("responses_web_search", bool(model["api"]["responses"]))
    return migrate_catalog_v4(migrated)


def migrate_catalog_v4(value: Any) -> dict[str, Any]:
    """Keep legacy booleans readable while adding explicit API/tool rules."""
    if not isinstance(value, dict) or value.get("schema_version") != 4:
        raise ValueError("Not a model catalogue v4 document")
    migrated = deepcopy(value)
    migrated["schema_version"] = 5
    migrated["catalog_version"] = max(7, migrated["catalog_version"])
    for metadata in (migrated["defaults"], *migrated["models"]):
        reasoning = metadata.get("reasoning")
        if reasoning is not None:
            reasoning["by_api"] = {
                api: {
                    "efforts": list(reasoning["efforts"])
                    if metadata["api"][api]
                    else []
                }
                for api in _APIS
            }
        if "function_calling" in metadata and "responses_web_search" in metadata:
            metadata["tools"] = {
                "function": {
                    api: _legacy_tool_rule(metadata["function_calling"][api])
                    for api in _APIS
                },
                "web_search": {
                    "responses": _legacy_tool_rule(metadata["responses_web_search"]),
                    "chat_completions": {"support": "never"},
                },
            }
    return validate_catalog(migrated)


def _legacy_tool_rule(value: Any) -> dict[str, Any]:
    if type(value) is bool:
        return {"support": "always" if value else "never"}
    return {
        "support": "conditional",
        "requires": {"reasoning_effort": list(value["allowed_reasoning_efforts"])},
    }


def validate_or_migrate_catalog(value: Any) -> tuple[dict[str, Any], bool]:
    if isinstance(value, dict) and value.get("schema_version") == 1:
        return migrate_catalog_v1(value), True
    if isinstance(value, dict) and value.get("schema_version") == 2:
        return migrate_catalog_v2(value), True
    if isinstance(value, dict) and value.get("schema_version") == 3:
        return migrate_catalog_v3(value), True
    if isinstance(value, dict) and value.get("schema_version") == 4:
        return migrate_catalog_v4(value), True
    return validate_catalog(value), False


def _effective_catalog(catalog: dict[str, Any] | None) -> dict[str, Any]:
    if catalog is None:
        return _bundled_effective
    prepared = (
        catalog if isinstance(catalog, _PreparedCatalog) else validate_catalog(catalog)
    )
    if prepared.merged is None:
        raw_models = {item["id"]: item for item in BUNDLED_CATALOG["models"]}
        raw_models.update({item["id"]: item for item in prepared["models"]})
        if raw_models.keys() == prepared.resolved.keys():
            resolved = prepared.resolved
        else:
            merged = validate_catalog({**prepared, "models": list(raw_models.values())})
            resolved = merged.resolved
        prepared.merged = {**prepared, "models": list(resolved.values())}
    return prepared.merged


def _model_metadata_from(catalog: dict[str, Any], model: str) -> dict[str, Any]:
    model_id = str(model or "").lower()
    for item in catalog["models"]:
        if item["id"] == model_id:
            return cast(dict[str, Any], deepcopy(item))
    result = cast(dict[str, Any], deepcopy(catalog["defaults"]))
    result.update({"id": model_id, "display_name": model_id, "kind": "custom"})
    return result


def catalog_model_metadata(
    catalog: dict[str, Any] | None, model: str
) -> dict[str, Any]:
    return _model_metadata_from(_effective_catalog(catalog), model)


def catalog_reasoning_efforts(catalog: dict[str, Any] | None) -> list[str]:
    effective = _effective_catalog(catalog)
    return list(
        dict.fromkeys(
            e for item in effective["models"] for e in item["reasoning"]["efforts"]
        )
    )


def catalog_picker_models(
    catalog: dict[str, Any] | None = None, selected_model: str | None = None
) -> list[dict[str, Any]]:
    """Return current models plus an exact selected deprecated/custom model."""
    effective = _effective_catalog(catalog)
    selected = str(selected_model or "").lower()
    result = [
        {
            "id": item["id"],
            "display_name": item["display_name"],
            "status": item["status"],
            "lifecycle_note": item.get("lifecycle_note"),
        }
        for item in effective["models"]
        if item["status"] == "current"
    ]
    if selected and selected not in {item["id"] for item in result}:
        metadata = _model_metadata_from(effective, selected)
        result.append(
            {
                "id": selected,
                "display_name": metadata.get("display_name", selected),
                "status": metadata["status"],
                "lifecycle_note": metadata.get("lifecycle_note"),
            }
        )
    return result


def _sampling_rank(capability: dict[str, Any]) -> tuple[int, frozenset[str]]:
    support = capability["support"]
    if support == "always":
        return 3, frozenset()
    if support == "conditional":
        return 2, frozenset(capability["allowed_reasoning_efforts"] or [])
    if support == "undocumented":
        return 1, frozenset()
    return 0, frozenset()


def function_calling_allowed(
    support: bool | dict[str, Any], effort: str | None
) -> bool:
    """Evaluate the small function capability shape for one resolved effort."""
    if type(support) is bool:
        return support
    return effort in support["allowed_reasoning_efforts"]


def evaluate_tool_rule(
    rule: dict[str, Any],
    *,
    effort: str | None = None,
    service_tier: str | None = None,
    streaming: bool | None = None,
) -> bool:
    if rule["support"] == "never":
        return False
    values = {
        "reasoning_effort": effort,
        "service_tier": service_tier,
        "streaming": streaming,
    }
    return all(
        values[key] in allowed for key, allowed in rule.get("requires", {}).items()
    ) and all(
        values[key] not in excluded
        for key, excluded in rule.get("excludes", {}).items()
    )


def _function_support_set(support: bool | dict[str, Any]) -> frozenset[str] | None:
    if support is True:
        return None  # All efforts.
    if support is False:
        return frozenset()
    return frozenset(support["allowed_reasoning_efforts"])


def validate_catalog_transition(
    current: dict[str, Any] | None, candidate: dict[str, Any]
) -> None:
    """Prevent downloaded metadata from silently narrowing durable capabilities."""
    before = _effective_catalog(current)
    after = _effective_catalog(candidate)
    ids = {item["id"] for item in before["models"]} | {
        item["id"] for item in after["models"]
    }
    for model_id in ids:
        old = _model_metadata_from(before, model_id)
        new = _model_metadata_from(after, model_id)
        if old["status"] == "unknown":
            # Newly documented models may legitimately be narrower than the
            # conservative unknown defaults. The manager checks saved choices.
            continue
        if not set(old["reasoning"]["efforts"]).issubset(new["reasoning"]["efforts"]):
            raise ValueError("Catalogue update cannot remove reasoning effort choices")
        for api in _APIS:
            if old["api"][api] and not new["api"][api]:
                raise ValueError("Catalogue update cannot remove an API path")
            old_allowed = _function_support_set(old["function_calling"][api])
            new_allowed = _function_support_set(new["function_calling"][api])
            if old_allowed != frozenset() and new_allowed == frozenset():
                raise ValueError(
                    "Catalogue update cannot remove function-calling support"
                )
            for tool in ("function", "web_search"):
                if (
                    old["tools"][tool][api]["support"] != "never"
                    and new["tools"][tool][api]["support"] == "never"
                ):
                    raise ValueError("Catalogue update cannot remove tool support")
        for capability in ("structured_outputs", "responses_web_search", "streaming"):
            if old[capability] and not new[capability]:
                raise ValueError(f"Catalogue update cannot remove {capability} support")
        if old["limits"]["max_output_tokens"] > new["limits"]["max_output_tokens"]:
            raise ValueError(
                "Catalogue update cannot lower max output without migration"
            )
        if not set(old["service_tiers"]).issubset(new["service_tiers"]):
            raise ValueError("Catalogue update cannot remove service-tier choices")
        for name in ("temperature", "top_p"):
            old_rank, old_allowed = _sampling_rank(old[name])
            new_rank, new_allowed = _sampling_rank(new[name])
            if old_rank > new_rank or (
                old[name]["support"] == new[name]["support"] == "conditional"
                and not old_allowed.issubset(new_allowed)
            ):
                raise ValueError(
                    f"Catalogue update cannot narrow {name} without migration"
                )


def activate_catalog(catalog: dict[str, Any] | None) -> None:
    global _active, _active_by_id
    _active = _effective_catalog(catalog)
    _active_by_id = {item["id"]: item for item in _active["models"]}


def all_reasoning_efforts() -> list[str]:
    return list(
        dict.fromkeys(
            e for item in _active["models"] for e in item["reasoning"]["efforts"]
        )
    )


def model_metadata(model: str) -> dict[str, Any]:
    """Return exact-ID metadata; unknown IDs receive conservative capabilities."""
    model_id = str(model or "").lower()
    item = _active_by_id.get(model_id)
    if item is not None:
        return deepcopy(item)
    result = cast(dict[str, Any], deepcopy(_active["defaults"]))
    result.update({"id": model_id, "display_name": model_id, "kind": "custom"})
    return result


def compatibility_capabilities(
    model: str,
    *,
    effort: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose legacy projections without making them authoritative."""
    del effort
    metadata = metadata if metadata is not None else model_metadata(model)
    evaluations = {
        api: {
            (candidate if candidate is not None else "null"): {
                "reasoning": candidate is None
                or candidate in metadata["reasoning"]["by_api"][api]["efforts"],
                "function": evaluate_tool_rule(
                    metadata["tools"]["function"][api],
                    effort=candidate,
                    streaming=metadata["streaming"],
                ),
                "web_search": evaluate_tool_rule(
                    metadata["tools"]["web_search"][api],
                    effort=candidate,
                    streaming=metadata["streaming"],
                ),
            }
            for candidate in (None, *metadata["reasoning"]["efforts"])
        }
        for api in _APIS
    }
    return {
        "supports_top_p": metadata["top_p"]["support"] in {"always", "conditional"},
        "supports_temperature": metadata["temperature"]["support"]
        in {"always", "conditional"},
        "supports_max_tokens": False,
        "supports_max_completion_tokens": True,
        "supports_reasoning_effort": metadata["reasoning"]["supported"],
        "supports_service_tier": bool(metadata["service_tiers"]),
        "service_tier_options": list(metadata["service_tiers"]),
        "reasoning_effort_options": list(metadata["reasoning"]["efforts"]),
        "api": deepcopy(metadata["api"]),
        "auto_api": metadata.get("auto_api"),
        "function_calling": deepcopy(metadata["function_calling"]),
        "tools": deepcopy(metadata["tools"]),
        "evaluations": evaluations,
        "reasoning": deepcopy(metadata["reasoning"]),
        "temperature": deepcopy(metadata["temperature"]),
        "top_p": deepcopy(metadata["top_p"]),
        "limits": deepcopy(metadata["limits"]),
        "streaming": metadata["streaming"],
        "structured_outputs": metadata["structured_outputs"],
        "responses_web_search": metadata["responses_web_search"],
        "output_tokens": deepcopy(metadata["output_tokens"]),
        "recommended_profile": deepcopy(metadata["recommended_profile"]),
        "status": metadata["status"],
        "alias_of": metadata.get("alias_of"),
        "lifecycle_note": metadata.get("lifecycle_note"),
    }
