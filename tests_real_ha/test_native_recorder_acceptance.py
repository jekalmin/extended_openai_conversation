"""Native History and Statistics against a real Recorder SQLite database."""

from __future__ import annotations

from datetime import timedelta
import json
from types import SimpleNamespace

import pytest
import yaml

from custom_components.extended_openai_conversation_responses.const import (
    CONF_FUNCTION_TOOLS,
)
from custom_components.extended_openai_conversation_responses.exceptions import (
    EntityNotExposed,
)
from custom_components.extended_openai_conversation_responses.functions.native import (
    NativeFunction,
)
from custom_components.extended_openai_conversation_responses.ha_tool_result_compat import (
    tool_result_data,
)
from homeassistant.components import conversation, recorder
from homeassistant.components.homeassistant.exposed_entities import async_expose_entity
from homeassistant.components.recorder import statistics
from homeassistant.components.recorder.models.statistics import StatisticMeanType
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import llm
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util
from tests_real_ha.test_acceptance_lifecycle import _make_entry, _setup_entry

_EXPOSED = [{"entity_id": "sensor.recorder_acceptance"}]


@pytest.mark.asyncio
async def test_native_history_reads_real_recorder_and_serializes_tool_result(
    hass: HomeAssistant, tmp_path
) -> None:
    assert await async_setup_component(
        hass,
        "recorder",
        {"recorder": {"db_url": f"sqlite:///{tmp_path / 'history.db'}"}},
    )
    tool = {
        "spec": {
            "name": "acceptance_history",
            "description": "Read known recorded states.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_ids": {"type": "array", "items": {"type": "string"}}
                },
            },
        },
        "function": {"type": "native", "name": "get_history"},
        "enabled": True,
    }
    entry = _make_entry(
        "Recorder History Acceptance",
        include_ai_task=False,
        conversation_options={CONF_FUNCTION_TOOLS: yaml.safe_dump([tool])},
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    async_expose_entity(hass, conversation.DOMAIN, "sensor.recorder_acceptance", True)

    hass.states.async_set("sensor.recorder_acceptance", "before")
    await hass.async_block_till_done()
    before = hass.states.get("sensor.recorder_acceptance").last_changed
    hass.states.async_set("sensor.recorder_acceptance", "inside")
    await hass.async_block_till_done()
    inside = hass.states.get("sensor.recorder_acceptance").last_changed
    hass.states.async_set("sensor.recorder_acceptance", "later")
    hass.states.async_set("sensor.unrelated_acceptance", "private")
    await hass.async_block_till_done()
    later = hass.states.get("sensor.recorder_acceptance").last_changed
    await hass.async_block_till_done()
    await recorder.get_instance(hass).async_block_till_done()
    between = inside + (later - inside) / 2

    args = {
        "entity_ids": ["sensor.recorder_acceptance"],
        "start_time": (before + timedelta(microseconds=1)).isoformat(),
        "end_time": between.isoformat(),
        "include_start_time_state": False,
        "significant_changes_only": False,
    }
    raw = await NativeFunction().get_history(hass, {}, args, None, _EXPOSED)
    assert [[item["state"] for item in group] for group in raw] == [["inside"]], (
        raw,
        before,
        inside,
        later,
    )
    with_boundary = await NativeFunction().get_history(
        hass,
        {},
        {
            **args,
            "start_time": (inside + timedelta(microseconds=1)).isoformat(),
            "include_start_time_state": True,
        },
        None,
        _EXPOSED,
    )
    assert [[item["state"] for item in group] for group in with_boundary] == [
        ["inside"]
    ]

    result = await agent._execute_function_tool(
        tool,
        llm.ToolInput(
            id="recorder-history-call",
            tool_name="acceptance_history",
            tool_args=args,
            external=True,
        ),
        SimpleNamespace(context=Context(), device_id=None),
        _EXPOSED,
    )
    serialized = tool_result_data(result)
    assert "inside" in json.dumps(serialized)
    assert "private" not in json.dumps(serialized)


@pytest.mark.asyncio
async def test_native_statistics_reads_real_recorder_and_enforces_exposure(
    hass: HomeAssistant, tmp_path
) -> None:
    assert await async_setup_component(
        hass,
        "recorder",
        {"recorder": {"db_url": f"sqlite:///{tmp_path / 'statistics.db'}"}},
    )
    start = dt_util.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(
        hours=2
    )
    for statistic_id, source in (
        ("sensor.recorder_acceptance", "recorder"),
        ("external:acceptance", "external"),
        ("sensor.hidden_acceptance", "recorder"),
    ):
        metadata = {
            "statistic_id": statistic_id,
            "source": source,
            "name": statistic_id,
            "unit_of_measurement": "°C",
            "unit_class": "temperature",
            "has_mean": True,
            "mean_type": StatisticMeanType.ARITHMETIC,
            "has_sum": False,
        }
        values = [
            {"start": start, "mean": 21.5, "min": 20.0, "max": 23.0},
            {
                "start": start + timedelta(hours=1),
                "mean": 22.5,
                "min": 21.0,
                "max": 24.0,
            },
        ]
        if source == "recorder":
            statistics.async_import_statistics(hass, metadata, values)
        else:
            statistics.async_add_external_statistics(hass, metadata, values)
    await recorder.get_instance(hass).async_block_till_done()

    args = {
        "statistic_ids": ["sensor.recorder_acceptance", "external:acceptance"],
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(hours=2)).isoformat(),
        "period": "hour",
        "types": {"mean", "min", "max"},
        "units": {"temperature": "°F"},
    }
    result = await NativeFunction().get_statistics(hass, {}, args, None, _EXPOSED)
    assert set(result) == set(args["statistic_ids"])
    for rows in result.values():
        assert len(rows) == 2
        assert [row["start"] for row in rows] == [
            start.timestamp(),
            (start + timedelta(hours=1)).timestamp(),
        ]
        assert rows[0]["mean"] == pytest.approx(70.7)
        assert rows[0]["min"] == pytest.approx(68.0)
        assert rows[0]["max"] == pytest.approx(73.4)
        assert "sum" not in rows[0]
    assert "sensor.hidden_acceptance" not in result
    with pytest.raises(EntityNotExposed):
        await NativeFunction().get_statistics(
            hass,
            {},
            {**args, "statistic_ids": ["sensor.hidden_acceptance"]},
            None,
            _EXPOSED,
        )
