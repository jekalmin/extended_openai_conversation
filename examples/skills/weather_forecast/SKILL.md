---
name: weather_forecast
description: Get a Home Assistant weather forecast for a location. Use when users ask for today's or tomorrow's forecast, or ask what the weather will be like.
---

# Weather forecast

## Requirements

This skill uses the `get_weather_forecasts` function documented in
[`examples/function/weather`](../../function/weather/README.md). Configure that function
and expose the weather entities the assistant should be allowed to query before enabling
this skill.

## Instructions

1. Identify the requested location from the user's wording and the names of the exposed
   weather entities. If there is only one exposed weather entity and the user did not name
   a location, use that entity. If multiple entities could match and the request does not
   identify one clearly, ask which location they mean. Do not invent or guess an entity ID.
2. Call `get_weather_forecasts` with that entity's exact ID and `type: "daily"`.
3. Use only the forecast data returned by the function. For today, use the first daily
   forecast entry; for tomorrow, use the second. If the requested entry or a value is
   missing, say it is unavailable instead of filling it in.
4. Give a concise spoken summary using the available condition, high and low temperatures,
   wind, and precipitation amount. Include units from the response when present. Do not
   invent a precipitation probability or infer that missing precipitation means no rain.
5. Name the location only when it is clear from the user's request or the selected weather
   entity's exposed name. Otherwise say "there" or ask for clarification.

## Examples

- “What's today's forecast?” — If one weather entity is exposed, fetch its daily forecast
  and summarize the first entry.
- “What's tomorrow's forecast in Ottawa?” — Use the exposed weather entity clearly named
  for Ottawa and summarize the second entry.
- “What's the forecast?” — If multiple exposed weather entities exist and none is an
  unambiguous default, ask which location the user means.

If `get_weather_forecasts` is unavailable, explain that the weather function needs to be
configured. Do not substitute remembered or external forecast data.
