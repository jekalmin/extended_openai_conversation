# Weather forecast skill

This skill answers today's and tomorrow's forecast questions from Home Assistant weather
entities. It does not include a location-specific entity ID or depend on one weather
provider.

## Setup

1. Configure the generic `get_weather_forecasts` function from
   [`examples/function/weather`](../../function/weather/README.md).
2. Expose the weather entities that the assistant may use. The skill uses the entity names
   and IDs available to the conversation to resolve locations. If more than one entity
   matches an unspecified location, it asks the user to choose.
3. Install this directory under
   `<config>/extended_openai_conversation/skills/weather_forecast/` and enable
   `weather_forecast` for the desired conversation.

The function is read-only and calls Home Assistant's `weather.get_forecasts` service. The
forecast fields depend on the weather integration; the skill uses fields that are present
and reports missing values as unavailable.
