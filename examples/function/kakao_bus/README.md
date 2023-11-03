## Function

### get_bus_info (scrape ver.)
```yaml
- spec:
    name: get_bus_info
    description: Use this function to get bus information
    parameters:
      type: object
      properties:
        dummy:
          type: string
          description: Nothing
  function:
    type: scrape
    resource: https://m.map.kakao.com/actions/busStationInfo?busStopId=BS219668
    value_template: "remain time: {{[next | trim, next_of_next | trim]}}"
    sensor:
      - name: next
        select: "li[data-id='1100061486'] span.txt_situation"
        index: 0
      - name: next_of_next
        select: "li[data-id='1100061486'] span.txt_situation"
        index: 1
```

### get_bus_info (rest ver.)
```yaml
- spec:
    name: get_bus_info
    description: Use this function to get bus information
    parameters:
      type: object
      properties:
        dummy:
          type: string
          description: Nothing.
  function:
    type: rest
    resource: https://m.map.kakao.com/actions/busesInBusStopJson?busStopId=BS219668
    value_template: '{{value_json["busesList"] | selectattr("id", "==", "1100061486") | map(attribute="vehicleStateMessage") | list }}'
```