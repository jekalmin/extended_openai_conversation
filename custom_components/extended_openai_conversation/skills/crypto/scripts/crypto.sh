#!/bin/bash
# Fetch cryptocurrency prices from CoinGecko API
# Usage: ./crypto.sh <symbol>
# Example: ./crypto.sh btc

set -e

SYMBOL="${1:-btc}"

# Map common symbols to CoinGecko IDs
case "${SYMBOL,,}" in
    btc|bitcoin)
        COIN_ID="bitcoin"
        ;;
    eth|ethereum)
        COIN_ID="ethereum"
        ;;
    sol|solana)
        COIN_ID="solana"
        ;;
    xrp|ripple)
        COIN_ID="ripple"
        ;;
    ada|cardano)
        COIN_ID="cardano"
        ;;
    doge|dogecoin)
        COIN_ID="dogecoin"
        ;;
    *)
        COIN_ID="${SYMBOL,,}"
        ;;
esac

# Fetch price from CoinGecko (free, no API key required)
RESPONSE=$(curl -s "https://api.coingecko.com/api/v3/simple/price?ids=${COIN_ID}&vs_currencies=usd,krw&include_24hr_change=true")

# Check if response is valid
if echo "$RESPONSE" | grep -q "error"; then
    echo "{\"error\": \"Failed to fetch price for ${SYMBOL}\", \"response\": ${RESPONSE}}"
    exit 1
fi

# Add timestamp and format response
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "{\"coin\": \"${COIN_ID}\", \"timestamp\": \"${TIMESTAMP}\", \"data\": ${RESPONSE}}"
