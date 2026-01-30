---
name: crypto
description: Get real-time cryptocurrency prices. Use when users ask about Bitcoin, Ethereum, or other crypto prices and market data.
---

# Cryptocurrency Price Checker

## Instructions

1. **Identify the cryptocurrency** the user is asking about (Bitcoin, Ethereum, Solana, etc.)

2. **Execute the appropriate script**
   - `scripts/crypto.sh btc` - Get Bitcoin price
   - `scripts/crypto.sh eth` - Get Ethereum price
   - `scripts/crypto.sh sol` - Get Solana price
   - `scripts/crypto.sh doge` - Get Dogecoin price
   - `scripts/crypto.sh <symbol>` - Get any crypto price by symbol

3. **Format the response** with:
   - USD price (and local currency if applicable)
   - 24-hour price change percentage
   - Source attribution to CoinGecko
   - Note that crypto prices are volatile and change rapidly

## Examples

**User:** "What's the current Bitcoin price?"
**Action:** Execute `scripts/crypto.sh btc`
**Response:** "Bitcoin is currently trading at $67,234. It has changed +2.3% in the last 24 hours. (Source: CoinGecko)"

**User:** "How much is Ethereum worth?"
**Action:** Execute `scripts/crypto.sh eth`
**Response:** "Ethereum is currently priced at $3,456, down -1.2% over the past 24 hours. (Source: CoinGecko)"

**User:** "Check Solana and Dogecoin prices"
**Action:** Execute `scripts/crypto.sh sol` and `scripts/crypto.sh doge`
**Response:** "Current prices - Solana: $145 (+5.7% 24h), Dogecoin: $0.12 (+0.8% 24h). (Source: CoinGecko)"
