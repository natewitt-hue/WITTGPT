# WittGPT — TSL Stat Hub

## File Structure
```
tsl_bot/
├── bot.py            # Discord bot, slash commands, Gemini calls
├── data_manager.py   # Loads & aggregates all league JSON files
├── analysis.py       # Query router, stat leaders, profiles, rankings
├── embeds.py         # Discord embed card builders
├── requirements.txt
└── league_data/      # Your JSON exports go here
    ├── info.json
    ├── offensive.json
    ├── defensive.json
    ├── teamStats.json
    ├── standings.json
    ├── teams.json
    ├── players.json
    ├── games.json
    ├── trades.json
    └── playerAbilities.json
```

## Setup
1. `pip install -r requirements.txt`
2. Create a `.env` file:
```
DISCORD_TOKEN=your_discord_bot_token
GEMINI_API_KEY=your_gemini_api_key
RECAP_CHANNEL_ID=0   # optional: channel ID for auto weekly recaps
```
3. Drop your `league_data/` folder next to `bot.py`
4. `python bot.py`

## Slash Commands
| Command | Description |
|---|---|
| `/stats [player]` | Full player stat card with bio, abilities, stats |
| `/team [name]` | Team profile — record, offense, defense, key players, form |
| `/matchup [team1] [team2]` | Side-by-side H2H comparison + all-time record |
| `/standings` | Full power rankings |
| `/recap [week?]` | Weekly game results + highlights |
| `/trades` | Recent trade history with WittGPT's take |
| `/form [team]` | Last 5 game results |

## @ Mention (Natural Language)
Mention the bot for anything:
- `@WittGPT worst QBs this season`
- `@WittGPT Chiefs vs Bengals`
- `@WittGPT how is Jalen Hurts doing`
- `@WittGPT power rankings`
- `@WittGPT red zone leaders`

## Auto Weekly Recap
Set `RECAP_CHANNEL_ID` in `.env` to a Discord channel ID and the bot will
automatically post weekly recaps + WittGPT's commentary once per day.

## When New Season Data Arrives
Data is loaded once at startup. Restart the bot after updating `league_data/`.
To hot-reload without restarting, future enhancement: add a `!reload` admin command.
