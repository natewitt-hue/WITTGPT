"""
rules.py

TSL Revolution Rulebook v26 integration.

Three capabilities:
  1. RULEBOOK_TEXT   — full structured rulebook fed to Gemini as context
  2. rule_lookup()   — keyword router that pulls the exact relevant rule section(s)
  3. validate_position_change() — live check of a player's stats against Section 6.4 requirements
"""

import pandas as pd
import data_manager as dm

# ── Full rulebook as structured text ─────────────────────────────────────────
# Stored as a dict of {section_id: {title, text}} for targeted lookup,
# plus a single FULL_TEXT blob for Gemini's context window.

RULES: dict[str, dict] = {

    "0.1": {
        "title": "Game Setup",
        "keywords": ["setup", "settings", "quarter length", "difficulty", "simulation", "coach"],
        "text": """Section 0.1 – Game Setup:
Coach Type: Created or Current Coach only.
Mode/Difficulty: Simulation Mode – All-Madden.
Quarter Length: 8 minutes, 20-second accelerated clock runoff.""",
    },

    "0.2": {
        "title": "Streaming",
        "keywords": ["stream", "streaming", "youtube", "record", "link"],
        "text": """Section 0.2 – Streaming:
Both owners must stream and save each game.
Post the link before kickoff in league chat. Preferred: YouTube.""",
    },

    "1.1": {
        "title": "Play Variety",
        "keywords": ["spam", "play variety", "formation", "one-dimensional", "same play", "gun bunch", "repeat"],
        "text": """Section 1.1 – Play Variety:
You must mix up offensive formations and plays — no spamming the same look.
Use different route combos, motion looks, run/pass concepts.
BANNED: Calling the same play 2+ times in a row excessively (e.g., Gun Bunch HB Base 9 plays in a row).
Penalty: 1st = Warning | 2nd = Game suspension | 3rd = Loss of picks / removal.""",
    },

    "1.2": {
        "title": "Play Action Rules",
        "keywords": ["play action", "pa", "handoff animation", "fake", "fake run"],
        "text": """Section 1.2 – Play Action Rules:
Play Action is ONLY allowed when you've built a run threat.
BANNED: PA on 3rd & 10+.
BANNED: Canceling the handoff animation or hot routing the fake RB into a route.
RULE: Total PA attempts cannot exceed total rushing attempts.
Example ✅: 2nd & 5 with 8 rushes already → PA fine.
Example ❌: 3rd & 12 with 3 rushes all game → Illegal.
Penalty: Same as 1.1.""",
    },

    "1.3": {
        "title": "Run/Pass Balance",
        "keywords": ["run pass", "balance", "ratio", "rushing", "pass ratio", "jet sweep", "screen", "rpo", "bunch", "snugs"],
        "text": """Section 1.3 – Run/Pass Balance:
Keep at least a 70/30 run/pass ratio over time (season average).
Hard caps per game (+1 in OT):
  - Jet sweeps / pop passes: Max 3 per game
  - Screens: Max 3 per game
  - RPOs: 3 max passing, 4 max total
  - Bunch/Snugs formations: Max 50% of snaps
Example ✅: 18 passes / 12 runs → Balanced.
Example ❌: 35 passes / 3 runs every game → Illegal.""",
    },

    "1.4": {
        "title": "Clock Management",
        "keywords": ["clock", "milk", "chew", "hurry", "spike", "fake snap", "time"],
        "text": """Section 1.4 – Clock Management (Milking):
You CANNOT chew clock to under 10 seconds every snap unless:
  - Under 4 minutes left in the game
  - Up by 24+ in the 2nd half
  - Both players agree
Only 1 fake snap per play — no spam hiking.
Example ✅: Final 2 minutes, chew to 3 seconds → Legal.
Example ❌: 2nd quarter, up 7, chewing to 1 second every snap → Illegal.""",
    },

    "1.5": {
        "title": "QB Movement",
        "keywords": ["qb movement", "scramble", "rollout", "sprint", "pocket", "motion snap", "scrambling"],
        "text": """Section 1.5 – QB Movement:
Move your QB like a real QB — not like a punt returner.
BANNED: Instant sprints out of the pocket unless chased by a defender or blitz.
BANNED: Drifting 20+ yards sideways to wait for zones to break.
BANNED: Motion snapping with manual movement.
Players must fully set before snapping (unless built-in motion).
Example ✅: Roll right because DE is unblocked → Legal.
Example ❌: Snap and immediately sprint to sideline with no pressure → Illegal.""",
    },

    "1.6": {
        "title": "No-Huddle / Hurry-Up",
        "keywords": ["no huddle", "hurry up", "hurry-up", "no-huddle", "two minute", "tempo"],
        "text": """Section 1.6 – No-Huddle / Hurry-Up:
Max 2 uses per game UNLESS:
  - Final 3 minutes of a half (unlimited)
  - Down by 17+ in the 2nd half (unlimited)
  - Down 8+ in the 4th quarter (unlimited)
Example ✅: End of 1st half, 45 seconds left → Legal.
Example ❌: 1st quarter, 0-0, spamming no-huddle → Illegal.""",
    },

    "1.7": {
        "title": "Adjustments & Hot Routes",
        "keywords": ["hot route", "audible", "hot routing", "pre-snap", "playmaker", "adjustment"],
        "text": """Section 1.7 – Adjustments & Hot Routes:
BANNED: Excessive hot routes every down.
BANNED: Playmaker (hot routing a WR while scrambling).
BANNED: Spending 15+ seconds reassigning everyone every snap.
Example ✅: Audible a slant to a go because CB is pressed → Legal.
Example ❌: Snap after 8 hot routes and a QB rollout → Illegal.""",
    },

    "1.8": {
        "title": "4th Down Rules",
        "keywords": ["4th down", "fourth down", "go for it", "field position", "field goal", "punting", "icing"],
        "text": """Section 1.8 – 4th Down Rules:
ALWAYS ALLOWED: Overtime | Final 2 minutes of either half | Trailing by 14+ in 2nd half | Trailing in 4th quarter | Opponent agrees.

Standard Conditions (by field position):
  Own 0–49:     NOT allowed (except 4th & inches at own 45–49 in 2nd half when tied or losing)
  Opp 50–40:    4th & 1 or less
  Opp 39–30:    4th & 2 or less
  Opp 29–20:    4th & 3 or less OR FG attempt is 53+ yards
  Opp 19–1:     Any distance (FG range — always allowed)

Weather Adjustments (snow/heavy rain/high wind):
  High wind = kicking into 15+ MPH wind/crosswind.
  FGs ≥43 yards treated as "out of range" in high wind.
  4th & 2 or less allowed from own 45+ at any time in 2nd half if tied or losing.""",
    },

    "1.9": {
        "title": "Blowout Protocols",
        "keywords": ["blowout", "up big", "blow out", "bench", "mercy", "running up", "stat padding", "28", "35", "42",
                     "up 2", "up 3", "up 4", "winning by", "what do i have to", "what am i supposed to",
                     "sub out", "kneel", "run up", "scoring too much"],
        "text": """Section 1.9 – Blowout Protocols:
If up by 28+ in the 4th quarter: Only pass on 3rd downs.
If up by 35+ at any time: Bench QB1, WR1, HB1, and any player with 3+ TDs, 150+ yards, or 4+ sacks.
Keep benched players out until lead drops below 35.
Example ✅: Up 38-0 in 3rd, sub in backup HB → Legal.
Example ❌: Up 42-7 in 4th, still throwing bombs → Illegal.""",
    },

    "2.1": {
        "title": "Defensive Play Variety",
        "keywords": ["defense variety", "same defense", "defensive play", "cooldown", "blitz every play"],
        "text": """Section 2.1 – Defensive Play Variety:
Mix up defensive calls — don't run the same blitz or coverage repeatedly.
League enforces: 4-play cooldown on defensive plays, max 4 times per game per play.
Example ✅: Alternate Cover 3, Cover 2, Man Blitz → Legal.
Example ❌: Call the same Nickel Blitz every snap → Illegal.""",
    },

    "2.2": {
        "title": "Coverage Adjustments",
        "keywords": ["press", "overtop", "coverage", "cheese defense", "press coverage"],
        "text": """Section 2.2 – Coverage Adjustments:
BANNED: Press + Overtop at the same time — this is cheese.
Example ✅: Press coverage alone → Legal.
Example ❌: Press + Overtop on every snap → Illegal.""",
    },

    "2.3": {
        "title": "Pass Rush Requirement",
        "keywords": ["pass rush", "drop 8", "rush", "rusher", "3 man rush", "drop back"],
        "text": """Section 2.3 – Pass Rush Requirement:
Must rush at least 3 defenders every play.
LBs and DBs count toward the 3 if blitzing.
BANNED: Drop-8 coverages every snap.
Example ✅: Rush 2 DEs + Nickel CB blitz → Legal.
Example ❌: Rush 2 DL and spy everyone else → Illegal.""",
    },

    "2.4": {
        "title": "Player Movement (Defense)",
        "keywords": ["defensive movement", "pre-snap movement", "user linebacker", "walk safety", "bait"],
        "text": """Section 2.4 – Defensive Player Movement:
BANNED: Manually moving defenders before the snap.
BANNED: Moving your safety into the A-gap pre-snap.
After snap: move to cover zones, man assignments, or lanes. No zig-zagging to bait throws.""",
    },

    "2.5": {
        "title": "Defensive Position Subs",
        "keywords": ["db linebacker", "wr te", "sub lb", "position sub", "defense sub"],
        "text": """Section 2.5 – Defensive Position Subs:
DBs can only sub into LB if they meet size/speed requirements AND have commish approval.
No WRs at TE unless approved. No unrealistic SubLB conversions.
(Full requirements: see Section 6.4)""",
    },

    "2.6": {
        "title": "Run Commit",
        "keywords": ["run commit", "commit", "goal line", "short yardage"],
        "text": """Section 2.6 – Run Commit:
Use sparingly — primarily for 3rd/4th & short or goal line.
BANNED: Run commit with 3 or fewer rushers.
Example ✅: 4th & inches at the goal line → Legal.
Example ❌: 2nd & 7 from midfield → Illegal.""",
    },

    "3.1": {
        "title": "Kickoffs & Punts",
        "keywords": ["kickoff", "onside", "punt", "scum kick", "kick"],
        "text": """Section 3.1 – Kickoffs & Punts:
BANNED: Scum kicks that glitch blockers/returners.
Onside kicks ONLY allowed if trailing in the 4th quarter.""",
    },

    "3.2": {
        "title": "Kicking Rules",
        "keywords": ["icing", "timeout", "ice the kicker", "spike"],
        "text": """Section 3.2 – Kicking Rules:
If opponent calls a timeout to ice the kicker, you CANNOT cancel it by spiking or using your own timeout.""",
    },

    "3.3": {
        "title": "Fakes",
        "keywords": ["fake punt", "fake field goal", "fake fg", "trick play"],
        "text": """Section 3.3 – Fakes:
BANNED: Fake punts. Period.
BANNED: Fake field goals. Period.""",
    },

    "4.1": {
        "title": "Scheduling",
        "keywords": ["schedule", "scheduling", "contact", "24 hours", "respond"],
        "text": """Section 4.1 – Scheduling:
Contact your opponent early and lock in a time.
Respond to messages within 24 hours.""",
    },

    "4.2": {
        "title": "No-Show",
        "keywords": ["no show", "ghost", "can't play", "forfeit", "ghosting", "removal"],
        "text": """Section 4.2 – No-Show / Can't Play:
If you can't play, tell the commish ASAP.
Ghosting = removal risk.""",
    },

    "5.2": {
        "title": "Disconnect Protocol",
        "keywords": ["disconnect", "dropped", "crash", "restart", "connection", "force win"],
        "text": """Section 5.2 – Disconnect Protocol:
1st Quarter: Restart unless up 28+
2nd Quarter: Restart unless up 24+
3rd Quarter: Restart unless up 21+
4th Quarter: Restart unless up 14+
Final 4 mins: Restart unless up 14+ with 8-4 min left, or up 10+ with <4 min left.
Always allowed to replay if both agree.""",
    },

    "5.3": {
        "title": "Stat Padding & Blowouts",
        "keywords": ["stat pad", "stat padding", "run up", "losing big"],
        "text": """Section 5.3 – Stat Padding & Blowouts:
If winning big: Run more, call new plays, kneel out.
If losing big: No blitz spamming or bomb chucking every down.""",
    },

    "5.4": {
        "title": "Playing CPU",
        "keywords": ["cpu", "sim", "simulate", "cpu game"],
        "text": """Section 5.4 – Playing CPU:
All CPU games = simmed. No exceptions.""",
    },

    "6.1": {
        "title": "Roster Size & Cap",
        "keywords": ["roster", "cap", "cap space", "51 players", "roster size", "week 1"],
        "text": """Section 6.1 – Roster Size & Cap:
Always carry 51+ players.
Be under the cap before Week 1 — no negative rollover allowed.""",
    },

    "6.2": {
        "title": "Trades",
        "keywords": ["trade", "trades", "trade limit", "lopsided", "trade window", "6 players", "3 trades"],
        "text": """Section 6.2 – Trades:
Trades must be realistic and not lopsided.
Max 3 trades per season and 6 total players moved.
No player traded more than once per season.
No trading until after 5 games unless approved by commish.
Submit all trades via MyMadden — verbal deals don't count.""",
    },

    "6.3": {
        "title": "Contracts",
        "keywords": ["contract", "contract length", "years", "max contract"],
        "text": """Section 6.3 – Contracts:
Max contract length = 7 years.""",
    },

    "6.4": {
        "title": "Position Changes",
        "keywords": ["position change", "convert", "switch position", "move to", "change position",
                     "rb fb", "wr te", "te wr", "fb te", "s lb", "s cb", "cb s", "cb lb", "wr hb",
                     "hb wr", "te wr slot", "edge lb", "edge dt", "db dl"],
        "text": """Section 6.4 – Position Change Rules:

GENERAL RULES:
- No Abilities Reset Abuse: Players with Superstar/X-Factor abilities CANNOT change positions.
- All swaps must be realistic in NFL terms.
- Formation subs = same rules as depth chart changes.
- Approval required: post in position-change channel or get commish approval first.

ALLOWED CHANGES & REQUIREMENTS:

RB → FB (Commissioner approval only):
  Height ≥ 5'10" | Weight ≥ 225 | Lead Block ≥ 65 | Impact Block ≥ 60 | Speed ≤ 89 | Agility ≤ 88 | Carrying ≥ 75

WR → TE:
  Height ≥ 6'2" | Weight ≥ 225 | Speed ≤ 90 | Strength ≥ 65 | Run Block ≥ 60 | Impact Block ≥ 60

TE → WR:
  Weight ≤ 240 | Speed ≤ 90 | Release ≥ 70 | Short Route Run ≥ 70

FB → TE:
  Height ≥ 6'1" | Weight ≥ 240 | Run Block ≥ 65 | Impact Block ≥ 65

S → LB (including SubLB):
  Height ≥ 5'11" | Weight ≥ 215 | Speed ≤ 92 | Agility & CoD ≤ 90 | Tackle ≥ 75 | Hit Power ≥ 75
  NOTE: No CBs may be moved to LB under any circumstances.

S → CB:
  Height ≥ 5'10" | Weight ≥ 190 | Speed ≥ 88 | Agility & CoD ≥ 85 | Man Coverage ≥ 70

CB → S:
  Height ≥ 5'10" | Weight ≥ 190 | Speed ≤ 93 | Agility & CoD ≤ 92 | Tackle ≥ 60 | Zone Coverage ≥ 70 | Pursuit ≥ 65

CB → LB: ❌ NOT ALLOWED UNDER ANY CIRCUMSTANCES.

WR → HB (Approval only):
  Carrying ≥ 75 | BCV ≥ 70 | Weight ≥ 210

HB/TE → WR Slot (Approval only):
  Catch ≥ 70 | Short Route Run ≥ 65

OL → TE/FB: ❌ NOT ALLOWED (except Madden "Extra OL" goal line package).

BANNED SWAPS (no exceptions):
- WR ↔ HB without commissioner approval
- EDGE ↔ LB in any form
- EDGE ↔ DT in any form
- DB ↔ DL in any form
- CB → LB in any form
- QB → any skill position
- Any swap intended to bypass thresholds

ENFORCEMENT:
- Illegal use = player nerf + possible draft pick forfeiture
- Repeat violations = removal from league""",
    },
}


FULL_RULEBOOK_TEXT = "\n\n".join(
    f"[Rule {sid}] {data['title']}:\n{data['text']}"
    for sid, data in RULES.items()
)


# ── Rule lookup ───────────────────────────────────────────────────────────────

RULE_TRIGGERS = [
    "rule", "legal", "illegal", "allowed", "banned", "can i", "is it ok",
    "am i allowed", "violation", "penalty", "against the rules", "rulebook",
    "is this legal", "can we", "what's the rule", "what are the rules",
    "how many", "limit", "max", "fake punt", "fake fg", "onside",
    "4th down", "fourth down", "no huddle", "play action", "hot route", "press overtop",
    "run commit", "blowout", "disconnect", "trade limit", "contract length",
    "position change", "convert", "cpu game", "stream", "streaming",
    "clock", "milk", "chew", "jet sweep", "screen limit", "rpo", "bunch",
    # Score/blowout numeric triggers
    "up 2", "up 3", "up 4",  # catches "up 28", "up 35", "up 42" etc.
    "winning by", "losing by", "down by", "trailing by",
    "bench", "sub out", "kneel", "blow out",
]


def is_rule_query(query: str) -> bool:
    q = query.lower()
    return any(t in q for t in RULE_TRIGGERS)


def lookup_rules(query: str) -> list[dict]:
    """Return list of relevant rule sections for a query."""
    q = query.lower()
    matched = []
    for sid, data in RULES.items():
        score = sum(1 for kw in data["keywords"] if kw in q)
        if score > 0:
            matched.append((score, sid, data))
    matched.sort(key=lambda x: x[0], reverse=True)
    return [{"section": sid, "title": d["title"], "text": d["text"]}
            for _, sid, d in matched[:3]]  # top 3 most relevant sections


def build_rule_context(query: str) -> str:
    """Build the rule context block for Gemini."""
    hits = lookup_rules(query)
    if not hits:
        return f"[TSL RULEBOOK — No specific rule matched, using full context]\n{FULL_RULEBOOK_TEXT[:3000]}"
    lines = ["[TSL RULEBOOK — RELEVANT SECTIONS]:"]
    for h in hits:
        lines.append(h["text"])
    return "\n\n".join(lines)


# ── Position change validator ─────────────────────────────────────────────────

# Requirements table — mirrors Section 6.4
POS_CHANGE_RULES = {
    ("RB", "FB"): {
        "commish_required": True,
        "checks": [
            ("height",             ">=", 70,  "Height ≥ 5'10\" (70 inches)"),
            ("weight",             ">=", 225, "Weight ≥ 225 lbs"),
            ("leadBlockRating",    ">=", 65,  "Lead Block ≥ 65"),
            ("impactBlockRating",  ">=", 60,  "Impact Block ≥ 60"),
            ("speedRating",        "<=", 89,  "Speed ≤ 89"),
            ("agilityRating",      "<=", 88,  "Agility ≤ 88"),
            ("carryRating",        ">=", 75,  "Carrying ≥ 75"),
        ],
    },
    ("WR", "TE"): {
        "commish_required": False,
        "checks": [
            ("height",             ">=", 74,  "Height ≥ 6'2\" (74 inches)"),
            ("weight",             ">=", 225, "Weight ≥ 225 lbs"),
            ("speedRating",        "<=", 90,  "Speed ≤ 90"),
            ("strengthRating",     ">=", 65,  "Strength ≥ 65"),
            ("runBlockRating",     ">=", 60,  "Run Block ≥ 60"),
            ("impactBlockRating",  ">=", 60,  "Impact Block ≥ 60"),
        ],
    },
    ("TE", "WR"): {
        "commish_required": False,
        "checks": [
            ("weight",                 "<=", 240, "Weight ≤ 240 lbs"),
            ("speedRating",            "<=", 90,  "Speed ≤ 90"),
            ("releaseRating",          ">=", 70,  "Release ≥ 70"),
            ("routeRunShortRating",    ">=", 70,  "Short Route Run ≥ 70"),
        ],
    },
    ("FB", "TE"): {
        "commish_required": False,
        "checks": [
            ("height",             ">=", 73,  "Height ≥ 6'1\" (73 inches)"),
            ("weight",             ">=", 240, "Weight ≥ 240 lbs"),
            ("runBlockRating",     ">=", 65,  "Run Block ≥ 65"),
            ("impactBlockRating",  ">=", 65,  "Impact Block ≥ 65"),
        ],
    },
    ("SS", "LB"): {
        "commish_required": True,
        "checks": [
            ("height",                    ">=", 71,  "Height ≥ 5'11\" (71 inches)"),
            ("weight",                    ">=", 215, "Weight ≥ 215 lbs"),
            ("speedRating",               "<=", 92,  "Speed ≤ 92"),
            ("agilityRating",             "<=", 90,  "Agility ≤ 90"),
            ("changeOfDirectionRating",   "<=", 90,  "CoD ≤ 90"),
            ("tackleRating",              ">=", 75,  "Tackle ≥ 75"),
            ("hitPowerRating",            ">=", 75,  "Hit Power ≥ 75"),
        ],
    },
    ("FS", "LB"): {
        "commish_required": True,
        "checks": [
            ("height",                    ">=", 71,  "Height ≥ 5'11\" (71 inches)"),
            ("weight",                    ">=", 215, "Weight ≥ 215 lbs"),
            ("speedRating",               "<=", 92,  "Speed ≤ 92"),
            ("agilityRating",             "<=", 90,  "Agility ≤ 90"),
            ("changeOfDirectionRating",   "<=", 90,  "CoD ≤ 90"),
            ("tackleRating",              ">=", 75,  "Tackle ≥ 75"),
            ("hitPowerRating",            ">=", 75,  "Hit Power ≥ 75"),
        ],
    },
    ("SS", "CB"): {
        "commish_required": False,
        "checks": [
            ("height",                    ">=", 70,  "Height ≥ 5'10\" (70 inches)"),
            ("weight",                    ">=", 190, "Weight ≥ 190 lbs"),
            ("speedRating",               ">=", 88,  "Speed ≥ 88"),
            ("agilityRating",             ">=", 85,  "Agility ≥ 85"),
            ("changeOfDirectionRating",   ">=", 85,  "CoD ≥ 85"),
            ("manCoverRating",            ">=", 70,  "Man Coverage ≥ 70"),
        ],
    },
    ("CB", "SS"): {
        "commish_required": False,
        "checks": [
            ("height",                    ">=", 70,  "Height ≥ 5'10\" (70 inches)"),
            ("weight",                    ">=", 190, "Weight ≥ 190 lbs"),
            ("speedRating",               "<=", 93,  "Speed ≤ 93"),
            ("agilityRating",             "<=", 92,  "Agility ≤ 92"),
            ("changeOfDirectionRating",   "<=", 92,  "CoD ≤ 92"),
            ("tackleRating",              ">=", 60,  "Tackle ≥ 60"),
            ("zoneCoverRating",           ">=", 70,  "Zone Coverage ≥ 70"),
            ("pursuitRating",             ">=", 65,  "Pursuit ≥ 65"),
        ],
    },
    ("CB", "FS"): {  # CB → S (free safety)
        "commish_required": False,
        "checks": [
            ("height",                    ">=", 70,  "Height ≥ 5'10\" (70 inches)"),
            ("weight",                    ">=", 190, "Weight ≥ 190 lbs"),
            ("speedRating",               "<=", 93,  "Speed ≤ 93"),
            ("agilityRating",             "<=", 92,  "Agility ≤ 92"),
            ("changeOfDirectionRating",   "<=", 92,  "CoD ≤ 92"),
            ("tackleRating",              ">=", 60,  "Tackle ≥ 60"),
            ("zoneCoverRating",           ">=", 70,  "Zone Coverage ≥ 70"),
            ("pursuitRating",             ">=", 65,  "Pursuit ≥ 65"),
        ],
    },
    ("WR", "HB"): {
        "commish_required": True,
        "checks": [
            ("carryRating",           ">=", 75,  "Carrying ≥ 75"),
            ("bCVRating",             ">=", 70,  "BCV ≥ 70"),
            ("weight",                ">=", 210, "Weight ≥ 210 lbs"),
        ],
    },
    ("HB", "WR"): {
        "commish_required": True,
        "checks": [
            ("catchRating",           ">=", 70,  "Catch ≥ 70"),
            ("routeRunShortRating",   ">=", 65,  "Short Route Run ≥ 65"),
        ],
    },
    ("TE", "WR_SLOT"): {
        "commish_required": True,
        "checks": [
            ("catchRating",           ">=", 70,  "Catch ≥ 70"),
            ("routeRunShortRating",   ">=", 65,  "Short Route Run ≥ 65"),
        ],
    },
}

# Permanently banned conversions
BANNED_CONVERSIONS = [
    ("CB",   "LB"),
    ("CB",   "MIKE"),
    ("CB",   "SAM"),
    ("CB",   "WILL"),
    ("LEDGE","LB"),
    ("REDGE","LB"),
    ("LEDGE","DT"),
    ("REDGE","DT"),
    ("CB",   "DL"),
    ("SS",   "DL"),
    ("FS",   "DL"),
    ("QB",   "WR"),
    ("QB",   "HB"),
    ("QB",   "TE"),
    ("OL",   "TE"),
    ("OL",   "FB"),
    ("LT",   "TE"),
    ("RT",   "TE"),
    ("LG",   "TE"),
    ("RG",   "TE"),
    ("C",    "TE"),
]


def _check_banned(from_pos: str, to_pos: str) -> bool:
    fp, tp = from_pos.upper(), to_pos.upper()
    return any(
        (fp == b[0] and tp == b[1]) or
        ("LB" in tp and fp == "CB")     # CB → any LB variant
        for b in BANNED_CONVERSIONS
    )


def _run_checks(player_row: dict, checks: list) -> tuple[list, list]:
    """Returns (passed_checks, failed_checks) as lists of strings."""
    passed, failed = [], []
    for attr, op, threshold, label in checks:
        val = player_row.get(attr)
        if val is None:
            failed.append(f"⚠️ {label} — data not found")
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            failed.append(f"⚠️ {label} — could not read value")
            continue

        if op == ">=" and val >= threshold:
            passed.append(f"✅ {label} (actual: {val:.0f})")
        elif op == "<=" and val <= threshold:
            passed.append(f"✅ {label} (actual: {val:.0f})")
        else:
            failed.append(f"❌ {label} (actual: {val:.0f})")
    return passed, failed


def validate_position_change(player_name: str, from_pos: str, to_pos: str) -> dict:
    """
    Live validation of a position change request against Section 6.4 rules.
    Looks up the player in df_players, checks all attribute requirements.
    Returns structured result dict.
    """
    fp = from_pos.upper().strip()
    tp = to_pos.upper().strip()

    # 1. Check blanket bans
    if _check_banned(fp, tp):
        return {
            "type":    "pos_change",
            "player":  player_name,
            "from":    fp,
            "to":      tp,
            "verdict": "BANNED",
            "reason":  f"{fp} → {tp} is PERMANENTLY BANNED under Section 6.4. No exceptions.",
            "passed":  [],
            "failed":  [],
            "commish": False,
        }

    # 2. Check Superstar/X-Factor restriction
    player_row = None
    if not dm.df_players.empty:
        mask = dm.df_players["fullName"].str.lower().str.contains(player_name.lower(), na=False)
        if not mask.any():
            # try last name
            last = player_name.split()[-1].lower()
            mask = dm.df_players["lastName"].str.lower() == last
        if mask.any():
            player_row = dm.df_players[mask].iloc[0].to_dict()

    if player_row:
        dev = player_row.get("dev", "Normal")
        if dev in ("Superstar X-Factor", "Superstar"):
            return {
                "type":    "pos_change",
                "player":  player_name,
                "from":    fp,
                "to":      tp,
                "verdict": "BANNED",
                "reason":  f"{player_row.get('firstName','')} {player_row.get('lastName','')} is a {dev} — ability reset abuse rule (6.4) bans position changes for SS/XF players.",
                "passed":  [],
                "failed":  [],
                "commish": False,
            }

    # 3. Look up requirements
    rule_key = (fp, tp)
    # Try normalized key variants (e.g. S → LB maps to both SS→LB and FS→LB)
    rule = POS_CHANGE_RULES.get(rule_key)
    if not rule:
        # Try swapping S for both SS and FS
        for variant in [(fp.replace("S", "SS"), tp), (fp.replace("S", "FS"), tp)]:
            rule = POS_CHANGE_RULES.get(variant)
            if rule:
                break

    if not rule:
        return {
            "type":    "pos_change",
            "player":  player_name,
            "from":    fp,
            "to":      tp,
            "verdict": "UNKNOWN",
            "reason":  f"No conversion rule exists for {fp} → {tp} in the TSL rulebook. It's likely not allowed — get commish approval.",
            "passed":  [],
            "failed":  [],
            "commish": True,
        }

    # 4. Run attribute checks
    if player_row:
        passed, failed = _run_checks(player_row, rule["checks"])
        verdict = "APPROVED" if not failed else "FAILED"
        pname = f"{player_row.get('firstName','')} {player_row.get('lastName','')}".strip() or player_name
    else:
        passed, failed = [], [f"⚠️ Player '{player_name}' not found in roster data — cannot validate attributes."]
        verdict = "PLAYER_NOT_FOUND"
        pname = player_name

    return {
        "type":         "pos_change",
        "player":       pname,
        "from":         fp,
        "to":           tp,
        "verdict":      verdict,
        "reason":       "",
        "passed":       passed,
        "failed":       failed,
        "commish":      rule.get("commish_required", False),
        "dev":          player_row.get("dev", "?") if player_row else "?",
        "team":         player_row.get("teamName", "?") if player_row else "?",
    }
