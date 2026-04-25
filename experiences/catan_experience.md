# Catan agent experience library

- Call get_state before placing initial settlements to analyze tile numbers, probabilities, and resource distribution.
- Place initial settlements adjacent to high-probability numbers (6, 8) and prioritize diverse resource types.
- Monitor resource production in the first 10 turns after setup; if you gain fewer than 5 total resources, initial settlement placement likely failed and recovery will be difficult.
- Build your third settlement before buying development cards; early expansion is critical for resource production.
- Don't pursue longest road (5+ roads) before your third settlement unless you have sheep and wheat for subsequent expansion.
- After achieving longest road, immediately prioritize settlements or cities over building additional roads.
- Use 4:1 maritime trades when stuck for 10+ turns with no building progress; passive waiting is worse than inefficient trades, but you need 4+ of a single resource to trade.
- Build roads to reach new settlement locations and access better resource tiles or 2:1 ports, not just for longest road points.
- If you lack a critical resource type for 5+ turns, prioritize expanding to tiles that produce it.
- When you receive zero new resources from dice rolls for 6+ consecutive turns, your settlements are on rarely-rolled numbers; immediately check all available trades and builds.
- When you cannot build or trade for 8+ turns (have fewer than 4 of any resource), actively use the robber to block opponent's highest-probability tiles and protect your own production.
- When moving the robber, target opponent tiles with high-probability numbers to disrupt their production.
- Always work toward a specific building goal (settlement, city, road to expansion); avoid passively rolling and ending turns without progress.
- Track which resources are needed for your next building goal and plan trades or builds accordingly.
- When 1 VP from winning, calculate exact resources needed (settlement: wood+brick+sheep+wheat; city: 3 wheat+2 ore; dev card: sheep+wheat+ore) and pursue the fastest path.
- If stuck at starting VP for 10+ turns, reassess settlement placement strategy and force expansion even via suboptimal trades.
