# Real-LLM spot-check: the 3 fixed PARTIAL cases

Checks whether a REAL (unscripted) model's own tool calls and phrasing still get caught by this session's new deterministic checks (group_size_imbalance, unusual_baseline_window, conclusion_guard).

## hard_confound_01b: PARTIAL
- explanation: Failed: tool_selection (expected one of ['GENERAL_ANALYSIS', 'EDA'], got {'STATISTICS', 'DATA_PROFILING'})
- final_answer_text: "Important caveat: at least one issue here is serious enough that a confident conclusion is not justified from the available evidence alone ('format' has a very different mix across the compared 'region' groups (North, South) -- this comparison may reflect a difference in 'format' rather than a true 'region' effect (a possible confounding variable). This split is severe (a 80-point gap) -- the two variables are barely distinguishable in this data, so no confident conclusion about 'region' alone is possible here.). Treat the analysis below with that in mind.\n\nThe data show that stores in the North region have a higher average basket size than stores in the South (mean\u202f$143.0 vs $87.9, a difference of about $55). The difference is statistically significant (two‑sample t‑test statistic\u202f7.11, p\u202f≈\u202f0). However, the comparison is confounded by the store “format” variable, which differs markedly between the two regions, so the observed basket‑size gap may be driven by format rather than region itself. Consequently, we cannot reliably attribute the higher basket size to the North region alone."
- limitations: [('methodological', 'reduces_confidence', "'avg_basket' results disagree between t_test (7.1075) and confidence_interval (143.0145) -- treat with caution."), ('methodological', 'blocks_conclusion', "'format' has a very different mix across the compared 'region' groups (North, South) -- this comparison may reflect a di")]

## hard_ab_01b: PROVIDER_ERROR
- PROVIDER_ERROR: The AI is receiving a lot of requests right now. Please try again in a moment.

## hard_price_01b: PROVIDER_ERROR
- PROVIDER_ERROR: The AI is receiving a lot of requests right now. Please try again in a moment.
