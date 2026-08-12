"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import (
    astock_special_reports_context,
    build_instrument_context,
    get_language_instruction,
    is_astock_instrument,
    market_scope_context,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)

# The schema alone cannot stop the model from putting price levels into the
# free-text reasoning field, so the prompt says it explicitly too.
_NO_LEVELS_INSTRUCTION = (
    "Explain the reasoning behind the direction. Do NOT state entry prices, "
    "stop-loss levels, target prices or position sizes for this security."
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        market_scope = market_scope_context(company_name)
        investment_plan = state["investment_plan"]

        policy_report = state.get("policy_report", "")
        hot_money_report = state.get("hot_money_report", "")
        lockup_report = state.get("lockup_report", "")
        special_reports = astock_special_reports_context(
            company_name,
            policy_report,
            hot_money_report,
            lockup_report,
        )

        if is_astock_instrument(company_name):
            specialization = "You are a trading agent specialising in A-share (China mainland) stocks."
            constraints = (
                "You must factor in A-stock trading constraints:\n"
                "- T+1 settlement: shares bought today cannot be sold until the next trading day\n"
                "- Daily price limits: main board +/-10%, STAR/ChiNext +/-20%, Beijing Stock "
                "Exchange +/-30%. ST/*ST does NOT narrow the band; main-board ST/*ST moved "
                "from +/-5% to +/-10% on 2026-07-06, and STAR/ChiNext ST/*ST have always been +/-20%\n"
                "- Newly listed stocks have no price limit for their first 5 trading days "
                "(Beijing Stock Exchange: first day only)\n"
                "- Minimum lot: 100 shares on main board and ChiNext (100-share multiples); "
                "STAR board 200 shares minimum (1-share increments); Beijing Stock Exchange "
                "100 shares minimum (1-share increments)\n"
                "- Trading hours (Beijing time): call auction 09:15-09:25, continuous "
                "09:30-11:30 / 13:00-14:57, closing auction 14:57-15:00, after-hours "
                "fixed-price session 15:05-15:30 (all A-shares since 2026-07-06)"
            )
            analyst_team = "market, sentiment, news, fundamentals, policy, capital flow, and lockup/reduction specialists"
            context_label = "Additional A-Stock Analyst Context"
        else:
            specialization = "You are a trading agent specialising in US-listed stocks."
            constraints = (
                "You must factor in US equity trading constraints and risk drivers:\n"
                "- No A-share daily limit-up/limit-down framework; use US volatility, circuit breaker/LULD, and gap-risk concepts instead\n"
                "- Regular session is 09:30-16:00 US Eastern time; pre-market and after-hours liquidity can be thin and spreads can widen\n"
                "- Settlement, liquidity, borrow/short-interest, options/implied volatility, and earnings-event gaps can materially change execution risk\n"
                "- Use SEC filings, earnings/guidance, analyst revisions, insider transactions, sector regulation, Fed/rates, and USD macro sensitivity\n"
                "- Ignore missing A-share policy, hot-money, dragon-tiger-board, northbound-flow, and lockup/reduction reports; they are not applicable"
            )
            analyst_team = "market, sentiment, news, and fundamentals specialists"
            context_label = "Market-Specific Analyst Context"

        messages = [
            {
                "role": "system",
                "content": (
                    f"{specialization} "
                    "Translate the Research Manager's investment plan into a structured "
                    "transaction view. "
                    f"{market_scope}\n"
                    f"{constraints}\n"
                    "Anchor your reasoning in the analysts' reports and the research plan. "
                    f"{_NO_LEVELS_INSTRUCTION} "
                    "（以上参数仅供技术研究参考，不构成投资建议）"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on a comprehensive analysis by a team of analysts (including "
                    f"{analyst_team}), here is an investment plan for {company_name}.\n\n"
                    f"{instrument_context}\n\n"
                    f"Proposed Investment Plan:\n{investment_plan}\n\n"
                    + (f"{context_label}:\n{special_reports}\n\n" if special_reports else "")
                    + "Leverage these insights to craft the transaction view."
                    + get_language_instruction()
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
