"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


_RATING_ZH = {
    PortfolioRating.BUY: "买入",
    PortfolioRating.OVERWEIGHT: "增持",
    PortfolioRating.HOLD: "持有",
    PortfolioRating.UNDERWEIGHT: "低配",
    PortfolioRating.SELL: "卖出",
}

_ACTION_ZH = {
    TraderAction.BUY: "买入",
    TraderAction.HOLD: "持有",
    TraderAction.SELL: "卖出",
}


def _is_chinese_output() -> bool:
    try:
        from tradingagents.dataflows.config import get_config

        lang = get_config().get("output_language", "English").strip().lower()
    except Exception:
        lang = "english"
    return lang in {"chinese", "zh", "zh-cn", "中文", "简体中文"}


def _rating_label(rating: PortfolioRating) -> str:
    return _RATING_ZH.get(rating, rating.value) if _is_chinese_output() else rating.value


def _action_label(action: TraderAction) -> str:
    return _ACTION_ZH.get(action, action.value) if _is_chinese_output() else action.value


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    if _is_chinese_output():
        return "\n".join([
            f"**投资建议**: {_rating_label(plan.recommendation)}",
            "",
            f"**核心理由**: {plan.rationale}",
            "",
            f"**策略行动**: {plan.strategic_actions}",
        ])
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then states a direction and the reasoning behind it.

    It deliberately carries **no executable price levels** — no entry price,
    no stop-loss, no position size. This project is a research and education
    implementation of the upstream TradingAgents framework, and concrete trade
    levels for a named security are what turn a research tool into an
    investment-advisory product. The capability is not shipped here; a
    downstream fork that wants it can add it under its own responsibility.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences. Do not quote specific "
            "entry, stop-loss or position-size levels."
        ),
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    if _is_chinese_output():
        return "\n".join([
            f"**操作方向**: {_action_label(proposal.action)}",
            "",
            f"**交易理由**: {proposal.reasoning}",
            "",
            f"最终交易建议：**{_action_label(proposal.action)}**",
        ])
    return "\n".join([
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.

    Like :class:`TraderProposal`, this carries no price target and no other
    executable level — see that class for why.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise summary of what drove the rating and the main "
            "considerations on each side. Two to four sentences. Do not quote "
            "specific entry, stop-loss, position-size or target-price levels."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Optional analysis horizon, e.g. '3-6 months'.",
    )


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    if _is_chinese_output():
        parts = [
            f"**最终评级**: {_rating_label(decision.rating)}",
            "",
            f"**执行摘要**: {decision.executive_summary}",
            "",
            f"**投资论点**: {decision.investment_thesis}",
        ]
        if decision.time_horizon:
            parts.extend(["", f"**时间周期**: {decision.time_horizon}"])
        return "\n".join(parts)

    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(parts)
