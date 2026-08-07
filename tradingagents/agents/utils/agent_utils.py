from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)
from tradingagents.agents.utils.signal_data_tools import (
    get_profit_forecast,
    get_hot_stocks,
    get_northbound_flow,
    get_concept_blocks,
    get_fund_flow,
    get_dragon_tiger_board,
    get_lockup_expiry,
    get_industry_comparison,
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to every saved report/debate agent so exported reports stay in the
    requested language end to end.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    if lang.strip().lower() in {"chinese", "zh", "zh-cn", "中文", "简体中文"}:
        return (
            " 请用简体中文输出完整回复。不要使用英文段落、英文角色前缀、"
            "英文小标题或英文结论标签；专有名词、股票代码、指标缩写"
            "（如 PE、MACD、RSI、ETF）可以保留英文。"
        )
    return f" Write your entire response in {lang}."


def is_chinese_output() -> bool:
    """Return True when the configured report language is Chinese."""
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English").strip().lower()
    return lang in {"chinese", "zh", "zh-cn", "中文", "简体中文"}


REPORT_LABELS_ZH = {
    "Trading Analysis Report": "交易分析报告",
    "Generated": "生成时间",
    "Analyst Team Reports": "分析师团队报告",
    "Market Analyst": "市场分析师",
    "Social Analyst": "社交情绪分析师",
    "News Analyst": "新闻分析师",
    "Fundamentals Analyst": "基本面分析师",
    "Research Team Decision": "研究团队决策",
    "Bull Researcher": "多头研究员",
    "Bear Researcher": "空头研究员",
    "Research Manager": "研究经理",
    "Trading Team Plan": "交易团队计划",
    "Trader": "交易员",
    "Risk Management Team Decision": "风险管理团队决策",
    "Aggressive Analyst": "激进风险分析师",
    "Conservative Analyst": "保守风险分析师",
    "Neutral Analyst": "中立风险分析师",
    "Portfolio Manager Decision": "投资组合经理决策",
    "Portfolio Manager": "投资组合经理",
}


def report_label(label: str) -> str:
    """Localize report section labels for exported markdown."""
    if is_chinese_output():
        return REPORT_LABELS_ZH.get(label, label)
    return label


def speaker_prefix(label: str) -> str:
    """Return the speaker prefix used inside debate transcripts."""
    return report_label(label)


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`). "
        "When a tool argument is named `ticker`, pass only this ticker value; "
        "do not pass company names, sectors, concepts, or search keywords."
    )

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
