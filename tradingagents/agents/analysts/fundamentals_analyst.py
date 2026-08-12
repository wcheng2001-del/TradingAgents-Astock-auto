from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_industry_comparison,
    get_insider_transactions,
    get_language_instruction,
    get_profit_forecast,
    is_astock_instrument,
    market_scope_context,
)
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        market_scope = market_scope_context(company_name)

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        if is_astock_instrument(company_name):
            tools.extend([get_profit_forecast, get_industry_comparison])
            system_message = (
                "你是一位专注于 A 股市场的基本面分析师。你的任务是全面分析目标公司的基本面信息，为投资决策提供扎实的数据支撑。"
            "\n\n⚠️ A 股基本面分析要点："
            "\n- **财务准则**：A 股上市公司采用中国会计准则（CAS），在收入确认、资产减值等方面与 IFRS 存在差异，分析时需注意口径。"
            "\n- **估值参照系**：A 股整体 PE 中位数偏高（30-50x 为常态），不能照搬美股 15-25x 标准；应对标同行业 A 股公司横向比较。"
            "\n- **核心指标**：重点关注营收增长率、归母净利润、扣非净利润（剔除非经常性损益）、ROE、毛利率、经营性现金流与净利润的匹配度。"
            "\n- **财报披露节奏**：一季报（4月底前）、半年报（8月底前）、三季报（10月底前）、年报（次年4月底前）。分析时注意数据的时效性。"
            "\n- **特殊风险关注**：商誉减值（并购后遗症）、股权质押比例、大股东减持计划、关联交易规模。"
            "\n\n请使用以下工具获取数据："
            "\n- `get_fundamentals`：获取公司综合基本面信息（PE/PB/总市值/季报财务快照/一致预期EPS/前向PE/PEG等）"
            "\n- `get_profit_forecast`：获取机构一致预期EPS详情（覆盖机构数、EPS区间、前向PE、PEG、PE消化时间）"
            "\n- `get_balance_sheet`：资产负债表详细数据"
            "\n- `get_cashflow`：现金流量表详细数据"
            "\n- `get_income_statement`：利润表详细数据"
            "\n- `get_industry_comparison(ticker, curr_date)`：获取全行业横向对比（90个行业涨跌幅/成交额/净流入排名，用于估值对标和行业定位）"
            "\n\n撰写详尽的基本面研究报告，给出具体数据支撑的分析结论（仅供研究参考，不构成投资建议）。报告末尾附 Markdown 表格汇总关键财务指标和估值水平。"
            "\n\n📋 必采清单 — 以下数据点必须出现在报告中，无法获取时标注 [数据缺失: xxx]："
            "\n1. PE（TTM）、PB、总市值"
            "\n2. 营收同比增长率"
            "\n3. 归母净利润及同比增长率"
            "\n4. ROE"
            "\n5. 资产负债率"
            "\n6. 经营性现金流与净利润比值"
            "\n7. 机构一致预期 EPS（调用 get_profit_forecast 获取）"
            + get_language_instruction()
            )
        else:
            system_message = (
                "你是一位专注于美股市场的基本面分析师。你的任务是全面分析目标公司的基本面信息，为投资决策提供扎实的数据支撑。"
                f"\n\n{market_scope}"
                "\n\n⚠️ 美股基本面分析要点："
                "\n- **财务准则与披露**：关注 US GAAP/IFRS 报表、SEC 披露、管理层指引、非 GAAP 调整和一次性项目。"
                "\n- **估值参照系**：使用同行业美股公司、成长率、毛利率、自由现金流和资产负债表质量进行估值比较；不要套用 A 股 PE 消化框架。"
                "\n- **核心指标**：重点关注营收增长率、EPS/净利润、毛利率、营业利润率、自由现金流、净现金/债务、回购/股权激励稀释。"
                "\n- **事件节奏**：注意财报日期、管理层 guidance、分析师预期修正和行业需求周期。"
                "\n- **范围约束**：不要调用或依赖 A 股一致预期、行业横向资金流、解禁/减持、游资或北向资金框架。"
                "\n\n请使用以下工具获取数据："
                "\n- `get_fundamentals`：获取公司综合基本面信息"
                "\n- `get_balance_sheet`：资产负债表详细数据"
                "\n- `get_cashflow`：现金流量表详细数据"
                "\n- `get_income_statement`：利润表详细数据"
                "\n\n撰写详尽的基本面研究报告，给出具体数据支撑的分析结论（仅供研究参考，不构成投资建议）。报告末尾附 Markdown 表格汇总关键财务指标和估值水平。"
                "\n\n📋 必采清单 — 以下数据点必须出现在报告中，无法获取时标注 [数据缺失: xxx]："
                "\n1. PE（TTM）、PB 或其他可用估值指标"
                "\n2. 营收同比增长率"
                "\n3. EPS/净利润及同比增长率"
                "\n4. 毛利率和营业利润率"
                "\n5. 资产负债表质量（现金、债务、流动性）"
                "\n6. 自由现金流或经营性现金流质量"
                "\n7. 最近 guidance/分析师预期变化（如可获取）"
                + get_language_instruction()
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " Do not add transaction stop-signal labels to your saved report;"
                    " focus only on your assigned analyst section."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
