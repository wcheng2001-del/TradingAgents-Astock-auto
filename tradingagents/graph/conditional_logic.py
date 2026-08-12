# TradingAgents/graph/conditional_logic.py

from tradingagents.agents.utils.agent_states import AgentState


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def should_continue_market(self, state: AgentState):
        """Determine if market analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_market"
        return "Msg Clear Market"

    def should_continue_social(self, state: AgentState):
        """Determine if social media analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_social"
        return "Msg Clear Social"

    def should_continue_news(self, state: AgentState):
        """Determine if news analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_news"
        return "Msg Clear News"

    def should_continue_fundamentals(self, state: AgentState):
        """Determine if fundamentals analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_fundamentals"
        return "Msg Clear Fundamentals"

    def should_continue_policy(self, state: AgentState):
        """Determine if policy analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_policy"
        return "Msg Clear Policy"

    def should_continue_hot_money(self, state: AgentState):
        """Determine if hot money tracking should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_hot_money"
        return "Msg Clear Hot_money"

    def should_continue_lockup(self, state: AgentState):
        """Determine if lockup/reduction analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_lockup"
        return "Msg Clear Lockup"

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue.

        Route by debate turn count instead of English speaker text. Reports may
        be forced to Chinese, so current_response can start with a localized
        label rather than "Bull"/"Bear".
        """

        count = state["investment_debate_state"]["count"]
        if count >= 2 * self.max_debate_rounds:
            return "Research Manager"
        return "Bear Researcher" if count % 2 == 1 else "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue.

        Route by turn count instead of speaker text so Chinese report labels do
        not break the graph path map.
        """
        count = state["risk_debate_state"]["count"]
        if count >= 3 * self.max_risk_discuss_rounds:
            return "Portfolio Manager"
        remainder = count % 3
        if remainder == 1:
            return "Conservative Analyst"
        if remainder == 2:
            return "Neutral Analyst"
        return "Aggressive Analyst"
