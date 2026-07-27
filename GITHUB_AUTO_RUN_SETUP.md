# GitHub Auto Run Setup

This copy is intended to run separately from the local main `TradingAgents-Astock`.

## Repository variables

Set these in GitHub:

`Settings -> Secrets and variables -> Actions -> Variables`

```text
STOCK_LIST=600519,300750,002594
TRADINGAGENTS_LLM_PROVIDER=deepseek
TRADINGAGENTS_DEEP_THINK_LLM=deepseek-v4-pro
TRADINGAGENTS_QUICK_THINK_LLM=deepseek-v4-flash
TRADINGAGENTS_OUTPUT_LANGUAGE=Chinese
EMAIL_SENDER=yourname@gmail.com
EMAIL_RECEIVERS=yourname@gmail.com
EMAIL_SENDER_NAME=TradingAgents Astock Auto
```

For Qwen instead:

```text
TRADINGAGENTS_LLM_PROVIDER=qwen-cn
TRADINGAGENTS_DEEP_THINK_LLM=qwen3.7-plus
TRADINGAGENTS_QUICK_THINK_LLM=qwen3.7-plus
```

## Repository secrets

Set these in:

`Settings -> Secrets and variables -> Actions -> Secrets`

```text
DEEPSEEK_API_KEY=sk-...
DASHSCOPE_CN_API_KEY=sk-...
EMAIL_PASSWORD=your Gmail app password without spaces
```

## Schedule

The workflow currently runs on Tuesday, Thursday, Saturday at Beijing 06:00.

GitHub Actions uses UTC, so the cron is:

```yaml
- cron: "0 22 * * 1,3,5"
```
