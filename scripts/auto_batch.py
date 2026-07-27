from __future__ import annotations

import os
import smtplib
import time
import zipfile
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

from cli.main import save_report_to_disk


SMTP_CONFIGS = {
    "gmail.com": ("smtp.gmail.com", 465),
    "outlook.com": ("smtp-mail.outlook.com", 587),
    "hotmail.com": ("smtp-mail.outlook.com", 587),
    "live.com": ("smtp-mail.outlook.com", 587),
    "qq.com": ("smtp.qq.com", 465),
    "foxmail.com": ("smtp.qq.com", 465),
    "163.com": ("smtp.163.com", 465),
    "126.com": ("smtp.126.com", 465),
}


def env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_config() -> dict:
    provider = env("TRADINGAGENTS_LLM_PROVIDER", "deepseek")
    quick_model = env("TRADINGAGENTS_QUICK_THINK_LLM", "deepseek-v4-flash")
    deep_model = env("TRADINGAGENTS_DEEP_THINK_LLM", "deepseek-v4-pro")

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = provider
    config["quick_think_llm"] = quick_model
    config["deep_think_llm"] = deep_model
    config["output_language"] = env("TRADINGAGENTS_OUTPUT_LANGUAGE", "Chinese")
    config["max_debate_rounds"] = int(env("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "1"))
    config["max_risk_discuss_rounds"] = int(env("TRADINGAGENTS_MAX_RISK_ROUNDS", "1"))
    config["data_vendors"] = {
        "core_stock_apis": "a_stock",
        "technical_indicators": "a_stock",
        "fundamental_data": "a_stock",
        "news_data": "a_stock",
        "signal_data": "a_stock",
    }
    return config


def run_stock(ticker: str, trade_date: str, output_root: Path, config: dict) -> Path | None:
    print(f"Analyzing {ticker} for {trade_date}")
    started = time.time()
    ta = TradingAgentsGraph(debug=True, config=config)
    try:
        final_state, decision = ta.propagate(ticker, trade_date)
    except Exception as exc:
        print(f"ERROR {ticker}: {exc}")
        return None

    ticker_dir = output_root / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    report_path = save_report_to_disk(final_state, ticker, ticker_dir)
    summary_path = ticker_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"ticker={ticker}",
                f"trade_date={trade_date}",
                f"duration_seconds={round(time.time() - started)}",
                f"decision={decision}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved report for {ticker}: {report_path}")
    return report_path


def make_zip(output_root: Path) -> Path:
    zip_path = output_root.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in output_root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(output_root.parent))
    return zip_path


def smtp_server_for(sender: str) -> tuple[str, int]:
    domain = sender.rsplit("@", 1)[-1].lower()
    return SMTP_CONFIGS.get(domain, ("smtp.gmail.com", 465))


def send_email(subject: str, body: str, attachment: Path) -> None:
    sender = env("EMAIL_SENDER")
    password = env("EMAIL_PASSWORD")
    receivers = split_csv(env("EMAIL_RECEIVERS"))
    sender_name = env("EMAIL_SENDER_NAME", "TradingAgents Astock Auto")

    if not sender or not password or not receivers:
        print("Email not configured; skipping notification.")
        return

    message = EmailMessage()
    message["From"] = f"{sender_name} <{sender}>"
    message["To"] = ", ".join(receivers)
    message["Subject"] = subject
    message.set_content(body)
    message.add_attachment(
        attachment.read_bytes(),
        maintype="application",
        subtype="zip",
        filename=attachment.name,
    )

    host, port = smtp_server_for(sender)
    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=30) as server:
            server.login(sender, password)
            server.send_message(message)
    else:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(message)
    print(f"Email sent to {', '.join(receivers)}")


def main() -> int:
    stocks = split_csv(env("STOCK_LIST", "600519"))
    trade_date = env(
        "TRADINGAGENTS_TRADE_DATE",
        datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d"),
    )
    run_stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output_root = Path(env("TRADINGAGENTS_RESULTS_DIR", "reports/auto")) / run_stamp
    output_root.mkdir(parents=True, exist_ok=True)

    config = build_config()
    print(
        "LLM profile: "
        f"provider={config['llm_provider']}, "
        f"quick={config['quick_think_llm']}, "
        f"deep={config['deep_think_llm']}"
    )
    print(f"Stocks: {', '.join(stocks)}")

    reports = [run_stock(stock, trade_date, output_root, config) for stock in stocks]
    success_count = sum(1 for report in reports if report is not None)
    zip_path = make_zip(output_root)

    subject = f"TradingAgents Astock report {trade_date}: {success_count}/{len(stocks)}"
    body = "\n".join(
        [
            "TradingAgents Astock auto run finished.",
            f"Trade date: {trade_date}",
            f"Stocks: {', '.join(stocks)}",
            f"Successful reports: {success_count}/{len(stocks)}",
            f"LLM provider: {config['llm_provider']}",
            f"Quick model: {config['quick_think_llm']}",
            f"Deep model: {config['deep_think_llm']}",
        ]
    )
    send_email(subject, body, zip_path)

    return 0 if success_count == len(stocks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
