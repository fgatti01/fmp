"""Command-line interface for FMP Analytics."""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from fmp_analytics.api.client import FMPClient
from fmp_analytics.pipeline.data_pipeline import DataPipeline
from fmp_analytics.pipeline.analysis_pipeline import AnalysisPipeline
from fmp_analytics.agent.financial_agent import create_agent

app = typer.Typer(
    name="fmp",
    help="FMP Financial Analytics CLI - CFA/FRM/CQF metrics and AI-powered analysis",
)
console = Console()


def _run_async(coro):
    """Run async function synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


@app.command()
def quote(
    symbols: str = typer.Argument(..., help="Comma-separated stock symbols"),
):
    """Get real-time stock quotes."""
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    async def fetch():
        async with DataPipeline() as pipeline:
            return await pipeline.get_quotes_batch(symbol_list)

    quotes = _run_async(fetch())

    table = Table(title="Stock Quotes")
    table.add_column("Symbol", style="cyan")
    table.add_column("Price", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("Change %", justify="right")
    table.add_column("Volume", justify="right")
    table.add_column("Market Cap", justify="right")

    for q in quotes:
        change_style = "green" if (q.change or 0) >= 0 else "red"
        table.add_row(
            q.symbol,
            f"${q.price:.2f}" if q.price else "N/A",
            f"[{change_style}]{q.change:+.2f}[/]" if q.change else "N/A",
            f"[{change_style}]{q.changes_percentage:+.2f}%[/]" if q.changes_percentage else "N/A",
            f"{q.volume:,}" if q.volume else "N/A",
            f"${q.market_cap:,.0f}" if q.market_cap else "N/A",
        )

    console.print(table)


@app.command()
def profile(
    symbol: str = typer.Argument(..., help="Stock symbol"),
):
    """Get company profile."""
    async def fetch():
        async with DataPipeline() as pipeline:
            return await pipeline.get_company_profile(symbol.upper())

    p = _run_async(fetch())

    console.print(Panel(
        f"""[bold]{p.company_name}[/bold] ({p.symbol})

[cyan]Sector:[/cyan] {p.sector}
[cyan]Industry:[/cyan] {p.industry}
[cyan]CEO:[/cyan] {p.ceo}
[cyan]Employees:[/cyan] {p.full_time_employees:,}
[cyan]Website:[/cyan] {p.website}

[cyan]Location:[/cyan] {p.city}, {p.state}, {p.country}

[cyan]Price:[/cyan] ${p.price:.2f}
[cyan]Market Cap:[/cyan] ${p.mkt_cap:,.0f}
[cyan]Beta:[/cyan] {p.beta:.2f}
[cyan]DCF Value:[/cyan] ${p.dcf:.2f}

[dim]{p.description[:300]}...[/dim]
""",
        title="Company Profile",
    ))


@app.command()
def analyze(
    symbol: str = typer.Argument(..., help="Stock symbol"),
):
    """Analyze a stock with CFA/FRM/CQF metrics."""
    async def fetch():
        async with DataPipeline() as pipeline:
            analysis = AnalysisPipeline(pipeline)
            return await analysis.analyze_stock(symbol.upper())

    r = _run_async(fetch())

    score_color = "green" if r.overall_score >= 70 else "yellow" if r.overall_score >= 50 else "red"

    console.print(Panel(
        f"""[bold]{r.company_name}[/bold] ({r.symbol})
[dim]{r.sector} | {r.industry}[/dim]

[bold]Valuation[/bold]
  Current Price: ${r.current_price:.2f}
  DCF Value: ${r.dcf_value:.2f}
  Upside: [{score_color}]{r.upside_potential:+.1f}%[/]
  P/E: {r.pe_ratio:.2f} | P/B: {r.pb_ratio:.2f} | EV/EBITDA: {r.ev_ebitda:.2f}

[bold]Profitability[/bold]
  Profit Margin: {r.profit_margin*100:.1f}%
  ROE: {r.roe*100:.1f}% | ROIC: {r.roic*100:.1f}%

[bold]Risk[/bold]
  Beta: {r.beta:.2f}
  Volatility: {r.volatility*100:.1f}%
  95% VaR: {r.var_95*100:.2f}%

[bold]Technicals[/bold]
  50-Day SMA: ${r.sma_50:.2f} ({r.price_vs_sma_50:+.1f}%)
  200-Day SMA: ${r.sma_200:.2f} ({r.price_vs_sma_200:+.1f}%)

[bold][{score_color}]Overall Score: {r.overall_score:.0f}/100[/][/bold]
""",
        title="Stock Analysis",
    ))


@app.command()
def optimize(
    symbols: str = typer.Argument(..., help="Comma-separated stock symbols"),
    risk_free_rate: float = typer.Option(0.05, help="Risk-free rate"),
):
    """Optimize portfolio weights (mean-variance optimization)."""
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    async def fetch():
        async with DataPipeline() as pipeline:
            analysis = AnalysisPipeline(pipeline)
            return await analysis.optimize_portfolio(
                symbol_list, risk_free_rate=risk_free_rate
            )

    result = _run_async(fetch())

    table = Table(title="Optimal Portfolio Weights")
    table.add_column("Symbol", style="cyan")
    table.add_column("Weight", justify="right")

    for symbol, weight in sorted(
        result["optimal_weights"].items(),
        key=lambda x: -x[1],
    ):
        table.add_row(symbol, f"{weight*100:.1f}%")

    console.print(table)
    console.print(f"\n[bold]Expected Return:[/bold] {result['expected_return']*100:.2f}%")
    console.print(f"[bold]Volatility:[/bold] {result['volatility']*100:.2f}%")
    console.print(f"[bold]Sharpe Ratio:[/bold] {result['sharpe_ratio']:.2f}")


@app.command()
def movers():
    """Show today's market movers."""
    async def fetch():
        async with DataPipeline() as pipeline:
            return await pipeline.get_market_movers()

    movers = _run_async(fetch())

    # Gainers
    table = Table(title="Top Gainers")
    table.add_column("Symbol", style="cyan")
    table.add_column("Price", justify="right")
    table.add_column("Change %", justify="right", style="green")

    for _, row in movers["gainers"].head(5).iterrows():
        table.add_row(
            row["symbol"],
            f"${row['price']:.2f}",
            f"+{row['changesPercentage']:.2f}%",
        )

    console.print(table)

    # Losers
    table = Table(title="Top Losers")
    table.add_column("Symbol", style="cyan")
    table.add_column("Price", justify="right")
    table.add_column("Change %", justify="right", style="red")

    for _, row in movers["losers"].head(5).iterrows():
        table.add_row(
            row["symbol"],
            f"${row['price']:.2f}",
            f"{row['changesPercentage']:.2f}%",
        )

    console.print(table)


@app.command()
def sectors():
    """Show sector performance."""
    async def fetch():
        async with DataPipeline() as pipeline:
            return await pipeline.get_sector_performance()

    sectors = _run_async(fetch())

    table = Table(title="Sector Performance")
    table.add_column("Sector", style="cyan")
    table.add_column("Change %", justify="right")

    for _, row in sectors.iterrows():
        change = float(row["changesPercentage"].replace("%", ""))
        style = "green" if change >= 0 else "red"
        table.add_row(row["sector"], f"[{style}]{change:+.2f}%[/]")

    console.print(table)


@app.command()
def chat():
    """Start interactive AI financial analyst chat."""
    console.print(Panel(
        """[bold]FMP Financial Analyst[/bold]

I'm your AI-powered financial analyst with expertise in:
- CFA: Portfolio management, valuation, fixed income
- FRM: Risk metrics, VaR, stress testing
- CQF: Derivatives, options pricing, quant methods

Type your questions or 'exit' to quit.
""",
        title="Welcome",
    ))

    try:
        agent = create_agent(debug=False)

        while True:
            try:
                user_input = console.input("\n[bold cyan]You:[/bold cyan] ")

                if user_input.lower() in ["exit", "quit", "q"]:
                    console.print("[dim]Goodbye![/dim]")
                    break

                if not user_input.strip():
                    continue

                console.print("\n[bold green]Analyst:[/bold green]")
                response = agent.chat(user_input)
                console.print(response)

            except KeyboardInterrupt:
                console.print("\n[dim]Goodbye![/dim]")
                break

    except Exception as e:
        console.print(f"[red]Error initializing agent: {e}[/red]")
        console.print("[dim]Make sure OPENAI_API_KEY is set in your .env file[/dim]")


@app.command()
def dcf(
    symbol: str = typer.Argument(..., help="Stock symbol"),
    growth: float = typer.Option(0.05, help="FCF growth rate"),
    terminal: float = typer.Option(0.02, help="Terminal growth rate"),
    discount: float = typer.Option(0.10, help="Discount rate (WACC)"),
):
    """Perform DCF valuation."""
    async def fetch():
        async with DataPipeline() as pipeline:
            analysis = AnalysisPipeline(pipeline)
            return await analysis.dcf_analysis(
                symbol.upper(), growth, terminal, discount
            )

    result = _run_async(fetch())

    upside = result["upside_potential"]
    style = "green" if upside > 0 else "red"

    console.print(Panel(
        f"""[bold]DCF Valuation: {result['symbol']}[/bold]

[cyan]Current Price:[/cyan] ${result['current_price']:.2f}
[cyan]Intrinsic Value:[/cyan] ${result['intrinsic_value']:.2f}
[{style}]Upside Potential: {upside:+.1f}%[/]

[bold]Assumptions:[/bold]
  Latest FCF: ${result['assumptions']['latest_fcf']:,.0f}
  Growth Rate: {result['assumptions']['growth_rate']*100:.1f}%
  Terminal Growth: {result['assumptions']['terminal_growth']*100:.1f}%
  Discount Rate: {result['assumptions']['discount_rate']*100:.1f}%
""",
        title="DCF Analysis",
    ))


@app.command()
def version():
    """Show version information."""
    from fmp_analytics import __version__

    console.print(f"FMP Financial Analytics v{__version__}")
    console.print("CFA/FRM/CQF metrics with AI-powered analysis")


if __name__ == "__main__":
    app()
