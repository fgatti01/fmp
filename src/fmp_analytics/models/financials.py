"""Financial statement data models."""

from datetime import date

from pydantic import Field

from fmp_analytics.models.base import FMPBaseModel


class IncomeStatement(FMPBaseModel):
    """Income statement data model."""

    date: date | None = Field(None, description="Statement date")
    symbol: str = Field(..., description="Stock symbol")
    reported_currency: str | None = Field(None, alias="reportedCurrency", description="Reported currency")
    cik: str | None = Field(None, description="CIK number")
    filling_date: date | None = Field(None, alias="fillingDate", description="Filing date")
    accepted_date: str | None = Field(None, alias="acceptedDate", description="Accepted date")
    calendar_year: str | None = Field(None, alias="calendarYear", description="Calendar year")
    period: str | None = Field(None, description="Period (FY, Q1, Q2, Q3, Q4)")

    # Revenue
    revenue: float | None = Field(None, description="Total revenue")
    cost_of_revenue: float | None = Field(None, alias="costOfRevenue", description="Cost of revenue")
    gross_profit: float | None = Field(None, alias="grossProfit", description="Gross profit")
    gross_profit_ratio: float | None = Field(None, alias="grossProfitRatio", description="Gross profit ratio")

    # Operating expenses
    research_and_development_expenses: float | None = Field(None, alias="researchAndDevelopmentExpenses")
    general_and_administrative_expenses: float | None = Field(None, alias="generalAndAdministrativeExpenses")
    selling_and_marketing_expenses: float | None = Field(None, alias="sellingAndMarketingExpenses")
    selling_general_and_administrative_expenses: float | None = Field(None, alias="sellingGeneralAndAdministrativeExpenses")
    other_expenses: float | None = Field(None, alias="otherExpenses")
    operating_expenses: float | None = Field(None, alias="operatingExpenses")
    cost_and_expenses: float | None = Field(None, alias="costAndExpenses")

    # Operating income
    operating_income: float | None = Field(None, alias="operatingIncome", description="Operating income")
    operating_income_ratio: float | None = Field(None, alias="operatingIncomeRatio")

    # Other income/expense
    interest_income: float | None = Field(None, alias="interestIncome")
    interest_expense: float | None = Field(None, alias="interestExpense")
    depreciation_and_amortization: float | None = Field(None, alias="depreciationAndAmortization")
    ebitda: float | None = Field(None, description="EBITDA")
    ebitda_ratio: float | None = Field(None, alias="ebitdaratio")

    # Income before tax
    income_before_tax: float | None = Field(None, alias="incomeBeforeTax")
    income_before_tax_ratio: float | None = Field(None, alias="incomeBeforeTaxRatio")
    income_tax_expense: float | None = Field(None, alias="incomeTaxExpense")

    # Net income
    net_income: float | None = Field(None, alias="netIncome", description="Net income")
    net_income_ratio: float | None = Field(None, alias="netIncomeRatio")

    # EPS
    eps: float | None = Field(None, description="Earnings per share")
    eps_diluted: float | None = Field(None, alias="epsdiluted", description="Diluted EPS")
    weighted_average_shs_out: float | None = Field(None, alias="weightedAverageShsOut")
    weighted_average_shs_out_dil: float | None = Field(None, alias="weightedAverageShsOutDil")

    link: str | None = Field(None, description="SEC filing link")
    final_link: str | None = Field(None, alias="finalLink")


class BalanceSheet(FMPBaseModel):
    """Balance sheet data model."""

    date: date | None = Field(None, description="Statement date")
    symbol: str = Field(..., description="Stock symbol")
    reported_currency: str | None = Field(None, alias="reportedCurrency")
    cik: str | None = Field(None)
    filling_date: date | None = Field(None, alias="fillingDate")
    accepted_date: str | None = Field(None, alias="acceptedDate")
    calendar_year: str | None = Field(None, alias="calendarYear")
    period: str | None = Field(None)

    # Current Assets
    cash_and_cash_equivalents: float | None = Field(None, alias="cashAndCashEquivalents")
    short_term_investments: float | None = Field(None, alias="shortTermInvestments")
    cash_and_short_term_investments: float | None = Field(None, alias="cashAndShortTermInvestments")
    net_receivables: float | None = Field(None, alias="netReceivables")
    inventory: float | None = Field(None)
    other_current_assets: float | None = Field(None, alias="otherCurrentAssets")
    total_current_assets: float | None = Field(None, alias="totalCurrentAssets")

    # Non-current Assets
    property_plant_equipment_net: float | None = Field(None, alias="propertyPlantEquipmentNet")
    goodwill: float | None = Field(None)
    intangible_assets: float | None = Field(None, alias="intangibleAssets")
    goodwill_and_intangible_assets: float | None = Field(None, alias="goodwillAndIntangibleAssets")
    long_term_investments: float | None = Field(None, alias="longTermInvestments")
    tax_assets: float | None = Field(None, alias="taxAssets")
    other_non_current_assets: float | None = Field(None, alias="otherNonCurrentAssets")
    total_non_current_assets: float | None = Field(None, alias="totalNonCurrentAssets")
    other_assets: float | None = Field(None, alias="otherAssets")
    total_assets: float | None = Field(None, alias="totalAssets")

    # Current Liabilities
    account_payables: float | None = Field(None, alias="accountPayables")
    short_term_debt: float | None = Field(None, alias="shortTermDebt")
    tax_payables: float | None = Field(None, alias="taxPayables")
    deferred_revenue: float | None = Field(None, alias="deferredRevenue")
    other_current_liabilities: float | None = Field(None, alias="otherCurrentLiabilities")
    total_current_liabilities: float | None = Field(None, alias="totalCurrentLiabilities")

    # Non-current Liabilities
    long_term_debt: float | None = Field(None, alias="longTermDebt")
    deferred_revenue_non_current: float | None = Field(None, alias="deferredRevenueNonCurrent")
    deferred_tax_liabilities_non_current: float | None = Field(None, alias="deferredTaxLiabilitiesNonCurrent")
    other_non_current_liabilities: float | None = Field(None, alias="otherNonCurrentLiabilities")
    total_non_current_liabilities: float | None = Field(None, alias="totalNonCurrentLiabilities")
    other_liabilities: float | None = Field(None, alias="otherLiabilities")
    capital_lease_obligations: float | None = Field(None, alias="capitalLeaseObligations")
    total_liabilities: float | None = Field(None, alias="totalLiabilities")

    # Equity
    preferred_stock: float | None = Field(None, alias="preferredStock")
    common_stock: float | None = Field(None, alias="commonStock")
    retained_earnings: float | None = Field(None, alias="retainedEarnings")
    accumulated_other_comprehensive_income_loss: float | None = Field(None, alias="accumulatedOtherComprehensiveIncomeLoss")
    other_total_stockholders_equity: float | None = Field(None, alias="othertotalStockholdersEquity")
    total_stockholders_equity: float | None = Field(None, alias="totalStockholdersEquity")
    total_equity: float | None = Field(None, alias="totalEquity")
    total_liabilities_and_stockholders_equity: float | None = Field(None, alias="totalLiabilitiesAndStockholdersEquity")
    minority_interest: float | None = Field(None, alias="minorityInterest")
    total_liabilities_and_total_equity: float | None = Field(None, alias="totalLiabilitiesAndTotalEquity")

    # Debt
    total_investments: float | None = Field(None, alias="totalInvestments")
    total_debt: float | None = Field(None, alias="totalDebt")
    net_debt: float | None = Field(None, alias="netDebt")

    link: str | None = Field(None)
    final_link: str | None = Field(None, alias="finalLink")


class CashFlowStatement(FMPBaseModel):
    """Cash flow statement data model."""

    date: date | None = Field(None)
    symbol: str = Field(...)
    reported_currency: str | None = Field(None, alias="reportedCurrency")
    cik: str | None = Field(None)
    filling_date: date | None = Field(None, alias="fillingDate")
    accepted_date: str | None = Field(None, alias="acceptedDate")
    calendar_year: str | None = Field(None, alias="calendarYear")
    period: str | None = Field(None)

    # Operating Activities
    net_income: float | None = Field(None, alias="netIncome")
    depreciation_and_amortization: float | None = Field(None, alias="depreciationAndAmortization")
    deferred_income_tax: float | None = Field(None, alias="deferredIncomeTax")
    stock_based_compensation: float | None = Field(None, alias="stockBasedCompensation")
    change_in_working_capital: float | None = Field(None, alias="changeInWorkingCapital")
    accounts_receivables: float | None = Field(None, alias="accountsReceivables")
    inventory: float | None = Field(None)
    accounts_payables: float | None = Field(None, alias="accountsPayables")
    other_working_capital: float | None = Field(None, alias="otherWorkingCapital")
    other_non_cash_items: float | None = Field(None, alias="otherNonCashItems")
    net_cash_provided_by_operating_activities: float | None = Field(None, alias="netCashProvidedByOperatingActivities")

    # Investing Activities
    investments_in_property_plant_and_equipment: float | None = Field(None, alias="investmentsInPropertyPlantAndEquipment")
    acquisitions_net: float | None = Field(None, alias="acquisitionsNet")
    purchases_of_investments: float | None = Field(None, alias="purchasesOfInvestments")
    sales_maturities_of_investments: float | None = Field(None, alias="salesMaturitiesOfInvestments")
    other_investing_activities: float | None = Field(None, alias="otherInvestingActivites")
    net_cash_used_for_investing_activities: float | None = Field(None, alias="netCashUsedForInvestingActivites")

    # Financing Activities
    debt_repayment: float | None = Field(None, alias="debtRepayment")
    common_stock_issued: float | None = Field(None, alias="commonStockIssued")
    common_stock_repurchased: float | None = Field(None, alias="commonStockRepurchased")
    dividends_paid: float | None = Field(None, alias="dividendsPaid")
    other_financing_activities: float | None = Field(None, alias="otherFinancingActivites")
    net_cash_used_provided_by_financing_activities: float | None = Field(None, alias="netCashUsedProvidedByFinancingActivities")

    # Net Change
    effect_of_forex_changes_on_cash: float | None = Field(None, alias="effectOfForexChangesOnCash")
    net_change_in_cash: float | None = Field(None, alias="netChangeInCash")
    cash_at_end_of_period: float | None = Field(None, alias="cashAtEndOfPeriod")
    cash_at_beginning_of_period: float | None = Field(None, alias="cashAtBeginningOfPeriod")
    operating_cash_flow: float | None = Field(None, alias="operatingCashFlow")
    capital_expenditure: float | None = Field(None, alias="capitalExpenditure")
    free_cash_flow: float | None = Field(None, alias="freeCashFlow")

    link: str | None = Field(None)
    final_link: str | None = Field(None, alias="finalLink")


class FinancialRatios(FMPBaseModel):
    """Financial ratios data model."""

    symbol: str = Field(...)
    date: date | None = Field(None)
    calendar_year: str | None = Field(None, alias="calendarYear")
    period: str | None = Field(None)

    # Profitability Ratios
    gross_profit_margin: float | None = Field(None, alias="grossProfitMargin")
    operating_profit_margin: float | None = Field(None, alias="operatingProfitMargin")
    pretax_profit_margin: float | None = Field(None, alias="pretaxProfitMargin")
    net_profit_margin: float | None = Field(None, alias="netProfitMargin")
    effective_tax_rate: float | None = Field(None, alias="effectiveTaxRate")
    return_on_assets: float | None = Field(None, alias="returnOnAssets")
    return_on_equity: float | None = Field(None, alias="returnOnEquity")
    return_on_capital_employed: float | None = Field(None, alias="returnOnCapitalEmployed")

    # Liquidity Ratios
    current_ratio: float | None = Field(None, alias="currentRatio")
    quick_ratio: float | None = Field(None, alias="quickRatio")
    cash_ratio: float | None = Field(None, alias="cashRatio")

    # Leverage Ratios
    debt_ratio: float | None = Field(None, alias="debtRatio")
    debt_equity_ratio: float | None = Field(None, alias="debtEquityRatio")
    long_term_debt_to_capitalization: float | None = Field(None, alias="longTermDebtToCapitalization")
    total_debt_to_capitalization: float | None = Field(None, alias="totalDebtToCapitalization")
    interest_coverage: float | None = Field(None, alias="interestCoverage")
    cash_flow_to_debt_ratio: float | None = Field(None, alias="cashFlowToDebtRatio")

    # Efficiency Ratios
    days_of_sales_outstanding: float | None = Field(None, alias="daysOfSalesOutstanding")
    days_of_inventory_outstanding: float | None = Field(None, alias="daysOfInventoryOutstanding")
    operating_cycle: float | None = Field(None, alias="operatingCycle")
    days_of_payables_outstanding: float | None = Field(None, alias="daysOfPayablesOutstanding")
    cash_conversion_cycle: float | None = Field(None, alias="cashConversionCycle")
    receivables_turnover: float | None = Field(None, alias="receivablesTurnover")
    payables_turnover: float | None = Field(None, alias="payablesTurnover")
    inventory_turnover: float | None = Field(None, alias="inventoryTurnover")
    fixed_asset_turnover: float | None = Field(None, alias="fixedAssetTurnover")
    asset_turnover: float | None = Field(None, alias="assetTurnover")

    # Valuation Ratios
    price_earnings_ratio: float | None = Field(None, alias="priceEarningsRatio")
    price_to_book_ratio: float | None = Field(None, alias="priceToBookRatio")
    price_to_sales_ratio: float | None = Field(None, alias="priceToSalesRatio")
    price_earnings_to_growth_ratio: float | None = Field(None, alias="priceEarningsToGrowthRatio")
    price_to_free_cash_flows_ratio: float | None = Field(None, alias="priceToFreeCashFlowsRatio")
    price_to_operating_cash_flows_ratio: float | None = Field(None, alias="priceToOperatingCashFlowsRatio")
    enterprise_value_multiple: float | None = Field(None, alias="enterpriseValueMultiple")

    # Per Share Ratios
    dividend_yield: float | None = Field(None, alias="dividendYield")
    dividend_per_share: float | None = Field(None, alias="dividendPerShare")
    payout_ratio: float | None = Field(None, alias="payoutRatio")


class KeyMetrics(FMPBaseModel):
    """Key financial metrics data model."""

    symbol: str = Field(...)
    date: date | None = Field(None)
    calendar_year: str | None = Field(None, alias="calendarYear")
    period: str | None = Field(None)

    # Valuation
    revenue_per_share: float | None = Field(None, alias="revenuePerShare")
    net_income_per_share: float | None = Field(None, alias="netIncomePerShare")
    operating_cash_flow_per_share: float | None = Field(None, alias="operatingCashFlowPerShare")
    free_cash_flow_per_share: float | None = Field(None, alias="freeCashFlowPerShare")
    cash_per_share: float | None = Field(None, alias="cashPerShare")
    book_value_per_share: float | None = Field(None, alias="bookValuePerShare")
    tangible_book_value_per_share: float | None = Field(None, alias="tangibleBookValuePerShare")
    shareholders_equity_per_share: float | None = Field(None, alias="shareholdersEquityPerShare")
    interest_debt_per_share: float | None = Field(None, alias="interestDebtPerShare")

    # Market Data
    market_cap: float | None = Field(None, alias="marketCap")
    enterprise_value: float | None = Field(None, alias="enterpriseValue")
    pe_ratio: float | None = Field(None, alias="peRatio")
    price_to_sales_ratio: float | None = Field(None, alias="priceToSalesRatio")
    pocf_ratio: float | None = Field(None, alias="pocfratio")
    pfcf_ratio: float | None = Field(None, alias="pfcfRatio")
    pb_ratio: float | None = Field(None, alias="pbRatio")
    ptb_ratio: float | None = Field(None, alias="ptbRatio")
    ev_to_sales: float | None = Field(None, alias="evToSales")
    ev_to_ebitda: float | None = Field(None, alias="enterpriseValueOverEBITDA")
    ev_to_operating_cash_flow: float | None = Field(None, alias="evToOperatingCashFlow")
    ev_to_free_cash_flow: float | None = Field(None, alias="evToFreeCashFlow")
    earnings_yield: float | None = Field(None, alias="earningsYield")
    free_cash_flow_yield: float | None = Field(None, alias="freeCashFlowYield")

    # Debt
    debt_to_equity: float | None = Field(None, alias="debtToEquity")
    debt_to_assets: float | None = Field(None, alias="debtToAssets")
    net_debt_to_ebitda: float | None = Field(None, alias="netDebtToEBITDA")

    # Returns
    roe: float | None = Field(None, alias="roe")
    roic: float | None = Field(None, alias="roic")
    roa: float | None = Field(None, alias="returnOnTangibleAssets")

    # Other
    current_ratio: float | None = Field(None, alias="currentRatio")
    interest_coverage: float | None = Field(None, alias="interestCoverage")
    income_quality: float | None = Field(None, alias="incomeQuality")
    dividend_yield: float | None = Field(None, alias="dividendYield")
    payout_ratio: float | None = Field(None, alias="payoutRatio")
    sales_general_and_administrative_to_revenue: float | None = Field(None, alias="salesGeneralAndAdministrativeToRevenue")
    research_and_development_to_revenue: float | None = Field(None, alias="researchAndDdevelopementToRevenue")
    intangibles_to_total_assets: float | None = Field(None, alias="intangiblesToTotalAssets")
    capex_to_operating_cash_flow: float | None = Field(None, alias="capexToOperatingCashFlow")
    capex_to_revenue: float | None = Field(None, alias="capexToRevenue")
    capex_to_depreciation: float | None = Field(None, alias="capexToDepreciation")
    stock_based_compensation_to_revenue: float | None = Field(None, alias="stockBasedCompensationToRevenue")
    graham_number: float | None = Field(None, alias="grahamNumber")
    graham_net_net: float | None = Field(None, alias="grahamNetNet")
    working_capital: float | None = Field(None, alias="workingCapital")
    tangible_asset_value: float | None = Field(None, alias="tangibleAssetValue")
    net_current_asset_value: float | None = Field(None, alias="netCurrentAssetValue")
    invested_capital: float | None = Field(None, alias="investedCapital")
    average_receivables: float | None = Field(None, alias="averageReceivables")
    average_payables: float | None = Field(None, alias="averagePayables")
    average_inventory: float | None = Field(None, alias="averageInventory")
    days_sales_outstanding: float | None = Field(None, alias="daysSalesOutstanding")
    days_payables_outstanding: float | None = Field(None, alias="daysPayablesOutstanding")
    days_of_inventory_on_hand: float | None = Field(None, alias="daysOfInventoryOnHand")
