"""Company-related data models."""

from datetime import date

from pydantic import Field

from fmp_analytics.models.base import FMPBaseModel


class CompanyProfile(FMPBaseModel):
    """Company profile data model."""

    symbol: str = Field(..., description="Stock symbol")
    price: float | None = Field(None, description="Current price")
    beta: float | None = Field(None, description="Beta coefficient")
    vol_avg: int | None = Field(None, alias="volAvg", description="Average volume")
    mkt_cap: int | None = Field(None, alias="mktCap", description="Market capitalization")
    last_div: float | None = Field(None, alias="lastDiv", description="Last dividend")
    range: str | None = Field(None, description="52-week range")
    changes: float | None = Field(None, description="Price change")
    company_name: str | None = Field(None, alias="companyName", description="Company name")
    currency: str | None = Field(None, description="Currency")
    cik: str | None = Field(None, description="CIK number")
    isin: str | None = Field(None, description="ISIN")
    cusip: str | None = Field(None, description="CUSIP")
    exchange: str | None = Field(None, description="Exchange")
    exchange_short_name: str | None = Field(None, alias="exchangeShortName", description="Exchange short name")
    industry: str | None = Field(None, description="Industry")
    website: str | None = Field(None, description="Website URL")
    description: str | None = Field(None, description="Company description")
    ceo: str | None = Field(None, description="CEO name")
    sector: str | None = Field(None, description="Sector")
    country: str | None = Field(None, description="Country")
    full_time_employees: int | None = Field(None, alias="fullTimeEmployees", description="Number of employees")
    phone: str | None = Field(None, description="Phone number")
    address: str | None = Field(None, description="Address")
    city: str | None = Field(None, description="City")
    state: str | None = Field(None, description="State")
    zip: str | None = Field(None, description="ZIP code")
    dcf_diff: float | None = Field(None, alias="dcfDiff", description="DCF difference")
    dcf: float | None = Field(None, description="DCF value")
    image: str | None = Field(None, description="Company logo URL")
    ipo_date: date | None = Field(None, alias="ipoDate", description="IPO date")
    default_image: bool | None = Field(None, alias="defaultImage", description="Using default image")
    is_etf: bool | None = Field(None, alias="isEtf", description="Is ETF")
    is_actively_trading: bool | None = Field(None, alias="isActivelyTrading", description="Is actively trading")
    is_adr: bool | None = Field(None, alias="isAdr", description="Is ADR")
    is_fund: bool | None = Field(None, alias="isFund", description="Is fund")


class KeyExecutive(FMPBaseModel):
    """Key executive data model."""

    title: str | None = Field(None, description="Executive title")
    name: str | None = Field(None, description="Executive name")
    pay: float | None = Field(None, description="Compensation")
    currency_pay: str | None = Field(None, alias="currencyPay", description="Currency of pay")
    gender: str | None = Field(None, description="Gender")
    year_born: int | None = Field(None, alias="yearBorn", description="Year born")
    title_since: date | None = Field(None, alias="titleSince", description="Title since date")


class StockPeer(FMPBaseModel):
    """Stock peer data model."""

    symbol: str = Field(..., description="Stock symbol")
    peers_list: list[str] = Field(default_factory=list, alias="peersList", description="List of peer symbols")


class AnalystRating(FMPBaseModel):
    """Analyst rating data model."""

    symbol: str = Field(..., description="Stock symbol")
    date: date | None = Field(None, description="Rating date")
    analyst_ratings_buy: int | None = Field(None, alias="analystRatingsbuy", description="Buy ratings count")
    analyst_ratings_hold: int | None = Field(None, alias="analystRatingsHold", description="Hold ratings count")
    analyst_ratings_sell: int | None = Field(None, alias="analystRatingsSell", description="Sell ratings count")
    analyst_ratings_strong_buy: int | None = Field(None, alias="analystRatingsStrongBuy", description="Strong buy count")
    analyst_ratings_strong_sell: int | None = Field(None, alias="analystRatingsStrongSell", description="Strong sell count")


class PriceTarget(FMPBaseModel):
    """Analyst price target data model."""

    symbol: str = Field(..., description="Stock symbol")
    published_date: date | None = Field(None, alias="publishedDate", description="Published date")
    news_url: str | None = Field(None, alias="newsURL", description="News URL")
    news_title: str | None = Field(None, alias="newsTitle", description="News title")
    analyst_name: str | None = Field(None, alias="analystName", description="Analyst name")
    price_target: float | None = Field(None, alias="priceTarget", description="Price target")
    adj_price_target: float | None = Field(None, alias="adjPriceTarget", description="Adjusted price target")
    price_when_posted: float | None = Field(None, alias="priceWhenPosted", description="Price when posted")
    news_publisher: str | None = Field(None, alias="newsPublisher", description="News publisher")
    analyst_company: str | None = Field(None, alias="analystCompany", description="Analyst company")


class CompanyRating(FMPBaseModel):
    """Company rating data model."""

    symbol: str = Field(..., description="Stock symbol")
    date: date | None = Field(None, description="Rating date")
    rating: str | None = Field(None, description="Overall rating")
    rating_score: int | None = Field(None, alias="ratingScore", description="Rating score")
    rating_recommendation: str | None = Field(None, alias="ratingRecommendation", description="Recommendation")
    rating_dcf_score: int | None = Field(None, alias="ratingDetailsDCFScore", description="DCF score")
    rating_dcf_recommendation: str | None = Field(None, alias="ratingDetailsDCFRecommendation", description="DCF recommendation")
    rating_roe_score: int | None = Field(None, alias="ratingDetailsROEScore", description="ROE score")
    rating_roe_recommendation: str | None = Field(None, alias="ratingDetailsROERecommendation", description="ROE recommendation")
    rating_roa_score: int | None = Field(None, alias="ratingDetailsROAScore", description="ROA score")
    rating_roa_recommendation: str | None = Field(None, alias="ratingDetailsROARecommendation", description="ROA recommendation")
    rating_de_score: int | None = Field(None, alias="ratingDetailsDEScore", description="D/E score")
    rating_de_recommendation: str | None = Field(None, alias="ratingDetailsDERecommendation", description="D/E recommendation")
    rating_pe_score: int | None = Field(None, alias="ratingDetailsPEScore", description="P/E score")
    rating_pe_recommendation: str | None = Field(None, alias="ratingDetailsPERecommendation", description="P/E recommendation")
    rating_pb_score: int | None = Field(None, alias="ratingDetailsPBScore", description="P/B score")
    rating_pb_recommendation: str | None = Field(None, alias="ratingDetailsPBRecommendation", description="P/B recommendation")
