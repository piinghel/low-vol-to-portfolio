"""Configuration for the low-volatility factor research pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    data_root: Path
    price_glob: str = "data/prices.parquet"
    constituents_glob: str = "data/constituents.parquet"
    index_glob: str = "data/index.parquet"
    date_column: str = "date"
    asset_column: str = "asset_id_bb_global"
    adjusted_price_column: str = "px_last"
    unadjusted_price_column: str = "px_last_unadjusted"
    total_return_column: str = "total_return"
    index_price_column: str = "px_last"
    minimum_unadjusted_price: float = 5.0
    maximum_held_absolute_daily_return: float = 3.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "data_root", Path(self.data_root).expanduser().resolve()
        )
        if self.minimum_unadjusted_price < 0:
            raise ValueError("minimum_unadjusted_price must be non-negative")
        if self.maximum_held_absolute_daily_return <= 0:
            raise ValueError("maximum_held_absolute_daily_return must be positive")


@dataclass(frozen=True)
class SignalConfig:
    # Medium-horizon defaults: approximately 1, 3, and 6 trading months.
    windows: tuple[int, ...] = (21, 63, 126)
    annualization_factor: int = 252
    minimum_annualized_volatility: float = 0.05
    maximum_annualized_volatility: float = 2.0
    return_column: str = "daily_return"
    signal_column: str = "selection_volatility"
    sizing_volatility_column: str = "sizing_volatility"

    def __post_init__(self) -> None:
        if not self.windows or any(window < 2 for window in self.windows):
            raise ValueError("windows must contain values of at least 2")
        if tuple(sorted(set(self.windows))) != self.windows:
            raise ValueError("windows must be unique and sorted")
        if self.annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive")
        if (
            not 0
            < self.minimum_annualized_volatility
            < self.maximum_annualized_volatility
        ):
            raise ValueError("volatility bounds are invalid")


@dataclass(frozen=True)
class BucketConfig:
    number_of_buckets: int = 10
    low_volatility_bucket: int = 1
    high_volatility_bucket: int = 10
    bucket_column: str = "volatility_decile"

    def __post_init__(self) -> None:
        if self.number_of_buckets < 2:
            raise ValueError("number_of_buckets must be at least 2")
        if self.low_volatility_bucket != 1:
            raise ValueError("low_volatility_bucket must be 1")
        if self.high_volatility_bucket != self.number_of_buckets:
            raise ValueError("high_volatility_bucket must equal number_of_buckets")


@dataclass(frozen=True)
class SizingConfig:
    volatility_window: int = 60
    annualized_stock_volatility_target: float = 0.20
    maximum_absolute_stock_weight: float = 0.04
    maximum_leg_gross_exposure: float = 1.0

    def __post_init__(self) -> None:
        if self.volatility_window < 2:
            raise ValueError("volatility_window must be at least 2")
        if self.annualized_stock_volatility_target <= 0:
            raise ValueError("annualized_stock_volatility_target must be positive")
        if not 0 < self.maximum_absolute_stock_weight <= 1:
            raise ValueError("maximum_absolute_stock_weight must be in (0, 1]")
        if self.maximum_leg_gross_exposure <= 0:
            raise ValueError("maximum_leg_gross_exposure must be positive")


@dataclass(frozen=True)
class CostConfig:
    equity_cost_bps: float = 5.0

    def __post_init__(self) -> None:
        if self.equity_cost_bps < 0:
            raise ValueError("equity_cost_bps must be non-negative")


@dataclass(frozen=True)
class BetaConfig:
    lookback: int = 252
    minimum_observations: int = 126
    beta_clip: tuple[float, float] = (-4.0, 4.0)

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if not 2 <= self.minimum_observations <= self.lookback:
            raise ValueError("minimum_observations must be between 2 and lookback")
        if self.beta_clip[0] >= self.beta_clip[1]:
            raise ValueError("beta_clip lower bound must be below upper bound")


@dataclass(frozen=True)
class BacktestConfig:
    # ISO weekday: Monday=1. On holidays, use the first trading day of the week.
    rebalance_weekday: int = 1
    execution_delay_trading_days: int = 1
    annualization_factor: int = 252

    def __post_init__(self) -> None:
        if not 1 <= self.rebalance_weekday <= 5:
            raise ValueError("rebalance_weekday must be between 1 and 5")
        if self.execution_delay_trading_days < 0:
            raise ValueError("execution_delay_trading_days must be non-negative")
        if self.annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive")


@dataclass(frozen=True)
class ScenarioConfig:
    low_volatility_long: str = "low_vol_long"
    high_volatility_long: str = "high_vol_long"
    naive_equal_weight_long_short: str = "naive_equal_ls"
    volatility_scaled_long_short: str = "vol_scaled_ls"

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if any(not value.strip() for value in values):
            raise ValueError("scenario names must be non-empty")
        if len(set(values)) != len(values):
            raise ValueError("scenario names must be unique")


@dataclass(frozen=True)
class PlotConfig:
    decile_profile_color: str = "#B2BBC3"
    low_volatility_color: str = "#3C8C83"
    high_volatility_color: str = "#D66A60"
    naive_long_short_color: str = "#9AA6AF"
    volatility_scaled_color: str = "#345B7E"
    realized_beta_color: str = "#3E4E5C"
    ex_ante_beta_color: str = "#B8C0C7"
    reference_low_color: str = "#637D91"
    reference_high_color: str = "#AEB9C1"
    text_color: str = "#2E3A45"
    muted_text_color: str = "#66737E"
    grid_color: str = "#E2E7EA"
    background_color: str = "#FFFFFF"
    zero_line_color: str = "#87939C"


@dataclass(frozen=True)
class ResearchConfig:
    data: DataConfig
    signal: SignalConfig = field(default_factory=SignalConfig)
    buckets: BucketConfig = field(default_factory=BucketConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    beta: BetaConfig = field(default_factory=BetaConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    scenarios: ScenarioConfig = field(default_factory=ScenarioConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)

    def __post_init__(self) -> None:
        if self.signal.annualization_factor != self.backtest.annualization_factor:
            raise ValueError("signal and backtest annualization factors must match")
