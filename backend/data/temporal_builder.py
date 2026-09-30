"""Leakage-safe monthly history construction from Home Credit auxiliary tables.

All event times are relative to the application date. Only strictly pre-cutoff
events are aggregated, applicants remain the unit of splitting, and short
histories are left padded with an explicit mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

MAX_SEQUENCE_LENGTH = 36
APPLICANT_ID = "SK_ID_CURR"

TEMPORAL_FEATURES = [
    "installment_amount",
    "payment_ratio",
    "days_past_due",
    "installment_outstanding",
    "payment_regularity",
    "pos_outstanding",
    "pos_days_past_due",
    "card_balance",
    "card_utilization",
    "bureau_active_count",
    "bureau_overdue_count",
    "previous_application_count",
    "previous_refusal_ratio",
]

SOURCE_AVAILABILITY = {
    "installment_amount": "Scheduled installment known before application cutoff.",
    "payment_ratio": "Payment recorded before application cutoff divided by installment.",
    "days_past_due": "Entry date minus scheduled date for pre-cutoff payments.",
    "installment_outstanding": "Pre-cutoff scheduled amount minus observed payment.",
    "payment_regularity": "Share of pre-cutoff installments paid no later than scheduled.",
    "pos_outstanding": "Remaining POS installments reported in a pre-cutoff month.",
    "pos_days_past_due": "POS delinquency reported in a pre-cutoff month.",
    "card_balance": "Credit-card balance reported in a pre-cutoff month.",
    "card_utilization": "Pre-cutoff balance divided by reported credit limit.",
    "bureau_active_count": "Active bureau accounts observed in a pre-cutoff month.",
    "bureau_overdue_count": "Bureau delinquency statuses observed in a pre-cutoff month.",
    "previous_application_count": "Applications decided before the prediction cutoff.",
    "previous_refusal_ratio": "Refusal share among applications decided before cutoff.",
}


@dataclass(frozen=True)
class TemporalBuildResult:
    applicant_ids: np.ndarray
    sequences: np.ndarray
    padding_mask: np.ndarray
    month_indices: np.ndarray
    feature_names: tuple[str, ...] = tuple(TEMPORAL_FEATURES)


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required temporal source is missing: {path}")
    return pd.read_csv(path, usecols=lambda name: name in columns)


def _month_from_days(values: pd.Series) -> pd.Series:
    return np.floor(pd.to_numeric(values, errors="coerce") / 30.0).astype("Int64")


def _filter_ids(frame: pd.DataFrame, applicant_ids: set[int]) -> pd.DataFrame:
    return frame[frame[APPLICANT_ID].isin(applicant_ids)].copy()


def _installment_events(data_dir: Path, applicant_ids: set[int]) -> pd.DataFrame:
    frame = _read(
        data_dir / "installments_payments.csv",
        [APPLICANT_ID, "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"],
    )
    frame = _filter_ids(frame, applicant_ids)
    frame["month"] = _month_from_days(frame["DAYS_INSTALMENT"])
    installment = pd.to_numeric(frame["AMT_INSTALMENT"], errors="coerce")
    payment = pd.to_numeric(frame["AMT_PAYMENT"], errors="coerce").fillna(0.0)
    frame["installment_amount"] = installment
    frame["payment_ratio"] = np.divide(
        payment,
        installment,
        out=np.zeros(len(frame), dtype=float),
        where=installment.fillna(0).to_numpy() > 0,
    )
    frame["days_past_due"] = (
        pd.to_numeric(frame["DAYS_ENTRY_PAYMENT"], errors="coerce")
        - pd.to_numeric(frame["DAYS_INSTALMENT"], errors="coerce")
    ).clip(lower=0)
    frame["installment_outstanding"] = (installment.fillna(0) - payment).clip(lower=0)
    frame["payment_regularity"] = (frame["days_past_due"].fillna(0) <= 0).astype(float)
    return frame.groupby([APPLICANT_ID, "month"], as_index=False).agg(
        installment_amount=("installment_amount", "sum"),
        payment_ratio=("payment_ratio", "mean"),
        days_past_due=("days_past_due", "mean"),
        installment_outstanding=("installment_outstanding", "sum"),
        payment_regularity=("payment_regularity", "mean"),
    )


def _pos_events(data_dir: Path, applicant_ids: set[int]) -> pd.DataFrame:
    frame = _read(
        data_dir / "POS_CASH_balance.csv",
        [APPLICANT_ID, "MONTHS_BALANCE", "CNT_INSTALMENT_FUTURE", "SK_DPD"],
    )
    frame = _filter_ids(frame, applicant_ids).rename(columns={"MONTHS_BALANCE": "month"})
    return frame.groupby([APPLICANT_ID, "month"], as_index=False).agg(
        pos_outstanding=("CNT_INSTALMENT_FUTURE", "sum"),
        pos_days_past_due=("SK_DPD", "max"),
    )


def _card_events(data_dir: Path, applicant_ids: set[int]) -> pd.DataFrame:
    frame = _read(
        data_dir / "credit_card_balance.csv",
        [APPLICANT_ID, "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"],
    )
    frame = _filter_ids(frame, applicant_ids).rename(columns={"MONTHS_BALANCE": "month"})
    balance = pd.to_numeric(frame["AMT_BALANCE"], errors="coerce").fillna(0.0)
    limit = pd.to_numeric(frame["AMT_CREDIT_LIMIT_ACTUAL"], errors="coerce").fillna(0.0)
    frame["card_balance"] = balance
    frame["card_utilization"] = np.divide(
        balance,
        limit,
        out=np.zeros(len(frame), dtype=float),
        where=limit.to_numpy() > 0,
    )
    return frame.groupby([APPLICANT_ID, "month"], as_index=False).agg(
        card_balance=("card_balance", "sum"),
        card_utilization=("card_utilization", "mean"),
    )


def _bureau_events(data_dir: Path, applicant_ids: set[int]) -> pd.DataFrame:
    bureau = _read(data_dir / "bureau.csv", [APPLICANT_ID, "SK_ID_BUREAU"])
    bureau = _filter_ids(bureau, applicant_ids)
    balance = _read(
        data_dir / "bureau_balance.csv", ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"]
    )
    frame = balance.merge(bureau, on="SK_ID_BUREAU", how="inner").rename(
        columns={"MONTHS_BALANCE": "month"}
    )
    frame["bureau_active_count"] = frame["STATUS"].isin(["0", "1", "2", "3", "4", "5"]).astype(float)
    frame["bureau_overdue_count"] = frame["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(float)
    return frame.groupby([APPLICANT_ID, "month"], as_index=False).agg(
        bureau_active_count=("bureau_active_count", "sum"),
        bureau_overdue_count=("bureau_overdue_count", "sum"),
    )


def _previous_application_events(data_dir: Path, applicant_ids: set[int]) -> pd.DataFrame:
    frame = _read(
        data_dir / "previous_application.csv",
        [APPLICANT_ID, "DAYS_DECISION", "NAME_CONTRACT_STATUS"],
    )
    frame = _filter_ids(frame, applicant_ids)
    frame["month"] = _month_from_days(frame["DAYS_DECISION"])
    frame["previous_application_count"] = 1.0
    frame["previous_refusal_ratio"] = (
        frame["NAME_CONTRACT_STATUS"].astype(str).str.lower() == "refused"
    ).astype(float)
    return frame.groupby([APPLICANT_ID, "month"], as_index=False).agg(
        previous_application_count=("previous_application_count", "sum"),
        previous_refusal_ratio=("previous_refusal_ratio", "mean"),
    )


def build_monthly_histories(
    data_dir: str | Path,
    applicant_ids: Iterable[int],
    *,
    cutoff_months: Mapping[int, int] | None = None,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
) -> TemporalBuildResult:
    """Build left-padded pre-cutoff sequences for the requested applicants."""
    ids = np.asarray(list(dict.fromkeys(int(value) for value in applicant_ids)), dtype=np.int64)
    id_set = set(ids.tolist())
    if not id_set:
        empty = np.empty((0, max_sequence_length, len(TEMPORAL_FEATURES)), dtype=np.float32)
        return TemporalBuildResult(
            ids,
            empty,
            np.empty((0, max_sequence_length), dtype=bool),
            np.empty((0, max_sequence_length), dtype=np.int16),
        )

    root = Path(data_dir)
    sources = [
        _installment_events(root, id_set),
        _pos_events(root, id_set),
        _card_events(root, id_set),
        _bureau_events(root, id_set),
        _previous_application_events(root, id_set),
    ]
    monthly = sources[0]
    for source in sources[1:]:
        monthly = monthly.merge(source, on=[APPLICANT_ID, "month"], how="outer")
    monthly["month"] = pd.to_numeric(monthly["month"], errors="coerce")
    monthly = monthly.dropna(subset=["month"])
    for feature in TEMPORAL_FEATURES:
        if feature not in monthly:
            monthly[feature] = np.nan

    sequences = np.zeros((len(ids), max_sequence_length, len(TEMPORAL_FEATURES)), dtype=np.float32)
    padding_mask = np.zeros((len(ids), max_sequence_length), dtype=bool)
    month_indices = np.zeros((len(ids), max_sequence_length), dtype=np.int16)
    cutoff_map = cutoff_months or {}

    for row_index, applicant_id in enumerate(ids):
        cutoff = int(cutoff_map.get(int(applicant_id), 0))
        history = monthly[
            (monthly[APPLICANT_ID] == applicant_id) & (monthly["month"] < cutoff)
        ].sort_values("month")
        history = history.tail(max_sequence_length)
        length = len(history)
        if length == 0:
            continue
        start = max_sequence_length - length
        sequences[row_index, start:] = history[TEMPORAL_FEATURES].fillna(0.0).to_numpy(
            dtype=np.float32
        )
        padding_mask[row_index, start:] = True
        month_indices[row_index, start:] = history["month"].to_numpy(dtype=np.int16)
        if np.any(month_indices[row_index, start:] >= cutoff):
            raise AssertionError("Post-cutoff event entered a temporal sequence")

    return TemporalBuildResult(ids, sequences, padding_mask, month_indices)


def assert_disjoint_applicant_splits(splits: Mapping[str, Iterable[int]]) -> None:
    """Raise when any borrower appears in more than one dataset split."""
    seen: dict[int, str] = {}
    for split_name, values in splits.items():
        for applicant_id in values:
            applicant_id = int(applicant_id)
            if applicant_id in seen:
                raise ValueError(
                    f"Applicant {applicant_id} appears in both {seen[applicant_id]} and {split_name}"
                )
            seen[applicant_id] = split_name
