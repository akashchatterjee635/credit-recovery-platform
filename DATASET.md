# Dataset provenance

The production experiments use the Home Credit Default Risk dataset and its
auxiliary history tables. Download it from the official Kaggle competition page
after accepting Kaggle's competition rules and dataset license. Raw files are
intentionally excluded from Git.

Place these files in `data/`:

- `application_train.csv`
- `installments_payments.csv`
- `bureau.csv` and `bureau_balance.csv`
- `POS_CASH_balance.csv`
- `credit_card_balance.csv`
- `previous_application.csv`

Run `python -m scripts.dataset_manifest --data-dir data` to record SHA-256
checksums before training. Never substitute a file without updating the manifest.
Applicant splits are persisted under `data/splits/`; the same applicant must not
appear in more than one split. The tiny file under `tests/fixtures/` is synthetic,
is licensed with this repository, and is only for CI architecture validation.

Temporal features use only records whose relative month is strictly before the
application cutoff. The feature-level availability justifications are exposed by
`backend.data.temporal_builder.SOURCE_AVAILABILITY`.
