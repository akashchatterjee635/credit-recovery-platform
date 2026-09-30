# Deterministic applicant splits

Generate `train_ids.txt`, `cal_ids.txt`, and `test_ids.txt` from the static
training command. These files contain applicant IDs only; raw attributes and
labels must not be committed. The temporal builder verifies that the sets are
disjoint before producing tensors.
