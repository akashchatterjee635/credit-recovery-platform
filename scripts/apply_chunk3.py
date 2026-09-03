import os

main_path = "backend/main.py"
with open(main_path, "r") as f:
    main_content = f.read()

# Update train loading
main_content = main_content.replace(
    "train_path = 'data/application_train.csv'\n    if _os.path.exists(train_path):\n        _training_sample = pd.read_csv(train_path, nrows=5000)",
    "train_path = 'data/train_reference.csv'\n    if _os.path.exists(train_path):\n        _training_sample = pd.read_csv(train_path)"
)

# Update ApplicantData
applicant_model = """class ApplicantData(BaseModel):
    AMT_CREDIT: float
    AMT_INCOME_TOTAL: float
    AMT_ANNUITY: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    NAME_EDUCATION_TYPE: str
    BUREAU_TOTAL_DEBT: Optional[float] = None
    BUREAU_MAX_OVERDUE: Optional[float] = None
    BUREAU_ACTIVE_COUNT: Optional[float] = None
    INST_LATE_RATIO: Optional[float] = None
    INST_AVG_DAYS_LATE: Optional[float] = None
    PREV_REFUSED_RATIO: Optional[float] = None"""

import re
main_content = re.sub(
    r'class ApplicantData\(BaseModel\):.*?NAME_EDUCATION_TYPE: str', 
    applicant_model, main_content, flags=re.DOTALL
)

roadmap_model = applicant_model.replace("ApplicantData", "RoadmapRequest") + """
    journey_id: Optional[int] = None
    borrower_id: Optional[int] = None"""

main_content = re.sub(
    r'class RoadmapRequest\(BaseModel\):.*?borrower_id: Optional\[int\] = None',
    roadmap_model, main_content, flags=re.DOTALL
)

with open(main_path, "w") as f:
    f.write(main_content)
print("Updated main.py")
