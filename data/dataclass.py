from dataclasses import dataclass, field, asdict
from typing import List, Optional
from datetime import date
import json


@dataclass
class Assessment:
    """emergency"""

    type: Optional[str] = None
    date: Optional[date] = None
    admission_date: Optional[date] = None
    referral_source: Optional[str] = None


@dataclass
class History:
    """nursing"""

    past: Optional[str] = None
    surgical: Optional[str] = None
    medication: Optional[str] = None


@dataclass
class SOAP:
    """emergency"""

    S: Optional[str] = None
    O: Optional[str] = None
    A: Optional[str] = None
    P: Optional[str] = None


@dataclass
class EHRRecord:
    id: str
    visit_date: Optional[date] = None
    department: Optional[str] = None
    sex: Optional[str] = None
    visit_type: Optional[str] = None
    assessment: Optional[Assessment] = None
    history: Optional[History] = None
    icd_codes: List[str] = field(default_factory=list)
    soap: Optional[SOAP] = None

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False)