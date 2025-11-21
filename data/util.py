import re
from datetime import date, datetime
from typing import Any, List, Mapping, Optional

from data.dataclass import SOAP, Assessment, EHRRecord, History


class MIMICNoteParser:  # TODO: make the cleaned mimic data fit the FEMH class
    """
    Utility for parsing raw discharge / EHR text into a structured `EHRRecord`.

    Usage:
        parser = EHRTextParser(raw_text)
        record = parser.parse()
    """

    def __init__(
        self,
        text: str,
        note_id: Optional[str] = None,
        subject_id: Optional[str] = None,
        hadm_id: Optional[str] = None,
        icd_code: Optional[str] = None,
        short_title: Optional[str] = None,
        long_title: Optional[str] = None,
    ):
        self.text = text or ""
        self.note_id = note_id
        self.subject_id = subject_id
        self.hadm_id = hadm_id
        self.icd_code = icd_code
        self.short_title = short_title
        self.long_title = long_title

    def _extract_date(self, pattern: str) -> Optional[date]:
        """Extract a date from text using the given regex pattern (YYYY-M-D)."""
        match = re.search(pattern, self.text)
        if match:
            date_str = match.group(1)
            try:
                return datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                return None
        return None

    def _extract_text(self, header: str, next_headers: List[str]) -> Optional[str]:
        """
        Extract text between one header and the next header (or end of document).

        header: starting header
        next_headers: possible next headers (since the exact next header is not fixed)
        """
        next_headers_pattern = "|".join([re.escape(h) for h in next_headers])
        pattern = rf"{re.escape(header)}:(.*?)(?:{next_headers_pattern}|==========|$)"

        match = re.search(pattern, self.text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _get_value(self, pattern: str) -> Optional[str]:
        """Extract a single-line value (e.g., Sex, Service)."""
        match = re.search(pattern, self.text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def parse(self) -> EHRRecord:
        """Parse the raw text into a structured `EHRRecord`."""
        # 1. Basic info
        adm_date = self._extract_date(
            r"Admission Date:\s*\[\*\*(\d{4}-\d{1,2}-\d{1,2})\*\*\]"
        )
        dis_date = self._extract_date(
            r"Discharge Date:\s*\[\*\*(\d{4}-\d{1,2}-\d{1,2})\*\*\]"
        )
        sex = self._get_value(r"Sex:\s*(M|F)")
        service = self._get_value(r"Service:\s*(.*)")

        # Visit type from hospital course
        course_text = self._extract_text(
            "Brief Hospital Course", ["Medications on Admission"]
        )
        visit_type = (
            "Elective"
            if course_text and "electively" in course_text.lower()
            else "Emergency"
        )

        # 2. History (use Admission Meds as medication)
        pmh = self._extract_text(
            "Past Medical History", ["Social History", "Family History"]
        )
        surgical_hx = self._extract_text(
            "Major Surgical or Invasive Procedure", ["History of Present Illness"]
        )
        adm_meds = self._extract_text(
            "Medications on Admission", ["Discharge Medications"]
        )

        history = History(past=pmh, surgical=surgical_hx, medication=adm_meds)

        # 3. Administrative assessment
        assessment = Assessment(
            type=visit_type,
            date=dis_date,  # use discharge date as assessment date
            admission_date=adm_date,
            referral_source=None,
        )

        # 4. SOAP mapping
        # S: chief complaint + HPI
        cc = self._extract_text("Chief Complaint", ["Major Surgical"])
        hpi = self._extract_text("History of Present Illness", ["Past Medical History"])
        s_content = f"Chief Complaint: {cc}\n\nHPI: {hpi}" if cc and hpi else None

        # O: physical exam + results
        pe = self._extract_text(
            "Physical Exam", ["Neurological Examination", "Pertinent Results"]
        )
        neuro = self._extract_text(
            "Neurological Examination", ["PHYSICAL EXAM UPON DISCHARGE"]
        )
        labs = self._extract_text("Pertinent Results", ["Brief Hospital Course"])
        o_content = f"Physical Exam: {pe}\n\nNeuro: {neuro}\n\nLabs/Imaging: {labs}"

        # A: hospital course + discharge diagnosis
        course = self._extract_text(
            "Brief Hospital Course", ["Medications on Admission"]
        )
        dx = self._extract_text("Discharge Diagnosis", ["Discharge Condition"])
        a_content = f"Diagnosis: {dx}\n\nHospital Course: {course}"

        # P: meds + instructions + follow-up
        rx = self._extract_text("Discharge Medications", ["Discharge Disposition"])
        instructions = self._extract_text(
            "Discharge Instructions", ["Followup Instructions"]
        )
        followup = self._extract_text("Followup Instructions", ["Completed by"])
        p_content = (
            f"Meds: {rx}\n\nInstructions: {instructions}\n\nFollowup: {followup}"
        )

        soap = SOAP(S=s_content, O=o_content, A=a_content, P=p_content)

        # 5. Combine into EHRRecord
        return EHRRecord(
            id=self.note_id or "UNKNOWN_ID",
            visit_date=adm_date,
            department=service,
            sex=sex,
            visit_type=visit_type,
            assessment=assessment,
            history=history,
            icd_codes=[self.icd_code] if self.icd_code else [],
            soap=soap,
        )

    @staticmethod
    def _clean_field(value: Any) -> Optional[str]:
        """Normalize CSV fields: treat None/NaN as None, otherwise string."""
        if value is None:
            return None
        # Pandas NaN is not equal to itself
        if isinstance(value, float) and value != value:
            return None
        try:
            text = str(value).strip()
        except Exception:
            return None
        return text or None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "MIMICNoteParser":
        """
        Build a parser from a MIMIC CSV row with TEXT/ICD9_CODE metadata.

        Expected keys (case-sensitive): ROW_ID_x, SUBJECT_ID, HADM_ID, TEXT,
        ICD9_CODE, SHORT_TITLE, LONG_TITLE.
        """
        text = cls._clean_field(row.get("TEXT")) or ""
        note_id = cls._clean_field(row.get("ROW_ID_x")) or cls._clean_field(
            row.get("ROW_ID")
        )
        subject_id = cls._clean_field(row.get("SUBJECT_ID"))
        hadm_id = cls._clean_field(row.get("HADM_ID"))
        icd_code = cls._clean_field(row.get("ICD9_CODE"))
        short_title = cls._clean_field(row.get("SHORT_TITLE"))
        long_title = cls._clean_field(row.get("LONG_TITLE"))

        return cls(
            text=text,
            note_id=note_id,
            subject_id=subject_id,
            hadm_id=hadm_id,
            icd_code=icd_code,
            short_title=short_title,
            long_title=long_title,
        )

    @classmethod
    def parse_row(cls, row: Mapping[str, Any]) -> EHRRecord:
        """Convenience for turning a CSV row into an `EHRRecord`."""
        parser = cls.from_row(row)
        return parser.parse()


def parse_ehr_text(text: str) -> EHRRecord:
    """Convenience function to parse raw EHR text into `EHRRecord`."""
    return MIMICNoteParser(text).parse()
