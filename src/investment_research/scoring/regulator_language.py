"""One vocabulary for "a regulator says the endpoint / trial does not qualify".

Phase 4.3L Correction 2. The Kill Gate's regulator rules and the 8-K
extractor (``research/sec_material_8k.py``) used to carry separate, narrow
regular expressions, so wording one understood and the other did not ("not
an APPROPRIATE endpoint to demonstrate efficacy" vs. "not SUFFICIENT") fell
between them: extracted as nothing, or extracted and then invisible to the
gate. Both now build their patterns from the fragments below, and the
extractor takes its patterns from the rules themselves.
"""

from __future__ import annotations

#: What the regulator is judging: the endpoint (or the named measure).
ENDPOINT_TERM = (
    r"(?:primary\s+(?:efficacy\s+)?(?:end\s?-?point|outcome)|end\s?-?points?"
    r"|rvef|right\s+ventricular\s+ejection\s+fraction)"
)
#: A negative adequacy judgement ("not sufficient / an appropriate / adequate ...").
NEGATIVE_ADEQUACY = (
    r"(?:(?:is|was|are|were|be)\s+)?(?:not|never)\s+(?:be\s+)?(?:an?\s+|the\s+)?"
    r"(?:sufficient|appropriate|adequate|acceptable|suitable)\b"
    r"|\b(?:insufficient|inappropriate|inadequate|unacceptable|unsuitable)\b"
)
#: What the endpoint is judged unfit to do.
GOAL = r"(?:demonstrate|establish|support|show|prove)\b[^.]{0,40}?(?:efficacy|effectiveness|benefit)"
#: How a regulator's communication is reported.
SPEECH = r"(?:advised|informed|told|stated|indicated|communicated|concluded|determined|recommended|said)"
#: A positive adequacy word, for "does not consider X an appropriate endpoint".
ADEQUACY = r"(?:appropriate|adequate|acceptable|suitable)"
CONSIDER = r"(?:consider|agree|view|regard|accept|find)"

ENDPOINT_NOT_SUFFICIENT = (
    rf"\b{ENDPOINT_TERM}[^.]{{0,160}}?(?:{NEGATIVE_ADEQUACY})[^.]{{0,80}}?{GOAL}"
    rf"|\b{SPEECH}\b[^.]{{0,120}}?\b{ENDPOINT_TERM}[^.]{{0,120}}?(?:{NEGATIVE_ADEQUACY})"
    rf"|\bendpoint\b[^.]{{0,120}}?\bcannot (?:prove|demonstrate|establish)\b"
)
ENDPOINT_NOT_ACCEPTED = (
    rf"(?:does not|did not|would not)\s+{CONSIDER}\b[^.]{{0,120}}\b{ENDPOINT_TERM}"
    rf"[^.]{{0,120}}(?:{ADEQUACY}|establish\s+effectiveness|establish\s+efficacy)"
    rf"|(?:does not|did not|would not)\s+{CONSIDER}\b[^.]{{0,80}}\b{ADEQUACY}\b"
    rf"[^.]{{0,60}}\b{ENDPOINT_TERM}"
)
REGISTRATIONAL_STATUS_WITHDRAWN = (
    r"\bno longer\b[^.]{0,80}?\b(?:pivotal|registrational)\b"
    r"|\bno longer\b[^.]{0,40}\b(?:refers to|calls|describes|considers)\b[^.]{0,80}\bpivotal\b"
    r"|\b(?:does not|did not)\s+(?:consider|regard|view|call|describe|refer to)\b"
    r"[^.]{0,80}\b(?:pivotal|registrational)\b"
)
#: Words that anchor a "could not" to a REPORTED, past outcome ("... could not
#: be agreed to WHILE the trial is still ongoing"). A bare "could not" is as
#: likely to be a hypothetical ("the FDA could not agree ...") as a report.
REPORTED_ANCHOR = r"(?:while|during|until|because|at\s+that\s+time|at\s+the\s+time)"
#: The company reporting that it and the regulator did NOT agree on an
#: endpoint. A "could not" form counts only with a REPORTED_ANCHOR; the
#: extractor additionally refuses it after any conditional wording.
CONSULTATION_NO_AGREEMENT = (
    rf"\b(?:(?:was|were)\s+unable|unable|not\s+able|failed|did\s+not|have\s+not|has\s+not)\b"
    rf"[^.;]{{0,40}}\bagree(?:ment|d)?\b[^.]{{0,120}}\b{ENDPOINT_TERM}"
    rf"|\bno\s+agreement\b[^.]{{0,120}}\b{ENDPOINT_TERM}"
    rf"|\bdisagree\w*[^.]{{0,120}}\b{ENDPOINT_TERM}"
    rf"|\b{ENDPOINT_TERM}[^.;]{{0,160}}?\bcould\s+not\s+be\s+agreed(?:\s+(?:to|upon))?\b"
    rf"[^.;]{{0,80}}?\b{REPORTED_ANCHOR}\b"
    rf"|\bcould\s+not\s+(?:reach\s+)?agree(?:ment)?\b[^.;]{{0,80}}?\b{ENDPOINT_TERM}"
    rf"[^.;]{{0,80}}?\b{REPORTED_ANCHOR}\b"
)
