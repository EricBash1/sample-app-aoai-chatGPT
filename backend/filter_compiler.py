# backend/filter_compiler.py
import json
import logging
from typing import Dict, List, Optional, Tuple

from openai import AsyncAzureOpenAI
from backend.settings import app_settings

# ---------- Public APIs ----------

async def build_odata_filter_from_query(
    query_text: str,
    aoai_client: AsyncAzureOpenAI,
) -> Tuple[Optional[str], Optional[dict]]:
    """
    (Kept for backward compat) Builds a single filter targeting the *docs* index shape.
    New code paths should prefer: extract_constraints() + the target-specific builders below.
    """
    if not isinstance(query_text, str) or not query_text.strip():
        return None, {"reason": "empty_query"}

    extracted = await _extract_constraints_with_llm(query_text, aoai_client)
    if not extracted:
        return None, {"reason": "llm_no_constraints"}

    try:
        # Build a docs-index filter (no employee IDs here).
        filter_str = constraints_to_odata_for_docs_index(extracted, employee_ids=None)
        return (filter_str if filter_str else None), extracted
    except Exception as e:
        logging.warning("Failed to convert constraints to OData filter: %s", e)
        return None, {"reason": "conversion_error", "raw": extracted}


async def extract_constraints(
    query_text: str,
    aoai_client: AsyncAzureOpenAI,
) -> Optional[dict]:
    """Exported helper: get normalized constraints JSON from the LLM."""
    if not isinstance(query_text, str) or not query_text.strip():
        return None
    return await _extract_constraints_with_llm(query_text, aoai_client)


def is_employee_intent(constraints: Optional[dict]) -> bool:
    """True if the user appears to be targeting employees."""
    if not constraints:
        return False
    e = constraints.get("employees") or {}
    if any(bool(_as_list(e.get(k))) for k in ("names", "job_roles", "states", "status")):
        return True
    if any(isinstance(e.get(k), (int, float)) for k in ("min_years_experience", "max_years_experience")):
        return True
    return False


def constraints_to_odata_for_employees_index(js: Dict) -> Optional[str]:
    """
    Build an OData $filter for the **employees** index (flat fields).
    Expect fields: employee_id (key), name, job_roles (collection), state, status, years_experience (int).
    """
    clauses: List[str] = []

    emp: Dict = js.get("employees") or {}
    emp_clauses: List[str] = []

    names = _as_list(emp.get("names"))
    if names:
        names_any = _or_join([f"name eq {_q(n)}" for n in names])
        emp_clauses.append(f"({names_any})")

    job_roles = _as_list(emp.get("job_roles"))
    if job_roles:
        roles_any = _or_join([f"r eq {_q(r)}" for r in job_roles])
        emp_clauses.append(f"(job_roles/any(r: {roles_any}))")

    emp_states = _as_list(emp.get("states"))
    if emp_states:
        es_any = _or_join([f"state eq {_q(s)}" for s in emp_states])
        emp_clauses.append(f"({es_any})")

    status_vals = _as_list(emp.get("status"))
    if status_vals:
        st_any = _or_join([f"status eq {_q(s)}" for s in status_vals])
        emp_clauses.append(f"({st_any})")

    min_y = emp.get("min_years_experience")
    if isinstance(min_y, (int, float)):
        emp_clauses.append(f"(years_experience ge {int(min_y)})")

    max_y = emp.get("max_years_experience")
    if isinstance(max_y, (int, float)):
        emp_clauses.append(f"(years_experience le {int(max_y)})")

    if emp_clauses:
        clauses.append(_and_join(emp_clauses))

    # Optionally narrow by top-level states/tags/client if you mirror those on the employees index
    # (most teams don't). Nothing added here by default.

    return _and_join(clauses)


def constraints_to_odata_for_docs_index(js: Dict, employee_ids: Optional[List[str]]) -> Optional[str]:
    """
    Build an OData $filter for the **documents** index (your andy-* index).
    This index now has top-level fields (client, states, tags, proposal_date, etc.) and
    a **collection** field 'employee_ids' (Collection(Edm.String)) when present.
    """
    clauses: List[str] = []

    # --- client ---
    client_vals: List[str] = _as_list(js.get("client"))
    client_hints: List[str] = _as_list(js.get("client_hints"))
    client_any = _or_join(
        [f"client eq {_q(v)}" for v in (client_vals + client_hints)]
    )
    if client_any:
        clauses.append(f"({client_any})")

    # --- project states (top-level 'states' field) ---
    states_vals: List[str] = _as_list(js.get("states"))
    if states_vals:
        states_any = _or_join([f"s eq {_q(v)}" for v in states_vals])
        clauses.append(f"(states/any(s: {states_any}))")

    # --- tags (top-level collection) ---
    tags_vals: List[str] = _as_list(js.get("tags"))
    if tags_vals:
        tags_any = _or_join([f"t eq {_q(v)}" for v in tags_vals])
        clauses.append(f"(tags/any(t: {tags_any}))")

    # --- employee_ids (Collection(Edm.String)) ---
    if employee_ids:
        # Use search.in for a compact 'IN' style clause
        id_csv = ",".join(sorted({str(x) for x in employee_ids if str(x).strip()}))
        if id_csv:
            clauses.append(f"(employee_ids/any(id: search.in(id, '{id_csv}', ',')))")

    return _and_join(clauses)


# ---------- LLM extraction (internal) ----------

async def _extract_constraints_with_llm(
    query_text: str,
    aoai_client: AsyncAzureOpenAI,
) -> Optional[dict]:
    system = (
        "You convert natural-language search requests into structured filter constraints "
        "for an Azure Cognitive Search index. Emit STRICT JSON only, no commentary."
    )
    user = f"""
Extract constraints for this index. Output a SINGLE JSON object with this shape:

{{
  "client": [] | ["City of Example", ...],
  "states": [] | ["OH","PA",...],                      // Must be 2-letter US postal codes if states are mentioned
  "tags": [] | ["roundabout","bridge",...],            // Leave empty if unknown
  "employees": {{
     "names": [] | ["Jane Smith", ...],
     "job_roles": [] | ["Project Manager","Engineer",...],
     "states": [] | ["OH","PA",...],                   // 2-letter postal codes if present
     "status": [] | ["Active","Inactive"],             // Leave empty if unknown
     "min_years_experience": null | 0,
     "max_years_experience": null | 40
  }},
  "client_hints": [] | ["City of Canton", "Canton City"]
}}

Guidance:
- Normalize states to 2-letter codes when possible.
- If a value is not clearly stated, LEAVE ARRAYS EMPTY and set min/max to null. Do not hallucinate.
- Do not include any keys other than the ones above.

Query: {query_text}
"""
    try:
        resp = await aoai_client.chat.completions.create(
            model=app_settings.azure_openai.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
        )
        raw = resp.choices[0].message.content if resp.choices else "{}"
        cleaned = raw.strip().strip("`").replace("json\n", "").replace("JSON\n", "")
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception as e:
        logging.warning("LLM extraction failed: %s", e)
        return None


# ---------- Helpers ----------

def _or_join(parts: List[str]) -> Optional[str]:
    parts = [p for p in parts if p]
    if not parts:
        return None
    return " or ".join(parts)

def _and_join(parts: List[str]) -> Optional[str]:
    parts = [p for p in parts if p]
    if not parts:
        return None
    return " and ".join(parts)

def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"

def _as_list(x) -> List[str]:
    if not x:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if (isinstance(v, (str, int, float)) and str(v).strip())]
    return [str(x)]
