# backend/filter_compiler.py
import json
import logging
from typing import Dict, List, Optional, Tuple

from openai import AsyncAzureOpenAI
from backend.settings import app_settings

# ---------- Public API ----------

async def build_odata_filter_from_query(
    query_text: str,
    aoai_client: AsyncAzureOpenAI,
) -> Tuple[Optional[str], Optional[dict]]:
    """
    Given a natural-language query (e.g., 'project managers in ohio with 10 years of experience'),
    ask the LLM to extract constraints for our index and convert them into an OData $filter string.

    Returns (odata_filter, debug_json). If no constraints => (None, debug_json).
    """
    if not isinstance(query_text, str) or not query_text.strip():
        return None, {"reason": "empty_query"}

    extracted = await _extract_constraints_with_llm(query_text, aoai_client)
    if not extracted:
        return None, {"reason": "llm_no_constraints"}

    try:
        filter_str = _constraints_to_odata(extracted)
        return (filter_str if filter_str else None), extracted
    except Exception as e:
        logging.warning("Failed to convert constraints to OData filter: %s", e)
        return None, {"reason": "conversion_error", "raw": extracted}


# ---------- LLM extraction ----------

async def _extract_constraints_with_llm(
    query_text: str,
    aoai_client: AsyncAzureOpenAI,
) -> Optional[dict]:
    """
    Prompts the LLM to emit a tightly-scoped JSON object that matches our index fields.
    We keep the schema intentionally small & opinionated for reliability.
    """

    system = (
        "You convert natural-language search requests into structured filter constraints "
        "for an Azure Cognitive Search index. Emit STRICT JSON only, no commentary."
    )

    # Schema notes the fields we actually support in the index:
    #   - client (filterable string)
    #   - states (filterable string collection of 2-letter state codes)
    #   - tags (filterable string collection)
    #   - employees (collection of complex):
    #       - name (string, filterable)
    #       - job_roles (string collection, filterable)
    #       - state (string, filterable)
    #       - status (string, filterable)  e.g., 'Active', 'Inactive'
    #       - years_experience (int, filterable)  -- we will only use min/max bounds
    #
    # We ask the model to normalize gently (e.g., 'Ohio' -> 'OH', 'project managers' -> 'Project Manager').
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
  "client_hints": [] | ["City of Canton", "Canton City"] // optional alternates if the user gave only a city
}}

Guidance:
- If the query mentions a city like "Canton" without a qualifier, you may set client to ["City of Canton"] and also include variants in client_hints.
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
        # Be defensive about code blocks
        cleaned = raw.strip().strip("`").replace("json\n", "").replace("JSON\n", "")
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception as e:
        logging.warning("LLM extraction failed: %s", e)
        return None


# ---------- OData composition ----------

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
    # Single-quote escape for OData literals
    return "'" + s.replace("'", "''") + "'"

def _constraints_to_odata(js: Dict) -> Optional[str]:
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

    # --- employees (complex collection) ---
    emp: Dict = js.get("employees") or {}
    emp_clauses: List[str] = []

    names = _as_list(emp.get("names"))
    if names:
        names_any = _or_join([f"e/name eq {_q(n)}" for n in names])
        emp_clauses.append(f"({names_any})")

    job_roles = _as_list(emp.get("job_roles"))
    if job_roles:
        roles_any = _or_join([f"r eq {_q(r)}" for r in job_roles])
        emp_clauses.append(f"(e/job_roles/any(r: {roles_any}))")

    emp_states = _as_list(emp.get("states"))
    if emp_states:
        es_any = _or_join([f"e/state eq {_q(s)}" for s in emp_states])
        emp_clauses.append(f"({es_any})")

    status_vals = _as_list(emp.get("status"))
    if status_vals:
        st_any = _or_join([f"e/status eq {_q(s)}" for s in status_vals])
        emp_clauses.append(f"({st_any})")

    min_y = emp.get("min_years_experience")
    if isinstance(min_y, (int, float)):
        emp_clauses.append(f"(e/years_experience ge {int(min_y)})")

    max_y = emp.get("max_years_experience")
    if isinstance(max_y, (int, float)):
        emp_clauses.append(f"(e/years_experience le {int(max_y)})")

    if emp_clauses:
        clauses.append(f"(employees/any(e: {_and_join(emp_clauses)}))")

    return _and_join(clauses)


def _as_list(x) -> List[str]:
    if not x:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if isinstance(v, (str, int, float)) and str(v).strip()]
    return [str(x)]
