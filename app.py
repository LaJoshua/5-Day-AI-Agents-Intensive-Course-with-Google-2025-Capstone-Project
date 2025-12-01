# ============================================
# Doctor Visit Companion – Full Streamlit App
# Gemini 1.5 Pro Integrated
# With Fixes for:
# 1. Gemini INVALID_ARGUMENT
# 2. Streamlit Nested Expanders
# ============================================

import streamlit as st
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any
from datetime import datetime, date, time
import uuid
import json
import os

# -------------------------
# Gemini LLM Imports
# -------------------------
from google import genai


# =========================
# Gemini LLM Client
# =========================

_gemini_client = None

def get_gemini_client():
    """
    Creates a global Gemini client instance.
    Requires GEMINI_API_KEY in environment.
    """
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client()
    return _gemini_client


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Calls Gemini 1.5 Pro and returns text output.

    We keep it simple and just prepend the system instructions
    to the user prompt in a single text input.
    """
    client = get_gemini_client()
    model_name = "gemini-2.5-flash"

    # Combine system + user into one prompt
    combined_prompt = (
        system_prompt.strip()
        + "\n\n---\n\n"
        + user_prompt.strip()
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[combined_prompt],
        )
    except Exception as e:
        return f"⚠️ Error calling Gemini: {e}"

    # Preferred: response.text convenience property
    text = getattr(response, "text", None)
    if text:
        return text.strip()

    # Fallback: stitch together parts if .text isn't available
    if not getattr(response, "candidates", None):
        return "⚠️ Gemini returned no candidates."
    candidate = response.candidates[0]
    if not candidate.content or not candidate.content.parts:
        return "⚠️ Gemini returned an empty response."

    parts = []
    for part in candidate.content.parts:
        if getattr(part, "text", None):
            parts.append(part.text)

    final = "\n".join(parts).strip()
    return final if final else "⚠️ Gemini returned an empty message."

# =========================
# Data Store / Memory Layer
# =========================

DATA_DIR = "data"
VISITS_FILE = os.path.join(DATA_DIR, "visits.json")
APPTS_FILE = os.path.join(DATA_DIR, "appointments.json")


@dataclass
class Medication:
    name: str
    dosage: str = ""
    instructions: str = ""
    explanation: str = ""


@dataclass
class Visit:
    id: str
    visit_date: str
    doctor_name: str
    doctor_type: str
    specialty: str
    location: str
    reason: str
    raw_notes: str
    summary: str = ""
    highlight_markdown: str = ""
    medications: List[Medication] = field(default_factory=list)


@dataclass
class Appointment:
    id: str
    visit_id: str
    appt_date: str
    appt_time: str
    doctor_name: str
    doctor_type: str
    specialty: str
    location: str
    notes: str = ""


def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_state():
    ensure_data_dir()
    visits_raw = load_json(VISITS_FILE, [])
    appts_raw = load_json(APPTS_FILE, [])

    visits = []
    for v in visits_raw:
        meds = [Medication(**m) for m in v.get("medications", [])]
        visits.append(
            Visit(
                id=v["id"],
                visit_date=v["visit_date"],
                doctor_name=v["doctor_name"],
                doctor_type=v["doctor_type"],
                specialty=v["specialty"],
                location=v["location"],
                reason=v["reason"],
                raw_notes=v["raw_notes"],
                summary=v.get("summary", ""),
                highlight_markdown=v.get("highlight_markdown", ""),
                medications=meds,
            )
        )

    appts = [Appointment(**a) for a in appts_raw]
    return visits, appts


def save_state(visits, appts):
    visits_raw = [asdict(v) for v in visits]
    appts_raw = [asdict(a) for a in appts]
    save_json(VISITS_FILE, visits_raw)
    save_json(APPTS_FILE, appts_raw)


# =========================
# AI Agents
# =========================

def summarize_visit_agent(visit: Visit) -> Dict[str, str]:
    system_prompt = (
        "You are a friendly, clear medical visit summarization assistant. "
        "Always remind users this is NOT medical advice."
    )

    user_prompt = f"""
Doctor Visit:

- Date: {visit.visit_date}
- Doctor: {visit.doctor_name}
- Type: {visit.doctor_type}
- Specialty: {visit.specialty}
- Location: {visit.location}
- Reason: {visit.reason}

Raw Notes:
\"\"\"{visit.raw_notes}\"\"\" 

Medications:
{[f"{m.name} ({m.dosage})" for m in visit.medications]}

Please provide:

**Visit Summary** – one friendly paragraph  
**Key Highlights** – bullet points (diagnosis, tests, treatment, follow-up)  
End with:  
**This summary does not replace medical advice.**
"""

    output = call_llm(system_prompt, user_prompt)

    return {"summary": output, "highlight_markdown": output}


def explain_medication_agent(med: Medication, visit: Visit) -> str:
    system_prompt = (
        "You explain medications in a simple, gentle, educational way. "
        "Do NOT alter doses. Encourage following doctor instructions."
    )

    user_prompt = f"""
Medication: {med.name}
Dosage: {med.dosage}

Visit Context:
- Reason: {visit.reason}
- Specialty: {visit.specialty}

Notes:
\"\"\"{visit.raw_notes[:1000]}\"\"\"

Explain:
- What the medication does  
- How it helps this condition  
- Common side effects (simple)  
- Safety reminders  
"""

    return call_llm(system_prompt, user_prompt)


def suggest_followup_agent(visit: Visit) -> str:
    system_prompt = (
        "You suggest reasonable follow-up timelines but always remind the user "
        "that their doctor's instructions come first."
    )

    user_prompt = f"""
Visit on: {visit.visit_date}
Reason: {visit.reason}

Notes:
\"\"\"{visit.raw_notes}\"\"\"

Provide:
- A suggested follow-up time  
- Bullet options (primary, specialist)  
- Disclaimer  
"""

    return call_llm(system_prompt, user_prompt)


# =========================
# Agent Trace
# =========================

def init_agent_trace():
    if "agent_trace" not in st.session_state:
        st.session_state.agent_trace = []


def log_agent_event(agent_name, visit_id, payload):
    st.session_state.agent_trace.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "agent": agent_name,
            "visit_id": visit_id,
            "payload": payload,
        }
    )


# =========================
# Streamlit UI
# =========================

def init_session_state():
    if "visits" not in st.session_state:
        visits, appts = load_state()
        st.session_state.visits = visits
        st.session_state.appointments = appts
    init_agent_trace()


def main():
    st.set_page_config(page_title="Doctor Visit Companion", page_icon="🩺", layout="wide")
    init_session_state()

    st.sidebar.title("🩺 Doctor Visit Companion")
    st.sidebar.write(
        "AI assistant to summarize, highlight, and organize your doctor visits.\n\n"
        "*Not a substitute for medical care.*"
    )

    tab_new, tab_history, tab_appts, tab_trace = st.tabs(
        ["➕ New Visit", "📚 Visit History", "📅 Appointments", "🔍 Agent Trace"]
    )

    with tab_new:
        render_new_visit_tab()
    with tab_history:
        render_history_tab()
    with tab_appts:
        render_appointments_tab()
    with tab_trace:
        render_trace_tab()


# =========================
# Tab 1: New Visit
# =========================

def render_new_visit_tab():
    st.header("Record a New Doctor Visit")

    col1, col2 = st.columns(2)

    with st.form("visit_form"):
        with col1:
            visit_date = st.date_input("Visit Date", value=date.today())
            doctor_name = st.text_input("Doctor's Name")
            doctor_type = st.selectbox("Doctor Type", ["Primary", "Specialist", "Other"])
            specialty = st.text_input("Specialty (e.g., Cardiology)")
            location = st.text_input("Location")

        with col2:
            reason = st.text_area("Reason for Visit", height=120)
            raw_notes = st.text_area("Visit Notes / Transcript", height=180)

        meds_text = st.text_area(
            "Medications (Name | dosage/instructions per line):",
            height=120,
        )

        submitted = st.form_submit_button("Save & Generate AI Summary")

    if submitted:
        visit_id = str(uuid.uuid4())

        medications = []
        for line in meds_text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|", maxsplit=1)]
            name = parts[0]
            dosage = parts[1] if len(parts) > 1 else ""
            medications.append(Medication(name=name, dosage=dosage))

        visit = Visit(
            id=visit_id,
            visit_date=visit_date.isoformat(),
            doctor_name=doctor_name,
            doctor_type=doctor_type,
            specialty=specialty,
            location=location,
            reason=reason,
            raw_notes=raw_notes,
            medications=medications,
        )

        with st.spinner("AI is analyzing your visit…"):
            summary_payload = summarize_visit_agent(visit)
            visit.summary = summary_payload["summary"]
            visit.highlight_markdown = summary_payload["highlight_markdown"]
            log_agent_event("VisitSummarizerAgent", visit_id, summary_payload)

            for med in visit.medications:
                explanation = explain_medication_agent(med, visit)
                med.explanation = explanation
                log_agent_event(
                    "MedicationExplainerAgent",
                    visit_id,
                    {"med": med.name, "preview": explanation[:150]},
                )

        st.session_state.visits.append(visit)
        save_state(st.session_state.visits, st.session_state.appointments)

        st.success("Visit saved!")

        st.subheader("AI Summary & Highlights")
        st.markdown(visit.highlight_markdown)

        if visit.medications:
            st.subheader("Medication Explanations")
            for med in visit.medications:
                st.markdown(f"### {med.name} ({med.dosage})")
                st.markdown(med.explanation)
                st.markdown("---")

        st.subheader("Follow-Up Suggestions")
        followup = suggest_followup_agent(visit)
        st.info(followup)
        log_agent_event("FollowUpPlannerAgent", visit_id, {"preview": followup[:150]})


# =========================
# Tab 2: Visit History
# =========================

def render_history_tab():
    st.header("Visit History")

    visits = st.session_state.visits
    if not visits:
        st.info("No visits yet.")
        return

    search = st.text_input("Search")

    filtered = []
    for v in visits:
        blob = " ".join(
            [
                v.doctor_name,
                v.specialty,
                v.reason,
                v.raw_notes,
                v.summary,
                v.highlight_markdown,
            ]
        ).lower()
        if search.lower() in blob:
            filtered.append(v)

    for v in sorted(filtered, key=lambda x: x.visit_date, reverse=True):
        with st.expander(f"{v.visit_date} – {v.doctor_name} ({v.specialty})"):
            st.markdown(f"**Location:** {v.location}")
            st.markdown(f"**Reason:** {v.reason}")

            st.markdown("### Summary & Highlights")
            st.markdown(v.highlight_markdown)

            if v.medications:
                st.markdown("### Medications")
                for m in v.medications:
                    st.write(f"- **{m.name}** — {m.dosage}")

                st.markdown("### Explanations")
                for m in v.medications:
                    st.markdown(f"#### {m.name} ({m.dosage})")
                    st.markdown(m.explanation)
                    st.markdown("---")


# =========================
# Tab 3: Appointments
# =========================

def render_appointments_tab():
    st.header("Appointments")

    visits = st.session_state.visits
    appts = st.session_state.appointments

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Create Follow-Up")

        if not visits:
            st.info("Record a visit first.")
            return

        choose = st.selectbox(
            "Choose Visit", [f"{v.visit_date} – {v.doctor_name}" for v in visits]
        )
        base = visits[[f"{v.visit_date} – {v.doctor_name}" for v in visits].index(choose)]

        appt_date = st.date_input("Appointment Date", value=date.today())
        appt_time = st.time_input("Appointment Time", value=time(9, 0))

        doctor_type = st.selectbox("Doctor Type", ["Same specialist", "Primary", "Other"])

        if doctor_type == "Same specialist":
            doctor_name = base.doctor_name
            specialty = base.specialty
        else:
            doctor_name = st.text_input("Doctor Name")
            specialty = st.text_input("Specialty")

        location = st.text_input("Location", value=base.location)
        notes = st.text_area("Notes", value=f"Follow-up for {base.reason}")

        if st.button("Save Appointment"):
            appt = Appointment(
                id=str(uuid.uuid4()),
                visit_id=base.id,
                appt_date=appt_date.isoformat(),
                appt_time=appt_time.strftime("%H:%M"),
                doctor_name=doctor_name,
                doctor_type=doctor_type,
                specialty=specialty,
                location=location,
                notes=notes,
            )
            appts.append(appt)
            st.session_state.appointments = appts
            save_state(st.session_state.visits, appts)
            st.success("Appointment Saved!")

    with col2:
        st.subheader("Upcoming Appointments")

        if not appts:
            st.info("No appointments yet.")
        else:
            for a in sorted(appts, key=lambda x: (x.appt_date, x.appt_time)):
                with st.expander(f"{a.appt_date} {a.appt_time} – {a.doctor_name}"):
                    st.write(f"**Specialty:** {a.specialty}")
                    st.write(f"**Location:** {a.location}")
                    st.write(f"**Notes:** {a.notes}")
                    st.code(
                        f"Event: Follow-up with {a.doctor_name}\n"
                        f"Date: {a.appt_date} at {a.appt_time}\n"
                        f"Location: {a.location}\n"
                    )


# =========================
# Tab 4: Agent Trace
# =========================

def render_trace_tab():
    st.header("Agent Trace")

    trace = st.session_state.agent_trace
    if not trace:
        st.info("No agent activity yet.")
        return

    filter_agent = st.selectbox("Filter", ["All"] + list({e["agent"] for e in trace}))

    for event in reversed(trace):
        if filter_agent != "All" and event["agent"] != filter_agent:
            continue

        with st.expander(f"{event['timestamp']} – {event['agent']}"):
            st.json(event["payload"])


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    main()
