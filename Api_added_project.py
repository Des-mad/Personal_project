import os
import re
import json
import sqlite3
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="B2B Lead Intelligence Agent", layout="wide")

DB_PATH = "lead_intelligence.db"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"  # check console.groq.com/docs/models for current free models


# ----------------- DATABASE SETUP (SQLITE) -----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS qualified_leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            company TEXT,
            stakeholder TEXT,
            scale TEXT,
            vertical TEXT,
            deal_score INTEGER,
            tier TEXT,
            pain_points TEXT,
            strategic_pitch TEXT,
            immediate_next_step TEXT,
            source TEXT DEFAULT 'Rules'
        )
    ''')
    # Migrate older databases that don't have the 'source' column yet
    cols = [row[1] for row in c.execute("PRAGMA table_info(qualified_leads)")]
    if "source" not in cols:
        c.execute("ALTER TABLE qualified_leads ADD COLUMN source TEXT DEFAULT 'Rules'")
    conn.commit()
    conn.close()


def insert_lead(data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO qualified_leads (
            timestamp, company, stakeholder, scale, vertical,
            deal_score, tier, pain_points, strategic_pitch, immediate_next_step, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['timestamp'], data['company'], data['stakeholder'], data['scale'],
        data['vertical'], data['deal_score'], data['tier'], data['pain_points'],
        data['strategic_pitch'], data['immediate_next_step'], data['source']
    ))
    conn.commit()
    conn.close()


def fetch_leads():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM qualified_leads ORDER BY id DESC", conn)
    conn.close()
    return df


init_db()


# ----------------- SHARED HELPERS -----------------
def has_any(text, keywords):
    """Whole-word / whole-phrase match (fixes 'cr' matching 'create', 'head' matching 'ahead')."""
    return any(re.search(r"\b" + re.escape(k) + r"\b", text) for k in keywords)


def tier_for(score):
    if score >= 80:
        return "Tier 1 (High Priority)"
    if score >= 50:
        return "Tier 2 (Moderate Fit)"
    return "Tier 3 (Disqualified)"


def rule_next_step(score, vertical):
    if score >= 80:
        if "Healthcare" in vertical:
            return "Direct Executive Outreach: Schedule 30-min Clinical Pilot & ROI alignment with leadership."
        if "Marketplace" in vertical:
            return "High-Touch GTM: Dispatch enterprise RFQ integration proposal and technical audit."
        return "Fast-track tailored proof-of-concept (PoC) scoping call with VP of Strategy."
    if score >= 50:
        return "Mid-Funnel Discovery: Send product capability deck and qualification survey to assess branch rollout roadmap."
    return "Automated Nurture: Add to standard educational newsletter drip; no direct rep outreach required."


def build_record(company, stakeholder, scale, vertical, score, pain_points, pitch, next_step, source):
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "company": company,
        "stakeholder": stakeholder,
        "scale": scale,
        "vertical": vertical,
        "deal_score": score,
        "tier": tier_for(score),
        "pain_points": pain_points,
        "strategic_pitch": pitch,
        "immediate_next_step": next_step,
        "source": source,
    }


# ----------------- RULE-BASED ENGINE (FALLBACK) -----------------
def analyze_lead_locally(company, stakeholder, scale, query):
    score = 50
    full_text = f"{company} {stakeholder} {scale} {query}".lower()

    if has_any(full_text, ["dental", "oral", "clinic", "clinics", "hospital", "patient", "patients", "diagnostic"]):
        vertical = "Healthcare & AI Oral Diagnostics"
        pain_points = "Turnaround report latency; Diagnostic variance across branch clinics; Manual imaging audits."
        pitch = "Deploy a centralized 30-day clinical pilot for 2D/3D image screening to standardize diagnosis and lift case acceptance."
    elif has_any(full_text, ["procure", "procurement", "supplier", "rfq", "manufacturing", "plant", "b2b", "sku"]):
        vertical = "B2B Marketplace & Supply Chain"
        pain_points = "Fragmented supplier quotes; High turnaround time on RFQ processing; Poor price discovery."
        pitch = "Integrate intelligent RFQ triage and automated supplier bid matching to compress procurement cycles."
    else:
        vertical = "B2B SaaS / Enterprise Automation"
        pain_points = "Manual workflow bottlenecks; Lack of structured data triage."
        pitch = "Standardize lead ingestion and process qualification using an automated intelligence layer."

    if has_any(full_text, ["multi", "chain", "network", "metro", "cr", "crore", "plants", "enterprise", "branches"]):
        score += 30
    if has_any(full_text, ["director", "head", "vp", "chief", "coo", "ceo", "founder"]):
        score += 15
    if has_any(full_text, ["solo", "single", "student", "free", "no budget"]):
        score -= 35
    score = max(15, min(score, 98))

    return build_record(company, stakeholder, scale, vertical, score, pain_points, pitch,
                        rule_next_step(score, vertical), "Rules")


# ----------------- LLM ENGINE (FREE GROQ API) -----------------
SYSTEM_PROMPT = """You are a B2B sales analyst qualifying inbound leads.
Return ONLY a JSON object (no markdown, no commentary) with exactly these keys:
  "vertical": short industry/vertical label (string),
  "pain_points": array of 2-4 short pain points inferred from the notes (strings),
  "deal_score": integer 0-100 for ICP fit,
  "strategic_pitch": one sentence tailored value pitch,
  "immediate_next_step": one sentence concrete next action for the sales rep.
Scoring guide: start at 50; +30 for multi-site/enterprise scale; +15 if the contact is a senior decision maker;
-35 for solo/tiny/student/no budget; clamp between 15 and 98. Base everything only on the lead details given."""


def extract_json(text):
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start:end + 1])


def analyze_lead_with_llm(company, stakeholder, scale, query, api_key, model):
    user_msg = (
        f"Company: {company}\nContact title: {stakeholder}\n"
        f"Scale/footprint: {scale}\nInbound notes: {query}"
    )
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.2,
            "max_tokens": 800,
        },
        timeout=30,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    data = extract_json(content)

    score = max(15, min(int(data["deal_score"]), 98))
    points = data["pain_points"]
    pain_points = "; ".join(str(p).strip().rstrip(";") for p in points) if isinstance(points, list) else str(points)
    vertical = str(data["vertical"]).strip()
    pitch = str(data["strategic_pitch"]).strip()
    next_step = str(data.get("immediate_next_step", "")).strip() or rule_next_step(score, vertical)
    if not pain_points or not pitch:
        raise ValueError("Model returned empty fields")

    return build_record(company, stakeholder, scale, vertical, score, pain_points, pitch, next_step, "LLM")


def analyze_lead(company, stakeholder, scale, query, use_llm, api_key, model):
    """Try the LLM first; fall back to rules on any failure. Returns (record, warning_or_None)."""
    if use_llm and api_key:
        try:
            return analyze_lead_with_llm(company, stakeholder, scale, query, api_key, model), None
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            reason = "rate limit reached" if code == 429 else f"API error {code}"
            warning = f"LLM unavailable ({reason}). Used rule-based scoring instead."
        except Exception as e:
            warning = f"LLM response could not be used ({type(e).__name__}). Used rule-based scoring instead."
        return analyze_lead_locally(company, stakeholder, scale, query), warning
    if use_llm and not api_key:
        return analyze_lead_locally(company, stakeholder, scale, query), "No API key set. Used rule-based scoring."
    return analyze_lead_locally(company, stakeholder, scale, query), None


def default_api_key():
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            key = ""
    return key


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("Analysis Engine")
    use_llm = st.toggle("Use LLM (free Groq API)", value=True)
    api_key = st.text_input("Groq API key", value=default_api_key(), type="password",
                            help="Free key from console.groq.com. Or set GROQ_API_KEY as an env var / Streamlit secret.")
    model_name = st.text_input("Model", value=DEFAULT_MODEL)
    st.caption("If the LLM fails or the free rate limit is hit, the app falls back to rule-based scoring automatically.")


# ----------------- STREAMLIT FRONTEND -----------------
st.title("B2B Lead Intelligence & GTM Triage Agent")
st.caption("LLM-assisted account scoring, ICP fit qualification, and strategic pitch generation, with a rule-based fallback")

df_current = fetch_leads()
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Ingested Leads", len(df_current))
with col2:
    tier1_count = len(df_current[df_current['tier'].str.contains("Tier 1")]) if not df_current.empty else 0
    st.metric("Tier 1 Priority Deals", tier1_count)
with col3:
    avg_score = round(df_current['deal_score'].mean(), 1) if not df_current.empty else 0.0
    st.metric("Avg ICP Fit Score", f"{avg_score}/100")
with col4:
    llm_count = int((df_current['source'] == "LLM").sum()) if not df_current.empty else 0
    st.metric("LLM-Analyzed Leads", llm_count)

st.divider()

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("1. Inbound Ingestion Form")

    flash = st.session_state.pop('flash', None)
    if flash:
        kind, msg = flash
        (st.warning if kind == "warning" else st.success)(msg)

    company_input = st.text_input("Company / Organization Name", "Apollo Care Dental Network")
    stakeholder_input = st.text_input("Contact Title / Stakeholder", "Chief Operating Officer")
    scale_input = st.text_input("Operational Scale / Footprint", "28 clinics across 3 states; 15,000 monthly patients")
    query_input = st.text_area(
        "Raw Inbound Requirement / Meeting Notes",
        "We are looking to implement AI triage across all our clinics to reduce diagnosis turnaround latency and standardize treatment plans."
    )

    if st.button("Run Analysis & Store", type="primary"):
        if not company_input or not query_input:
            st.error("Please provide at least a Company Name and Query.")
        else:
            with st.spinner("Analyzing lead..."):
                analysis, warning = analyze_lead(
                    company_input, stakeholder_input, scale_input, query_input,
                    use_llm, api_key, model_name,
                )
                insert_lead(analysis)
                st.session_state['latest_lead'] = analysis
                if warning:
                    st.session_state['flash'] = ("warning", warning)
                else:
                    st.session_state['flash'] = ("success", f"Lead analyzed ({analysis['source']}) and saved.")
                st.rerun()

with right_col:
    st.subheader("2. Strategic Account Triage Output")
    latest = st.session_state.get('latest_lead', None)

    if latest is None and not df_current.empty:
        latest = df_current.iloc[0].to_dict()

    if latest:
        st.markdown(f"### **{latest['company']}** — `{latest['vertical']}`")
        score_color = "green" if latest['deal_score'] >= 80 else ("orange" if latest['deal_score'] >= 50 else "red")
        st.markdown(
            f"**Fit Score:** :{score_color}[**{latest['deal_score']}/100**] | "
            f"**Priority Tier:** `{latest['tier']}` | **Engine:** `{latest.get('source', 'Rules')}`"
        )

        st.markdown("#### **Extracted Operational Pain Points**")
        for point in latest['pain_points'].split("; "):
            st.markdown(f"- {point}")

        st.markdown("#### **Tailored Strategic Value Pitch**")
        st.info(latest['strategic_pitch'])

        st.markdown("#### **Immediate Next Step**")
        st.warning(latest['immediate_next_step'])
    else:
        st.info("Submit a lead using the form on the left to see the generated intelligence brief.")

st.divider()
st.subheader("3. Pipeline Database & SQL Profiling")

df_display = fetch_leads()
if not df_display.empty:
    st.dataframe(
        df_display[['timestamp', 'company', 'vertical', 'deal_score', 'tier', 'source',
                    'pain_points', 'immediate_next_step']],
        use_container_width=True
    )

    csv_data = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Pipeline Analytics (CSV)",
        data=csv_data,
        file_name="b2b_lead_pipeline_analytics.csv",
        mime="text/csv"
    )
else:
    st.caption("No records stored in database yet.")