import os
import io
import re
import uuid
from datetime import date, datetime
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.ensemble import IsolationForest
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

os.makedirs("static", exist_ok=True)

# 1. Self-contained Python OCR (RapidOCR - no external .exe required)
RAPID_OCR_ENGINE = None
try:
    from rapidocr_onnxruntime import RapidOCR
    RAPID_OCR_ENGINE = RapidOCR()
except Exception:
    try:
        from rapidocr import RapidOCR
        RAPID_OCR_ENGINE = RapidOCR()
    except Exception:
        RAPID_OCR_ENGINE = None

# 2. Fallback Tesseract OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    TESSERACT_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    GoogleTranslator = None
    TRANSLATOR_AVAILABLE = False

try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    pypdf = None
    PDF_AVAILABLE = False

app = FastAPI(title="VoltGuard | AI Energy & Bill Protection")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

TRANSLATION_CACHE: Dict[str, str] = {}
WHATSAPP_SESSIONS: Dict[str, Dict[str, Any]] = {}
ELECTRICIAN_BOOKINGS: List[Dict[str, Any]] = []


def translate_text_single(text: str, target: str) -> str:
    if not text or target == "en":
        return text
    cache_key = f"{target}:{text.strip()}"
    if cache_key in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[cache_key]
    if TRANSLATOR_AVAILABLE and GoogleTranslator is not None:
        try:
            trans = GoogleTranslator(source="en", target=target).translate(text)
            TRANSLATION_CACHE[cache_key] = trans
            return trans
        except Exception:
            return text
    return text


def extract_text_from_image(image_bytes: bytes) -> str:
    """Extracts text from a bill photo using RapidOCR (or Tesseract fallback)."""
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")

    if RAPID_OCR_ENGINE is not None:
        try:
            img_np = np.array(img)
            res = RAPID_OCR_ENGINE(img_np)
            if isinstance(res, tuple):
                items = res[0]
                if items:
                    return "\n".join([str(line[1]) for line in items if len(line) >= 2])
            elif hasattr(res, "txts") and res.txts:
                return "\n".join([str(t) for t in res.txts])
        except Exception:
            pass

    if TESSERACT_AVAILABLE and pytesseract is not None:
        try:
            return pytesseract.image_to_string(img)
        except Exception:
            pass

    return ""


def calculate_tariff_bill(units: float, consumer_type: str = "standard") -> float:
    """Calculates realistic Indian domestic or commercial slab tariff in Rupees."""
    if units <= 0:
        return 0.0

    if consumer_type == "commercial":
        if units <= 100:
            return units * 7.50
        elif units <= 300:
            return (100 * 7.50) + ((units - 100) * 8.90)
        else:
            return (100 * 7.50) + (200 * 8.90) + ((units - 300) * 9.50)

    if units <= 50:
        return units * 1.95
    elif units <= 100:
        return (50 * 1.95) + ((units - 50) * 3.10)
    elif units <= 200:
        return (50 * 1.95) + (50 * 3.10) + ((units - 100) * 4.80)
    elif units <= 300:
        return (50 * 1.95) + (50 * 3.10) + (100 * 4.80) + ((units - 200) * 7.70)
    elif units <= 400:
        return (50 * 1.95) + (50 * 3.10) + (100 * 4.80) + (100 * 7.70) + ((units - 300) * 9.00)
    else:
        return (50 * 1.95) + (50 * 3.10) + (100 * 4.80) + (100 * 7.70) + (100 * 9.00) + ((units - 400) * 9.50)


def parse_ocr_text(text: str) -> Dict[str, Any]:
    patterns = {
        "units": r"(?:billed\s*units|units\s*consumed|net\s*units|consumption|units)\s*[:\-=]?\s*([\d,]+(?:\.\d+)?)",
        "days": r"(?:billing\s*days|no\.?\s*of\s*days|bill\s*days|\bdays\b)\s*[:\-=]?\s*(\d{1,3})\b",
        "contracted_kw": r"(?:contracted\s*load|sanctioned\s*load|conn(?:ected)?\.?\s*load|\bcmd\b)\s*[:\-=]?\s*([\d,]+(?:\.\d+)?)",
        "recorded_md": r"(?:recorded\s*(?:maximum\s*demand|md)|maximum\s*demand|\brmd\b|\bbmd\b|\bmd\b)\s*[:\-=]?\s*([\d,]+(?:\.\d+)?)",
        "gross_amount": r"(?:gross\s*(?:bill\s*)?amount|net\s*(?:bill\s*)?amount|total\s*(?:bill\s*)?amount|bill\s*amount|amount\s*payable|payable\s*amount|net\s*payable|curr(?:ent)?\s*bill|total\s*amt|bill\s*amt|net\s*amt)\s*[:\-=]?\s*(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)",
        "gjs_subsidy": r"(?:gjs\s*subsidy|gruha\s*jyothi|subsidy)\s*[:\-=]?\s*(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)",
        "sd_available": r"(?:sd\s*available|security\s*deposit)\s*[:\-=]?\s*(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)",
        "energy_charges": r"(?:energy\s*charges)\s*[:\-=]?\s*(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)",
        "fixed_charges": r"(?:fixed\s*charges)\s*[:\-=]?\s*(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)",
        "customer_charges": r"(?:customer\s*charges)\s*[:\-=]?\s*(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)",
        "electricity_duty": r"(?:electricity\s*duty)\s*[:\-=]?\s*(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)"
    }
    extracted: Dict[str, Any] = {}
    for key, pat in patterns.items():
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                extracted[key] = float(m.group(1).replace(",", ""))
            except ValueError:
                pass

    # Extract string identifiers if present
    usc_m = re.search(r"(?:usc\s*no\.?|service\s*no\.?)\s*[:\-=]?\s*(\d{6,14})", text, flags=re.IGNORECASE)
    if usc_m:
        extracted["usc_no"] = usc_m.group(1)

    name_m = re.search(r"(?:name)\s*[:\-=]\s*([A-Z][A-Z\s\.]{2,35})", text)
    if name_m:
        extracted["consumer_name"] = name_m.group(1).strip()

    pres_m = re.search(r"present\s+[\d/\-]+\s+\d+\s+(\d+)", text, flags=re.IGNORECASE)
    prev_m = re.search(r"previous\s+[\d/\-]+\s+\d+\s+(\d+)", text, flags=re.IGNORECASE)
    if pres_m:
        extracted["present_kwh"] = int(pres_m.group(1))
    if prev_m:
        extracted["previous_kwh"] = int(prev_m.group(1))

    has_gjs = bool(re.search(r"(?:gruha\s*jyothi|gjs)", text, flags=re.IGNORECASE))
    extracted["has_gjs"] = has_gjs
    if not has_gjs and "gjs_subsidy" not in extracted:
        extracted["gjs_subsidy"] = 0.0

    return extracted


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/translate-batch")
async def translate_batch(payload: Dict[str, Any]):
    texts: List[str] = payload.get("texts", [])
    target: str = payload.get("lang", "en")

    if target == "en" or not texts:
        return {"translations": {t: t for t in texts}}

    result: Dict[str, str] = {}
    to_fetch: List[str] = []

    for t in texts:
        clean = t.strip()
        if not clean:
            result[t] = t
            continue
        cache_key = f"{target}:{clean}"
        if cache_key in TRANSLATION_CACHE:
            result[t] = TRANSLATION_CACHE[cache_key]
        else:
            to_fetch.append(clean)

    if to_fetch and TRANSLATOR_AVAILABLE and GoogleTranslator is not None:
        try:
            translator = GoogleTranslator(source="en", target=target)
            delimiter = " ||| "
            chunk_size = 25
            for i in range(0, len(to_fetch), chunk_size):
                chunk = to_fetch[i:i + chunk_size]
                joined = delimiter.join(chunk)
                translated_joined = translator.translate(joined)
                parts = [p.strip() for p in translated_joined.split("|||")]
                if len(parts) == len(chunk):
                    for orig, trans in zip(chunk, parts):
                        TRANSLATION_CACHE[f"{target}:{orig}"] = trans
                        result[orig] = trans
                else:
                    for orig in chunk:
                        try:
                            trans = translator.translate(orig)
                            TRANSLATION_CACHE[f"{target}:{orig}"] = trans
                            result[orig] = trans
                        except Exception:
                            result[orig] = orig
        except Exception:
            for orig in to_fetch:
                result[orig] = orig
    else:
        for orig in to_fetch:
            result[orig] = orig

    return {"translations": result}


@app.post("/api/ocr-bill")
async def ocr_bill(file: UploadFile = File(...)):
    content = await file.read()
    filename = (file.filename or "").lower()
    raw_text = ""

    if filename.endswith(".pdf") and PDF_AVAILABLE and pypdf is not None:
        try:
            reader = pypdf.PdfReader(io.BytesIO(content))
            for page in reader.pages:
                raw_text += (page.extract_text() or "") + "\n"
        except Exception:
            pass
    else:
        raw_text = extract_text_from_image(content)

    extracted = parse_ocr_text(raw_text) if raw_text.strip() else {}
    numeric_fields = {k: v for k, v in extracted.items() if k not in ("has_gjs", "usc_no", "consumer_name")}

    if not numeric_fields:
        if raw_text.strip():
            snippet = " ".join(raw_text.split())[:110]
            return {
                "success": False,
                "extracted": {},
                "note": f"Scanned receipt text (\"{snippet}...\"), but could not match standard labels. Please enter Units & Amount below."
            }
        return {
            "success": False,
            "extracted": {},
            "note": "Could not read text from this photo (image blurry or low light). Please type your bill's Units and Amount below."
        }

    return {
        "success": True,
        "extracted": extracted,
        "note": f"✅ Scanned {len(numeric_fields)} bill field(s) automatically! Verify below or adjust your appliances."
    }


@app.post("/api/analyze-bill")
async def analyze_bill(payload: Dict[str, Any]):
    consumer_type = payload.get("consumer_type", "standard")
    units = float(payload.get("units", 245))
    days = max(1, int(payload.get("days", 30)))
    contracted_kw = max(0.1, float(payload.get("contracted_kw", 2.0)))
    recorded_md = float(payload.get("recorded_md", 0.0))
    gross_amount = float(payload.get("gross_amount", 0.0))

    bhk = int(payload.get("bhk", 2))
    members = int(payload.get("members", 4))
    fans = int(payload.get("fans", 3))
    lights = int(payload.get("lights", 5))
    fridges = int(payload.get("fridges", 1))
    acs = int(payload.get("acs", 0))
    geysers = int(payload.get("geysers", 1))
    motors = int(payload.get("motors", 1))

    expected_daily_kwh = (
        (fans * 0.075 * 9) +
        (lights * 0.012 * 6) +
        (fridges * 1.35) +
        (acs * 1.25 * 5) +
        (geysers * 2.0 * 0.4) +
        (motors * 0.75 * 0.35) +
        (members * 0.22) +
        (bhk * 0.15)
    )
    if consumer_type == "commercial":
        expected_daily_kwh += 2.5

    expected_daily_kwh = round(expected_daily_kwh, 2)
    actual_daily_kwh = round(units / days, 2)
    projected_30d_units = round(actual_daily_kwh * 30, 1)
    expected_30d_units = round(expected_daily_kwh * 30, 1)

    excess_30d_units = max(0.0, round(projected_30d_units - expected_30d_units, 1))
    estimated_bill_30d = round(calculate_tariff_bill(projected_30d_units, consumer_type), 2)
    expected_bill_30d = round(calculate_tariff_bill(expected_30d_units, consumer_type), 2)

    if consumer_type == "gruha_jyothi" and projected_30d_units > 200:
        rupees_lost = estimated_bill_30d
    else:
        rupees_lost = round(max(0.0, estimated_bill_30d - expected_bill_30d), 2)

    alerts = []
    status = "SAFE"

    md_excess_pct = round(((recorded_md - contracted_kw) / contracted_kw) * 100, 1)
    if recorded_md > 0 and md_excess_pct > 25:
        status = "WARNING"
        alerts.append({
            "severity": "WARNING",
            "title": f"Peak Load Overload: {recorded_md} kW used vs. {contracted_kw} kW sanctioned (+{md_excess_pct}%)",
            "detail": "Running too many heavy appliances at the same time pushed your peak load past your sanctioned limit. Your electricity board may charge extra fixed demand penalties."
        })

    if consumer_type == "gruha_jyothi":
        if projected_30d_units > 200:
            status = "LEAK"
            alerts.append({
                "severity": "LEAK",
                "title": f"Gruha Jyothi 200-Unit Limit Crossed ({projected_30d_units} / 200 Units)",
                "detail": f"At {actual_daily_kwh} units/day, your monthly total hits {projected_30d_units} units—canceling your ₹0 free bill and costing ~₹{int(estimated_bill_30d)}."
            })
        elif projected_30d_units >= 175:
            if status != "LEAK":
                status = "WARNING"
            alerts.append({
                "severity": "WARNING",
                "title": f"Close to 200-Unit Free Limit ({projected_30d_units} / 200 Units)",
                "detail": f"Only {round(200 - projected_30d_units, 1)} units remaining before your free electricity subsidy stops."
            })

    if excess_30d_units > 30:
        status = "LEAK"
        alerts.append({
            "severity": "LEAK",
            "title": f"Unexplained Extra Power Usage (+{excess_30d_units} Units/Month)",
            "detail": f"Your bill shows {actual_daily_kwh} units/day, but your appliances only need ~{expected_daily_kwh} units/day. This extra consumption is costing you ~₹{int(rupees_lost)}/month."
        })
    elif excess_30d_units > 15:
        if status == "SAFE":
            status = "WARNING"
        alerts.append({
            "severity": "WARNING",
            "title": f"Slightly Higher Than Expected (+{excess_30d_units} Units/Month)",
            "detail": f"Your usage ({actual_daily_kwh} units/day) is moderately above your appliance estimate ({expected_daily_kwh} units/day). Check if a fridge or geyser is running longer than usual."
        })

    if not alerts:
        alerts.append({
            "severity": "SAFE",
            "title": "Your Bill Matches Your Appliances — No Leak Detected",
            "detail": f"Your daily usage ({actual_daily_kwh} units/day) is completely normal for the appliances you selected."
        })

    spoken_summary = (
        f"VoltGuard Bill Result: {alerts[0]['title']}. "
        f"Your bill averages {actual_daily_kwh} units per day compared to {expected_daily_kwh} units expected for your appliances. "
        f"Estimated extra money spent is {int(rupees_lost)} rupees per month. {alerts[0]['detail']}"
    )

    return {
        "status": status,
        "consumer_type": consumer_type,
        "actual_daily_kwh": actual_daily_kwh,
        "expected_daily_kwh": expected_daily_kwh,
        "projected_30d_units": projected_30d_units,
        "expected_30d_units": expected_30d_units,
        "rupees_lost": rupees_lost,
        "estimated_bill_30d": estimated_bill_30d,
        "alerts": alerts,
        "spoken_summary": spoken_summary
    }


@app.post("/api/ml-fingerprint")
async def ml_fingerprint(
    old_appliance_watts: float = Form(220.0),
    new_5star_watts: float = Form(65.0),
    upgrade_cost: float = Form(14500.0),
    csv_file: Optional[UploadFile] = File(None)
):
    df = None
    data_source_label = "Sample 30-Day Meter History"

    if csv_file is not None and csv_file.filename:
        try:
            raw = await csv_file.read()
            parsed = pd.read_csv(io.BytesIO(raw))
            if "Date" in parsed.columns and "Units_kWh" in parsed.columns and len(parsed) >= 7:
                parsed = parsed[["Date", "Units_kWh"] + (["Night_kWh"] if "Night_kWh" in parsed.columns else [])].dropna()
                parsed["Date"] = parsed["Date"].astype(str)
                parsed["Units_kWh"] = pd.to_numeric(parsed["Units_kWh"], errors="coerce")
                parsed = parsed.dropna(subset=["Units_kWh"])
                if len(parsed) >= 7:
                    if "Night_kWh" not in parsed.columns:
                        med = parsed["Units_kWh"].median()
                        parsed["Night_kWh"] = np.where(
                            parsed["Units_kWh"] > med * 1.35,
                            np.round(parsed["Units_kWh"] * 0.58, 2),
                            np.round(parsed["Units_kWh"] * 0.22, 2)
                        )
                    df = parsed.reset_index(drop=True)
                    data_source_label = f"Uploaded File: {csv_file.filename}"
        except Exception:
            df = None

    if df is None:
        np.random.seed(42)
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=30, freq="D")
        units = np.random.normal(5.7, 0.32, 30)
        night = np.random.normal(1.15, 0.14, 30)
        spike_days = [5, 6, 14, 15, 22, 23]
        units[spike_days] += np.array([4.9, 5.5, 5.2, 6.1, 4.8, 5.7])
        night[spike_days] += np.array([4.4, 5.0, 4.7, 5.5, 4.3, 5.1])
        df = pd.DataFrame({
            "Date": dates.strftime("%b %d"),
            "Units_kWh": np.round(units, 2),
            "Night_kWh": np.round(night, 2)
        })

    contamination = min(0.35, max(0.1, 4.0 / len(df)))
    iso = IsolationForest(contamination=contamination, random_state=42)
    df["flag"] = iso.fit_predict(df[["Units_kWh", "Night_kWh"]])
    median_u = float(df["Units_kWh"].median())
    df["is_anomaly"] = (df["flag"] == -1) & (df["Units_kWh"] > median_u + 0.75)

    normal_df = df[~df["is_anomaly"]]
    baseline_u = float(normal_df["Units_kWh"].median()) if not normal_df.empty else median_u
    excess_kwh = float(np.maximum(df["Units_kWh"] - baseline_u, 0).sum())
    rupees_lost = round(excess_kwh * 7.5, 2)

    anomaly_rows = df[df["is_anomaly"]]
    if len(anomaly_rows) == 0:
        status = "SAFE"
        headline = "No Unusual Power Spikes Found"
        explanation = "Your daily meter readings are steady and consistent. No hidden appliance leaks or overnight theft patterns were detected."
        action_steps = ["Keep monitoring your monthly bill to ensure consumption stays steady."]
    else:
        status = "LEAK"
        night_share = float((anomaly_rows["Night_kWh"] / anomaly_rows["Units_kWh"]).mean())
        consecutive_run = int(df["is_anomaly"].astype(int).groupby((~df["is_anomaly"]).cumsum()).sum().max())
        if night_share > 0.50 and consecutive_run < 6:
            headline = f"AI Found {len(anomaly_rows)} Days with Heavy Overnight Spikes (1 AM – 4 AM)"
            explanation = "Most of the extra electricity on these spike days was used late at night when normal appliances are off. This usually happens when an external wire is tapped into your line, or a water heater/motor is left switched ON overnight."
            action_steps = [
                "Use Tab 3 (Overnight Meter Check) tonight to confirm if your meter jumps while you sleep.",
                "Check if your geyser or water pump switch was left on overnight.",
                "Inspect the black service wire outside your meter box for any unauthorized hooks."
            ]
        else:
            headline = f"AI Found Continuous 24-Hour Power Drain (+{round(df['Units_kWh'].max() - baseline_u, 1)} Units/Day)"
            explanation = "Your meter jumped to a permanently higher level day and night. This happens when a refrigerator or AC compressor runs non-stop due to a worn rubber door seal, or when damp walls cause internal wiring leakage."
            action_steps = [
                "Check if your refrigerator door closes tightly and the back coils are not overheating.",
                "Run the 10-Minute MCB Check in Tab 4 to test for damp-wall wiring leakage.",
                "Check below how fast upgrading an old fridge/AC pays for itself."
            ]

    daily_kwh_saved = max(0.1, ((old_appliance_watts - new_5star_watts) / 1000.0) * 16.0)
    monthly_kwh_saved = round(daily_kwh_saved * 30, 1)
    monthly_rupees_saved = round(monthly_kwh_saved * 7.5, 2)
    payback_months = round(upgrade_cost / monthly_rupees_saved, 1) if monthly_rupees_saved > 0 else 999

    spoken_summary = (
        f"AI Meter Analysis Result: {headline}. {explanation} "
        f"Total extra electricity detected is {round(excess_kwh, 1)} units, costing about {int(rupees_lost)} rupees."
    )

    return {
        "status": status,
        "source_label": data_source_label,
        "headline": headline,
        "explanation": explanation,
        "action_steps": action_steps,
        "baseline_kwh": round(baseline_u, 2),
        "excess_kwh": round(excess_kwh, 1),
        "rupees_lost": rupees_lost,
        "chart": {
            "dates": df["Date"].tolist(),
            "units": df["Units_kWh"].tolist(),
            "night": df["Night_kWh"].tolist(),
            "anomalies": df["is_anomaly"].tolist()
        },
        "roi": {
            "monthly_kwh_saved": monthly_kwh_saved,
            "monthly_rupees_saved": monthly_rupees_saved,
            "payback_months": payback_months
        },
        "spoken_summary": spoken_summary
    }


@app.post("/api/sleep-tenant-guard")
async def sleep_tenant_guard(payload: Dict[str, Any]):
    r_10pm = float(payload.get("reading_10pm", 4120.5))
    r_6am = float(payload.get("reading_6am", 4123.7))
    fridge_on = bool(payload.get("fridge_on", True))
    ac_hours = float(payload.get("ac_night_hours", 0.0))

    night_units = max(0.0, round(r_6am - r_10pm, 2))
    expected_night = round((0.55 if fridge_on else 0.05) + (ac_hours * 1.15), 2)
    unexplained_night = max(0.0, round(night_units - expected_night, 2))
    monthly_night_loss = round(unexplained_night * 30 * 7.5, 2)

    if unexplained_night > 0.8:
        sleep_status = "LEAK"
        sleep_msg = f"Warning: Your meter used {night_units} units overnight, which is {unexplained_night} units MORE than your fridge/AC needs (costing ₹{int(monthly_night_loss)}/month). Either a water heater was left on or someone is tapping your wire at night."
    else:
        sleep_status = "SAFE"
        sleep_msg = f"Safe: Your overnight usage ({night_units} units) matches your fridge and AC standby usage ({expected_night} units). No overnight theft or leak detected."

    sub_units_30d = float(payload.get("tenant_submeter_units", 165.0))
    tenant_fans = int(payload.get("tenant_fans", 1))
    tenant_lights = int(payload.get("tenant_lights", 3))
    tenant_fridge = bool(payload.get("tenant_fridge", False))
    tenant_ac_hrs = float(payload.get("tenant_ac_hrs", 0.0))

    expected_tenant_30d = round(30 * (
        (tenant_fans * 0.075 * 10) +
        (tenant_lights * 0.012 * 6) +
        (1.35 if tenant_fridge else 0.0) +
        (tenant_ac_hrs * 1.2) +
        0.25
    ), 1)
    tenant_overcharge_units = max(0.0, round(sub_units_30d - expected_tenant_30d, 1))
    tenant_overcharge_rs = round(tenant_overcharge_units * 8.0, 2)

    if tenant_overcharge_units > 25:
        tenant_status = "LEAK"
        tenant_msg = f"Sub-Meter Overcharge: Your room sub-meter shows {sub_units_30d} units, which is +{tenant_overcharge_units} units higher than your room appliances can use (₹{int(tenant_overcharge_rs)}/month extra). Common building lights or a water motor may be connected to your sub-meter."
    else:
        tenant_status = "SAFE"
        tenant_msg = f"Sub-Meter Accurate: Your {sub_units_30d} units match your room's expected {expected_tenant_30d} units."

    spoken_summary = f"Overnight Meter Check: {sleep_msg} Rented Room Sub-Meter Check: {tenant_msg}"
    return {
        "sleep": {
            "status": sleep_status,
            "night_units": night_units,
            "expected_night": expected_night,
            "unexplained_night": unexplained_night,
            "monthly_loss_rs": monthly_night_loss,
            "message": sleep_msg
        },
        "tenant": {
            "status": tenant_status,
            "recorded_units": sub_units_30d,
            "expected_units": expected_tenant_30d,
            "overcharge_units": tenant_overcharge_units,
            "overcharge_rs": tenant_overcharge_rs,
            "message": tenant_msg
        },
        "spoken_summary": spoken_summary
    }


@app.post("/api/mcb-wizard")
async def mcb_wizard(payload: Dict[str, Any]):
    blinks = int(payload.get("blinks", 0))
    seconds = max(10, int(payload.get("seconds", 60)))
    imp_per_kwh = max(100, int(payload.get("imp_per_kwh", 3200)))

    phantom_watts = round((blinks * 3_600_000) / (seconds * imp_per_kwh), 1)
    monthly_kwh = round((phantom_watts / 1000.0) * 24 * 30, 1)
    monthly_loss_rs = round(monthly_kwh * 7.5, 2)

    if phantom_watts > 45:
        status = "LEAK"
        verdict = f"Wall / Earthing Wiring Leak Detected ({phantom_watts} Watts)"
        advice = f"Even with all switches OFF, your meter is burning {phantom_watts} Watts ({monthly_kwh} units/month = ₹{int(monthly_loss_rs)}/month). Call an electrician to check for damp-wall current leakage."
    elif phantom_watts > 15:
        status = "WARNING"
        verdict = f"Minor Standby Power Drain ({phantom_watts} Watts)"
        advice = f"Caused by plugged-in TV boxes, Wi-Fi routers, or chargers. Switching off wall plugs saves ₹{int(monthly_loss_rs)}/month."
    else:
        status = "SAFE"
        verdict = f"Wiring Healthy — No Leak ({phantom_watts} Watts)"
        advice = "Your home wiring is safe and does not waste electricity when appliances are turned off."

    spoken_summary = f"MCB Wiring Check: {verdict}. {advice}"
    return {
        "status": status,
        "phantom_watts": phantom_watts,
        "monthly_kwh": monthly_kwh,
        "monthly_loss_rs": monthly_loss_rs,
        "verdict": verdict,
        "advice": advice,
        "spoken_summary": spoken_summary
    }


# ==============================================================================
# TAB 5A: DIRECT ELECTRICIAN BOOKING & INSPECTION CHECKLIST
# ==============================================================================
@app.post("/api/book-electrician")
async def book_electrician(payload: Dict[str, Any]):
    booking_id = "VG-ELC-" + str(uuid.uuid4().hex[:5]).upper()
    customer_name = payload.get("name", "").strip() or "Homeowner"
    phone = payload.get("phone", "").strip() or "Not Provided"
    address = payload.get("address", "").strip() or "Local Area"
    electrician = payload.get("electrician", "Suresh Electricals (Licensed Grade-A)")
    slot = payload.get("slot", "Today — Within 2 Hours")
    issue = payload.get("issue", "High Bill / Wall Earthing & Peak Load Inspection")

    booking_record = {
        "booking_id": booking_id,
        "date": date.today().isoformat(),
        "customer_name": customer_name,
        "phone": phone,
        "address": address,
        "electrician": electrician,
        "slot": slot,
        "issue": issue,
        "checklist": [
            "1. Clamp Meter Earth-Leakage Test (Check >30mA leakage with indoor switches OFF)",
            "2. Neutral-to-Earth Socket Voltage Check (Must be < 3V)",
            "3. Peak Load Balancing (Distribute heavy appliances across phases to avoid MD penalties)",
            "4. Geyser, Fridge Compressor & Inverter Auto-Cutoff Inspection",
            "5. Outdoor Service Cable & Meter Seal Tapping Check"
        ]
    }
    ELECTRICIAN_BOOKINGS.append(booking_record)
    return {"success": True, "booking": booking_record}


@app.post("/api/generate-document")
async def generate_document(payload: Dict[str, Any]):
    doc_type = payload.get("doc_type", "electrician")
    consumer_name = payload.get("name", "Consumer")
    usc_no = payload.get("usc_no", "Not Provided")
    circle = payload.get("circle", "Electricity Sub-Division Office")
    units = payload.get("units", 0)
    rupees_lost = payload.get("rupees_lost", 0)
    finding = payload.get("finding", "Unexplained excess electricity consumption")

    if doc_type == "electrician":
        text = f"""====================================================================
VOLTGUARD — ELECTRICIAN WIRING & APPLIANCE INSPECTION CHECKLIST
====================================================================
Date: {date.today().isoformat()}
Consumer Name: {consumer_name} | Service No (USC): {usc_no}
Detected Issue: {finding}
Estimated Extra Cost: Rs. {rupees_lost}/month ({units} units/30 days)

ON-SITE CHECKS FOR ELECTRICIAN:
[ ] 1. CLAMP METER EARTH-LEAKAGE TEST:
       Turn OFF all indoor switches. Measure Phase and Neutral current
       at the main MCB board. Any difference > 30mA confirms wall earthing leak.

[ ] 2. NEUTRAL-TO-EARTH VOLTAGE TEST:
       Measure voltage between Neutral and Earth pins on kitchen/bathroom
       sockets. Reading > 3V indicates a loose neutral joint.

[ ] 3. FRIDGE & GEYSER THERMOSTAT CHECK:
       Check if water heater auto-cutoff or fridge compressor is running
       continuously without tripping.

[ ] 4. INVERTER CHARGER CHECK:
       Unplug inverter AC input and observe if meter blink speed drops.

[ ] 5. SERVICE WIRE INSPECTION:
       Inspect overhead cable from pole to meter box for damaged insulation
       or unauthorized wire tapping.
====================================================================
Generated by VoltGuard AI Energy Protection System
"""
        filename = "VoltGuard_Electrician_Checklist.txt"
    else:
        text = f"""To,
The Assistant Engineer (Operations),
Electricity Distribution Office,
{circle}

Date: {date.today().isoformat()}
Subject: Request for Meter Accuracy Check & Service Line Inspection — Service No: {usc_no}

Respected Sir/Madam,

I am an electricity consumer under your section with the following details:
- Consumer Name: {consumer_name}
- Service Connection (USC) No: {usc_no}
- Recent Billed / Projected Units: {units} kWh

During a recent audit of my electricity bill, an unexplained consumption spike was detected ({finding}), resulting in an extra monthly burden of approximately Rs. {rupees_lost}.

I have already checked my indoor appliances and MCB switches to rule out internal faults. Therefore, I kindly request your office to:
1. Inspect the external service cable and meter terminal seals for any loose neutral connection or unauthorized tapping.
2. Conduct a standard meter accuracy test (accu-check) on my meter.
3. Verify my Recorded Maximum Demand against my sanctioned load.

I am ready to pay the prescribed meter-testing fee if required. Kindly schedule an inspection at your earliest convenience.

Thanking you,

Yours faithfully,
{consumer_name}
Service No: {usc_no}
"""
        filename = "VoltGuard_DISCOM_Dispute_Letter.txt"

    return JSONResponse({"filename": filename, "content": text})


# ==============================================================================
# TAB 5B: INTERACTIVE WHATSAPP BILL SCANNER & DIAGNOSTIC CHATBOT
# ==============================================================================
def build_bill_explanation_message(ex: Dict[str, Any], source_note: str = "") -> str:
    units = ex.get("units", 287)
    days = ex.get("days", 30)
    cmd = ex.get("contracted_kw", 1.0)
    rmd = ex.get("recorded_md", 3.05)
    gross = ex.get("gross_amount", 1838.0)
    pres = ex.get("present_kwh")
    prev = ex.get("previous_kwh")
    sd = ex.get("sd_available")
    has_gjs = ex.get("has_gjs", False)
    gjs_sub = ex.get("gjs_subsidy", 0.0)

    lines = [f"📄 *VoltGuard Bill Breakdown* {source_note}".strip()]
    if ex.get("consumer_name"):
        lines.append(f"👤 *Consumer:* {ex['consumer_name']} (USC: {ex.get('usc_no', 'N/A')})")
    if pres and prev:
        lines.append(f"🔢 *Meter Reading:* {prev} → {pres} = *{units} Units* ({days} days)")
    else:
        lines.append(f"⚡ *Units Used:* *{units} kWh* in {days} days (~{round(units / max(1, days), 1)} units/day)")

    lines.append(f"💰 *Total Bill Amount:* *₹{gross}*")

    if has_gjs and gjs_sub > 0:
        lines.append(f"🎁 *Gruha Jyothi Subsidy:* ₹{gjs_sub} applied")
    else:
        lines.append("ℹ️ *Subsidy:* *₹0 (No subsidy on this bill)*")

    if sd:
        lines.append(f"🏦 *SD Available (₹{sd}):* This is your refundable Security Deposit, not a subsidy.")

    if rmd and cmd and rmd > cmd * 1.25:
        lines.append(f"🚨 *Peak Load Alert:* Your Recorded MD is *{rmd} kW*, which is higher than your Sanctioned Load (*{cmd} kW*)!")

    lines.append("")
    lines.append("❓ *Question 1 of 2:* To check if your units are normal or if power is leaking—*how many Fans, Fridges, ACs, or Geysers* do you use at home?")
    lines.append("_(Example: \"3 fans, 1 fridge, 1 geyser, no AC\")_")
    return "\n".join(lines)


def process_whatsapp_turn(
    session_id: str,
    message_text: str = "",
    file_bytes: Optional[bytes] = None,
    filename: str = "",
    use_sample_bill: bool = False,
    lang: str = "en"
) -> Dict[str, Any]:
    sess = WHATSAPP_SESSIONS.setdefault(session_id, {"stage": "awaiting_bill", "bill": {}})
    msg = (message_text or "").strip()

    # Case 1: User uploaded a bill photo/PDF or clicked Sample Bill
    if file_bytes is not None or use_sample_bill:
        if use_sample_bill:
            extracted = {
                "consumer_name": "YARAKALA RAJITHA",
                "usc_no": "109963665",
                "present_kwh": 1127,
                "previous_kwh": 840,
                "units": 287,
                "days": 30,
                "contracted_kw": 1.0,
                "recorded_md": 3.05,
                "gross_amount": 1838.0,
                "sd_available": 200.0,
                "has_gjs": False,
                "gjs_subsidy": 0.0
            }
            reply = build_bill_explanation_message(extracted, "(Sample Receipt)")
        else:
            raw_text = ""
            if filename.lower().endswith(".pdf") and PDF_AVAILABLE and pypdf is not None:
                try:
                    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                    for p in reader.pages:
                        raw_text += (p.extract_text() or "") + "\n"
                except Exception:
                    pass
            else:
                raw_text = extract_text_from_image(file_bytes)

            extracted = parse_ocr_text(raw_text) if raw_text.strip() else {}
            if "units" not in extracted:
                sess["stage"] = "awaiting_units_manual"
                reply = (
                    "📷 I received your bill photo, but some numbers were hard to read.\n\n"
                    "❓ *Quick Question:* Please reply with your *Units Consumed* and *Bill Amount* "
                    "(Example: *287 units, 1838 rupees*)."
                )
                return {
                    "reply": translate_text_single(reply, lang),
                    "quick_replies": ["287 units, 1838 rupees", "Send Sample Bill"]
                }
            reply = build_bill_explanation_message(extracted)

        sess["bill"] = extracted
        sess["stage"] = "awaiting_appliances"
        return {
            "reply": translate_text_single(reply, lang),
            "quick_replies": [
                "3 fans, 1 fridge, 1 geyser, no AC",
                "2 fans, 1 fridge, 1 AC",
                "Explain Peak Load Alert"
            ]
        }

    # Case 2: Manual units entry if OCR couldn't read
    if sess.get("stage") == "awaiting_units_manual":
        nums = re.findall(r"(\d+(?:\.\d+)?)", msg)
        if nums:
            units = float(nums[0])
            amt = float(nums[1]) if len(nums) > 1 else round(calculate_tariff_bill(units), 0)
            extracted = {"units": units, "days": 30, "gross_amount": amt, "has_gjs": False, "gjs_subsidy": 0.0}
            sess["bill"] = extracted
            sess["stage"] = "awaiting_appliances"
            reply = build_bill_explanation_message(extracted)
            return {
                "reply": translate_text_single(reply, lang),
                "quick_replies": ["3 fans, 1 fridge, 1 geyser, no AC", "2 fans, 1 fridge, 1 AC"]
            }

    # Case 3: User asks about Peak Load / MD or Subsidy
    if "peak" in msg.lower() or "md" in msg.lower() or "load" in msg.lower():
        bill = sess.get("bill", {})
        cmd = bill.get("contracted_kw", 1.0)
        rmd = bill.get("recorded_md", 3.05)
        reply = (
            f"⚡ *What is Recorded MD ({rmd} kW) vs Sanctioned Load ({cmd} kW)?*\n"
            f"• *Sanctioned Load ({cmd} kW):* The maximum power limit approved for your meter.\n"
            f"• *Recorded MD ({rmd} kW):* The highest power pulled at one moment when heavy appliances (geyser + water motor + iron/AC) ran together.\n\n"
            f"💡 *Tip:* Avoid running your geyser and water pump at the exact same time, or ask an electrician to check for short-circuit spikes.\n\n"
            f"❓ Now tell me: *How many Fans, Fridges, ACs, or Geysers* do you use at home?"
        )
        return {
            "reply": translate_text_single(reply, lang),
            "quick_replies": ["3 fans, 1 fridge, 1 geyser, no AC", "Book an Electrician"]
        }

    # Case 4: User replies with their appliances -> Bot calculates leak & asks Question 2!
    if sess.get("stage") == "awaiting_appliances":
        bill = sess.get("bill", {})
        units = float(bill.get("units", 287))
        days = max(1, int(bill.get("days", 30)))

        def extract_count(keyword: str, default: int = 0) -> int:
            m = re.search(rf"(\d+)\s*{keyword}", msg, flags=re.IGNORECASE)
            if m:
                return int(m.group(1))
            if re.search(rf"no\s*{keyword}", msg, flags=re.IGNORECASE):
                return 0
            return default

        fans = extract_count("fan", 2)
        fridges = extract_count("fridge", 1)
        acs = extract_count("ac", 0)
        geysers = extract_count("geyser", 0)

        expected_daily = round((fans * 0.68) + (fridges * 1.35) + (acs * 6.25) + (geysers * 0.8) + 1.1, 2)
        expected_30d = round(expected_daily * 30, 1)
        actual_30d = round((units / days) * 30, 1)
        extra_units = max(0.0, round(actual_30d - expected_30d, 1))
        extra_rs = int(round(extra_units * 7.5))

        sess["stage"] = "awaiting_action"
        if extra_units > 30:
            reply = (
                f"🚨 *Leak Diagnosis Complete!*\n"
                f"• *Your Bill Pace:* {actual_30d} units/month\n"
                f"• *Expected for your appliances:* ~{expected_30d} units/month\n"
                f"• *Unexplained Extra Usage:* *+{extra_units} units/month (~₹{extra_rs} wasted)*\n\n"
                f"This usually happens due to a *wall wiring earthing leak*, a *stuck geyser thermostat*, or *neighbor wire tapping*.\n\n"
                f"❓ *Question 2 of 2:* What would you like to do next?\n"
                f"1️⃣ Reply *TEST* for a 1-minute meter blink wiring check\n"
                f"2️⃣ Reply *BOOK* to book a certified VoltGuard electrician"
            )
        else:
            reply = (
                f"✅ *Good News! Your Usage Matches Your Appliances*\n"
                f"• *Your Bill Pace:* {actual_30d} units/month\n"
                f"• *Expected for your appliances:* ~{expected_30d} units/month\n\n"
                f"Your unit consumption is normal! However, if your Recorded MD exceeded your sanctioned load, avoid running heavy appliances simultaneously.\n\n"
                f"❓ Would you like to *BOOK* an electrician for a load check or *UPLOAD* another bill?"
            )

        return {
            "reply": translate_text_single(reply, lang),
            "quick_replies": ["1️⃣ TEST (1-Min Wiring Check)", "2️⃣ BOOK Electrician Now"]
        }

    # Case 5: User chooses TEST or BOOK
    if "test" in msg.lower() or "1" == msg.strip():
        reply = (
            "🔌 *1-Minute Meter LED Blink Test:*\n"
            "1. Turn OFF all indoor switches (Fridge, Geyser, AC, TV, Inverter).\n"
            "2. Look at the red blinking light on your electricity meter.\n"
            "3. Count how many times it blinks in *60 seconds*.\n\n"
            "👉 If it blinks *more than 3 times* in 60 seconds with everything OFF, you have a wall wiring leak! Use *Tab 4* to calculate exact watts lost or reply *BOOK* to call an electrician."
        )
        return {
            "reply": translate_text_single(reply, lang),
            "quick_replies": ["2️⃣ BOOK Electrician Now", "Send Another Bill"]
        }

    if "book" in msg.lower() or "2" == msg.strip():
        b_id = "VG-WA-" + str(uuid.uuid4().hex[:4]).upper()
        reply = (
            f"🛠️ *Electrician Booking Request Created!*\n"
            f"• *Reference ID:* {b_id}\n"
            f"• *Inspection Checklist Included:* Earth-leakage clamp test, Neutral voltage check & Peak MD load balancing.\n\n"
            f"👉 Complete your preferred time slot on the *left panel of Tab 5* to confirm technician dispatch!"
        )
        return {
            "reply": translate_text_single(reply, lang),
            "quick_replies": ["Send Another Bill"]
        }

    # Default greeting / reset
    sess["stage"] = "awaiting_bill"
    welcome = (
        "👋 Namaste! I am the *VoltGuard WhatsApp Bill Assistant*.\n\n"
        "📎 *Send a photo of your electricity bill* (or tap *Send Sample Bill* below). "
        "I will explain every charge on your bill and ask 2 quick questions to find out if power is leaking!"
    )
    return {
        "reply": translate_text_single(welcome, lang),
        "quick_replies": ["Send Sample Bill"]
    }


@app.post("/api/whatsapp-chat")
async def whatsapp_chat(
    session_id: str = Form("web-demo-user"),
    message: str = Form(""),
    use_sample: str = Form("false"),
    lang: str = Form("en"),
    file: Optional[UploadFile] = File(None)
):
    file_bytes = await file.read() if (file is not None and file.filename) else None
    filename = file.filename if file is not None else ""
    res = process_whatsapp_turn(
        session_id=session_id,
        message_text=message,
        file_bytes=file_bytes,
        filename=filename or "",
        use_sample_bill=(use_sample.lower() == "true"),
        lang=lang
    )
    return JSONResponse(res)


@app.post("/api/whatsapp-webhook")
async def whatsapp_twilio_webhook(
    From: str = Form("whatsapp:+910000000000"),
    Body: str = Form(""),
    NumMedia: int = Form(0)
):
    """Live webhook ready for Twilio / Meta WhatsApp Business API integration."""
    res = process_whatsapp_turn(session_id=From, message_text=Body)
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{res['reply']}</Message>
</Response>"""
    return Response(content=xml, media_type="application/xml")