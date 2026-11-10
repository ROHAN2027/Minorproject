"""
Food Vision Module for G2P2C Simulation Console
=================================================
Analyzes food images using Groq Vision API and matches detected items
against the INDB nutrition database and the local food_database.py.

Provides camera/upload integration for the Streamlit app, converting
AI-detected food items into the carbohydrate format needed by the
glucose simulation.

Usage (standalone test):
    python food_vision.py

Requirements:
    pip install rapidfuzz pillow-heif groq python-decouple
"""

import os
import re
import json
import base64
from io import BytesIO
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
from rapidfuzz import process, fuzz
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # HEIC support optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

# Vision API settings
GROQ_MODEL = "qwen/qwen3.8-27b"
FALLBACK_GROQ_MODELS = ["meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.2-11b-vision-preview"]
IMAGE_MAX_SIZE = 512
IMAGE_QUALITY = 65
MAX_COMPLETION_TOKENS = 800

# Matching thresholds
INDB_SCORE_CUTOFF = 78
LOCAL_DB_SCORE_CUTOFF = 75


# ---------------------------------------------------------------------------
# Groq Client (lazy init)
# ---------------------------------------------------------------------------
_groq_client = None


def _get_groq_client():
    """Lazily initialize and return the Groq client."""
    global _groq_client
    if _groq_client is not None:
        return _groq_client

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found! Add it to your .env file:\n"
            "GROQ_API_KEY=gsk_your_key_here"
        )

    from groq import Groq
    _groq_client = Groq(
        api_key=api_key,
        timeout=90.0,
        max_retries=2,
    )
    return _groq_client


# ---------------------------------------------------------------------------
# INDB Excel Dataset Loading
# ---------------------------------------------------------------------------
_indb_cache: Optional[Tuple[pd.DataFrame, list, dict]] = None


def load_indb_dataset(file_path: Optional[str] = None) -> Tuple[pd.DataFrame, list, dict]:
    """Load and index the INDB nutrition Excel dataset.

    Returns:
        (DataFrame, search_choices list, choice_to_idx dict)
    """
    global _indb_cache
    if _indb_cache is not None:
        return _indb_cache

    if file_path is None:
        # Auto-find in project folder
        for candidate in ("indb.xlsx", "indb.xls", "indb.csv"):
            p = os.path.join(SCRIPT_DIR, candidate)
            if os.path.exists(p):
                file_path = p
                break
        if file_path is None:
            raise FileNotFoundError(
                "INDB dataset not found. Place indb.xlsx in the G2P2C project folder."
            )

    if file_path.lower().endswith((".xlsx", ".xls")):
        excel_file = pd.ExcelFile(file_path)
        df = None
        for sheet in excel_file.sheet_names:
            temp_df = pd.read_excel(file_path, sheet_name=sheet)
            temp_df.columns = temp_df.columns.astype(str).str.strip()
            if "food_name" in temp_df.columns:
                df = temp_df
                break
        if df is None:
            df = pd.read_excel(file_path, sheet_name=0)
    else:
        df = pd.read_csv(file_path)

    df.columns = df.columns.astype(str).str.strip()
    df = df.dropna(subset=["food_name"]).reset_index(drop=True)

    search_choices: List[str] = []
    choice_to_idx: Dict[str, int] = {}

    for idx, row in df.iterrows():
        raw_name = str(row["food_name"]).strip()
        full_clean = raw_name.lower()

        search_choices.append(full_clean)
        choice_to_idx[full_clean] = idx

        # Also index bracketed aliases  e.g. "Paneer in butter sauce (Shahi Paneer)"
        bracket_match = re.match(r"^(.*?)\s*\((.*?)\)$", raw_name)
        if bracket_match:
            for alias in (bracket_match.group(1).strip().lower(),
                          bracket_match.group(2).strip().lower()):
                if alias and alias not in choice_to_idx:
                    search_choices.append(alias)
                    choice_to_idx[alias] = idx

    _indb_cache = (df, search_choices, choice_to_idx)
    return _indb_cache


def fuzzy_match_indb(
    query_name: str,
    df: pd.DataFrame,
    search_choices: list,
    choice_to_idx: dict,
    score_cutoff: int = INDB_SCORE_CUTOFF,
) -> Tuple[Optional[pd.Series], float]:
    """Fuzzy-match a food name against the INDB dataset."""
    clean_query = query_name.lower().strip()
    match = process.extractOne(
        clean_query,
        search_choices,
        scorer=fuzz.token_set_ratio,
        score_cutoff=score_cutoff,
    )
    if match:
        matched_alias, score, _ = match
        row_idx = choice_to_idx[matched_alias]
        return df.iloc[row_idx], round(score, 1)
    return None, 0.0


# ---------------------------------------------------------------------------
# Local food_database.py Matching
# ---------------------------------------------------------------------------

def fuzzy_match_local_db(
    query_name: str,
    food_list: List[str],
    score_cutoff: int = LOCAL_DB_SCORE_CUTOFF,
) -> Tuple[Optional[str], float]:
    """Fuzzy-match a food name against the local food_database.py entries."""
    if not food_list:
        return None, 0.0
    clean_query = query_name.lower().strip()
    choices_lower = [f.lower() for f in food_list]
    match = process.extractOne(
        clean_query,
        choices_lower,
        scorer=fuzz.token_set_ratio,
        score_cutoff=score_cutoff,
    )
    if match:
        matched_lower, score, idx = match
        return food_list[idx], round(score, 1)
    return None, 0.0


# ---------------------------------------------------------------------------
# Image Processing
# ---------------------------------------------------------------------------

def compress_and_encode_image(
    image_source,
    max_size: int = IMAGE_MAX_SIZE,
    quality: int = IMAGE_QUALITY,
) -> Tuple[str, dict]:
    """Compress an image and return base64 + metadata.

    Args:
        image_source: file path (str), BytesIO, or Streamlit UploadedFile
        max_size: max dimension in pixels
        quality: JPEG quality (0-100)

    Returns:
        (base64_string, {"original_kb": ..., "compressed_kb": ..., "size": (w, h)})
    """
    if isinstance(image_source, str):
        if not os.path.exists(image_source):
            raise FileNotFoundError(f"Image not found: {image_source}")
        orig_size_kb = os.path.getsize(image_source) / 1024
        img = Image.open(image_source)
    else:
        # BytesIO or Streamlit UploadedFile
        image_source.seek(0)
        raw_bytes = image_source.read()
        orig_size_kb = len(raw_bytes) / 1024
        image_source.seek(0)
        img = Image.open(image_source)

    img = ImageOps.exif_transpose(img)

    # Convert to RGB
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        alpha = img.convert("RGBA")
        bg = Image.new("RGB", alpha.size, (255, 255, 255))
        bg.paste(alpha, mask=alpha.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    compressed_bytes = buffer.getvalue()
    new_size_kb = len(compressed_bytes) / 1024

    meta = {
        "original_kb": round(orig_size_kb, 1),
        "compressed_kb": round(new_size_kb, 1),
        "size": img.size,
    }

    return base64.b64encode(compressed_bytes).decode("utf-8"), meta


# ---------------------------------------------------------------------------
# Groq Vision API Call
# ---------------------------------------------------------------------------

VISION_PROMPT = """
You are a strict computer vision food analyzer calibrated for the Indian Nutrient Databank (INDB).
STRICT RULES:
1. ONLY report food or beverage items physically visible in the image. Do NOT guess hidden side dishes.
2. Name foods using standard English or Indian names (e.g., 'Hot tea (Garam Chai)', 'Samosa', 'Dal Tadka', 'Roti', 'Idli').
3. Provide 'visual_location' for every item to prove it exists in the image.
4. Return ONLY a valid JSON object matching this exact schema:
{
  "cuisine": "String",
  "items": [
    {
      "name": "Standard name of the dish/drink",
      "visual_location": "where it is located in the photo",
      "count": 1,
      "serving_unit": "piece | tea cup | tall glass | katori | bowl | plate",
      "estimated_grams_per_unit": 150,
      "kcal_per_100g": 120
    }
  ]
}
"""


def call_groq_vision(base64_image: str) -> dict:
    """Send image to Groq Vision API and return parsed JSON response."""
    client = _get_groq_client()

    models_to_try = [GROQ_MODEL] + [m for m in FALLBACK_GROQ_MODELS if m != GROQ_MODEL]
    last_err = None

    for model_name in models_to_try:
        try:
            # Build API kwargs
            api_kwargs = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                "model": model_name,
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
                "max_completion_tokens": MAX_COMPLETION_TOKENS,
            }
            # qwen3.8-27b supports reasoning_effort="none" for fast 1-2s response
            if "qwen" in model_name:
                api_kwargs["reasoning_effort"] = "none"

            chat_completion = client.chat.completions.create(**api_kwargs)
            result_text = chat_completion.choices[0].message.content.strip()
            raw_json = json.loads(result_text)

            token_info = {
                "input": chat_completion.usage.prompt_tokens,
                "output": chat_completion.usage.completion_tokens,
                "total": chat_completion.usage.total_tokens,
                "model": model_name,
            }

            return {"raw": raw_json, "tokens": token_info}
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"All Groq Vision models failed. Last error: {last_err}")


# ---------------------------------------------------------------------------
# Standardize AI Output with INDB + Local DB
# ---------------------------------------------------------------------------

def standardize_with_databases(
    raw_data: dict,
    indb_df: Optional[pd.DataFrame] = None,
    indb_choices: Optional[list] = None,
    indb_idx_map: Optional[dict] = None,
    local_food_list: Optional[List[str]] = None,
    local_get_food_info=None,
) -> dict:
    """Match AI-detected items against INDB and local food_database.py.

    Returns a standardized result with carb data and match confidence.
    """
    standardized_items = []
    grand_total_carbs = 0.0
    grand_total_calories = 0.0

    for item in raw_data.get("items", []):
        detected_name = item.get("name", "unknown").strip()
        count = float(item.get("count", 1))
        serving_unit = item.get("serving_unit", "piece")
        estimated_grams = float(item.get("estimated_grams_per_unit", 100.0))
        ai_kcal_per_100g = float(item.get("kcal_per_100g", 150.0))

        matched_source = "ai_estimate"
        match_score = 0.0
        carbs_per_unit = 0.0
        total_item_carbs = 0.0
        total_item_calories = 0.0
        matched_food_name = detected_name
        local_db_food_name = None  # for linking to food_database.py entry
        protein_g = None
        fat_g = None

        # --- Strategy 1: Try local food_database.py first (best for simulation) ---
        if local_food_list and local_get_food_info:
            local_match, local_score = fuzzy_match_local_db(
                detected_name, local_food_list
            )
            if local_match:
                f_info = local_get_food_info(local_match)
                if f_info:
                    local_db_food_name = local_match
                    carbs_per_unit = float(f_info["carbs_per_unit"])
                    total_item_carbs = round(carbs_per_unit * count, 1)
                    serving_unit = f_info["default_unit"]
                    matched_food_name = local_match
                    match_score = local_score
                    matched_source = "local_db"

        # --- Strategy 2: Try INDB Excel (richer nutrition data) ---
        if indb_df is not None and indb_choices and indb_idx_map:
            indb_row, indb_score = fuzzy_match_indb(
                detected_name, indb_df, indb_choices, indb_idx_map
            )
            if indb_row is not None:
                # Use INDB if no local match, or if INDB score is significantly higher
                use_indb = (matched_source == "ai_estimate") or (indb_score > match_score + 10)

                if use_indb:
                    excel_food_name = str(indb_row["food_name"])
                    kcal_per_100g = float(indb_row.get("energy_kcal", 0.0))

                    # Try unit serving data first
                    unit_kcal = indb_row.get("unit_serving_energy_kcal")
                    if pd.notna(unit_kcal) and float(unit_kcal) > 0:
                        cal_per_unit = round(float(unit_kcal), 2)
                        std_grams = round(
                            (cal_per_unit / kcal_per_100g) * 100.0, 1
                        ) if kcal_per_100g > 0 else 100.0

                        carb_per_unit_val = float(indb_row.get("unit_serving_carb_g", 0.0))
                        protein_g = round(float(indb_row.get("unit_serving_protein_g", 0.0)) * count, 2)
                        fat_g = round(float(indb_row.get("unit_serving_fat_g", 0.0)) * count, 2)
                    else:
                        std_grams = estimated_grams
                        cal_per_unit = round((std_grams / 100.0) * kcal_per_100g, 2)
                        factor = (std_grams / 100.0) * count
                        carb_per_unit_val = float(indb_row.get("carb_g", 0.0)) * (std_grams / 100.0)
                        protein_g = round(float(indb_row.get("protein_g", 0.0)) * factor, 2)
                        fat_g = round(float(indb_row.get("fat_g", 0.0)) * factor, 2)

                    carbs_per_unit = round(carb_per_unit_val, 1)
                    total_item_carbs = round(carbs_per_unit * count, 1)
                    total_item_calories = round(cal_per_unit * count, 2)
                    matched_food_name = excel_food_name
                    match_score = indb_score
                    matched_source = "indb"

                elif matched_source == "local_db":
                    # Keep local_db match but supplement with INDB macros if available
                    if pd.notna(indb_row.get("unit_serving_protein_g")):
                        protein_g = round(float(indb_row.get("unit_serving_protein_g", 0.0)) * count, 2)
                        fat_g = round(float(indb_row.get("unit_serving_fat_g", 0.0)) * count, 2)

        # --- Strategy 3: AI estimate fallback ---
        if matched_source == "ai_estimate":
            # Use the AI's own rough estimates
            carbs_per_unit = round(estimated_grams * 0.25, 1)  # ~25% carbs assumption
            total_item_carbs = round(carbs_per_unit * count, 1)
            total_item_calories = round(
                (estimated_grams / 100.0) * ai_kcal_per_100g * count, 2
            )

        grand_total_carbs += total_item_carbs
        grand_total_calories += total_item_calories

        standardized_items.append({
            "detected_name": detected_name,
            "matched_name": matched_food_name,
            "local_db_name": local_db_food_name,
            "visual_location": item.get("visual_location"),
            "count": count,
            "serving_unit": serving_unit,
            "carbs_per_unit": carbs_per_unit,
            "total_carbs": total_item_carbs,
            "total_calories": total_item_calories,
            "macros": {
                "protein_g": protein_g,
                "carb_g": total_item_carbs,
                "fat_g": fat_g,
            },
            "match_source": matched_source,
            "match_score": match_score,
        })

    return {
        "cuisine": raw_data.get("cuisine", "Indian"),
        "items": standardized_items,
        "total_carbs": round(grand_total_carbs, 1),
        "total_calories": round(grand_total_calories, 1),
    }


# ---------------------------------------------------------------------------
# High-Level API: Analyze a Food Image
# ---------------------------------------------------------------------------

def analyze_food_image(
    image_source,
    local_food_list: Optional[List[str]] = None,
    local_get_food_info=None,
) -> dict:
    """Full pipeline: compress image → Groq Vision → standardize with databases.

    Args:
        image_source: file path, BytesIO, or Streamlit UploadedFile
        local_food_list: list from food_database.get_food_list()
        local_get_food_info: function food_database.get_food_info

    Returns:
        dict with keys: items, total_carbs, total_calories, image_meta, tokens
    """
    # 1. Compress and encode
    base64_image, image_meta = compress_and_encode_image(image_source)

    # 2. Call Groq Vision
    api_result = call_groq_vision(base64_image)

    # 3. Load INDB if available
    try:
        indb_df, indb_choices, indb_idx_map = load_indb_dataset()
    except FileNotFoundError:
        indb_df, indb_choices, indb_idx_map = None, None, None

    # 4. Standardize with both databases
    standardized = standardize_with_databases(
        raw_data=api_result["raw"],
        indb_df=indb_df,
        indb_choices=indb_choices,
        indb_idx_map=indb_idx_map,
        local_food_list=local_food_list,
        local_get_food_info=local_get_food_info,
    )

    standardized["image_meta"] = image_meta
    standardized["tokens"] = api_result["tokens"]

    return standardized


# ---------------------------------------------------------------------------
# Convert to Simulation Format
# ---------------------------------------------------------------------------

def detection_to_meal_items(detection_result: dict) -> List[dict]:
    """Convert detection result into a list ready for the simulation UI.

    Each item has:
        food_name, carbs_per_unit, quantity, unit, total_carbs,
        confidence_badge, match_source, local_db_name
    """
    items = []
    for det in detection_result.get("items", []):
        source = det["match_source"]
        score = det["match_score"]

        if source == "local_db":
            badge = f"🟢 Local DB ({score}%)"
        elif source == "indb":
            badge = f"🔵 INDB ({score}%)"
        else:
            badge = "🔴 AI Estimate"

        items.append({
            "food_name": det["matched_name"],
            "detected_as": det["detected_name"],
            "carbs_per_unit": det["carbs_per_unit"],
            "quantity": det["count"],
            "unit": det["serving_unit"],
            "total_carbs": det["total_carbs"],
            "confidence_badge": badge,
            "match_source": source,
            "local_db_name": det.get("local_db_name"),
            "macros": det.get("macros", {}),
        })

    return items


# ---------------------------------------------------------------------------
# Standalone Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        # Look for any image in current directory
        for f in os.listdir("."):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".heic")):
                test_image = f
                break
        else:
            print("Usage: python food_vision.py <image_path>")
            sys.exit(1)

    print(f"Analyzing: {test_image}")
    result = analyze_food_image(test_image)
    print(json.dumps(result, indent=2, default=str))

