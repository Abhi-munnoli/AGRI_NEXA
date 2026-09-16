from flask import Flask, render_template, request, redirect, url_for, session
import os
import json
import pickle
import uuid
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf

import firebase_admin
from firebase_admin import credentials

# Firestore REST API
import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google.cloud.firestore_v1.base_query import FieldFilter


# =========================================================
# APP SETUP
# =========================================================

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "yieldnext-local-secret-change-me"
)

SERVICE_KEY = os.path.join(BASE, "serviceAccountKey.json")

if not os.path.exists(SERVICE_KEY):
    raise FileNotFoundError(
        "serviceAccountKey.json was not found in the project folder."
    )


# =========================================================
# FIREBASE / FIRESTORE REST CLIENT
# =========================================================
# IMPORTANT:
# google-cloud-firestore 2.28.1 does NOT accept transport="rest"
# in Client(...). The Firestore Python client is gRPC based.
#
# Your machine was receiving:
#   400 Invalid database id %28default%29
#
# Therefore this version talks to the official Firestore REST API
# directly using the service-account OAuth token. This completely
# bypasses the failing gRPC path.
#
# Official Firestore REST API:
# https://firestore.googleapis.com/v1/
# =========================================================

if not firebase_admin._apps:
    firebase_admin.initialize_app(
        credentials.Certificate(SERVICE_KEY)
    )

firebase_app = firebase_admin.get_app()

EXPECTED_PROJECT = "crops-aa277"

if firebase_app.project_id != EXPECTED_PROJECT:
    raise RuntimeError(
        "Wrong Firebase project in serviceAccountKey.json. "
        f"Found: {firebase_app.project_id}; expected: {EXPECTED_PROJECT}"
    )


class FirestoreREST:
    """Small Firestore REST client used by YieldNext."""

    def __init__(self, project_id, service_key):
        self.project_id = project_id
        # Firestore uses "(default)" when the standard/default database is used.
        # You can override this with FIRESTORE_DATABASE_ID if your project uses
        # a named Firestore database.
        self.database_id = os.environ.get(
            "FIRESTORE_DATABASE_ID",
            "(default)"
        )
        self.base_url = (
            "https://firestore.googleapis.com/v1/"
            f"projects/{project_id}/databases/"
            f"{self.database_id}/documents"
        )

        self.credentials = service_account.Credentials.from_service_account_file(
            service_key,
            scopes=[
                "https://www.googleapis.com/auth/datastore",
                "https://www.googleapis.com/auth/cloud-platform",
            ],
        )

    def _headers(self):
        if not self.credentials.valid:
            self.credentials.refresh(Request())

        return {
            "Authorization": f"Bearer {self.credentials.token}",
            "Content-Type": "application/json",
        }

    def _request(self, method, url, **kwargs):
        headers = self._headers()
        response = requests.request(
            method,
            url,
            headers=headers,
            timeout=30,
            **kwargs,
        )

        if not response.ok:
            try:
                details = response.json()
            except Exception:
                details = response.text

            raise RuntimeError(
                f"Firestore REST API error {response.status_code}: "
                f"{details}"
            )

        if not response.content:
            return {}

        return response.json()

    @staticmethod
    def _encode_value(value):
        if value is None:
            return {"nullValue": None}

        if isinstance(value, bool):
            return {"booleanValue": value}

        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)

            return {
                "timestampValue": value.astimezone(
                    timezone.utc
                ).isoformat().replace("+00:00", "Z")
            }

        if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            return {"integerValue": str(int(value))}

        if isinstance(value, (np.floating, float)):
            return {"doubleValue": float(value)}

        if isinstance(value, dict):
            return {
                "mapValue": {
                    "fields": {
                        str(k): FirestoreREST._encode_value(v)
                        for k, v in value.items()
                    }
                }
            }

        if isinstance(value, (list, tuple)):
            return {
                "arrayValue": {
                    "values": [
                        FirestoreREST._encode_value(v)
                        for v in value
                    ]
                }
            }

        return {"stringValue": str(value)}

    @staticmethod
    def _decode_value(value):
        if "nullValue" in value:
            return None

        if "booleanValue" in value:
            return value["booleanValue"]

        if "integerValue" in value:
            return int(value["integerValue"])

        if "doubleValue" in value:
            return float(value["doubleValue"])

        if "stringValue" in value:
            return value["stringValue"]

        if "timestampValue" in value:
            raw = value["timestampValue"]
            try:
                return datetime.fromisoformat(
                    raw.replace("Z", "+00:00")
                )
            except Exception:
                return raw

        if "mapValue" in value:
            return {
                k: FirestoreREST._decode_value(v)
                for k, v in value.get("mapValue", {}).get(
                    "fields", {}
                ).items()
            }

        if "arrayValue" in value:
            return [
                FirestoreREST._decode_value(v)
                for v in value.get("arrayValue", {}).get(
                    "values", []
                )
            ]

        if "geoPointValue" in value:
            return value["geoPointValue"]

        if "referenceValue" in value:
            return value["referenceValue"]

        return None

    @classmethod
    def _encode_fields(cls, data):
        return {
            str(key): cls._encode_value(value)
            for key, value in data.items()
        }

    @classmethod
    def _decode_document(cls, document):
        return {
            key: cls._decode_value(value)
            for key, value in document.get("fields", {}).items()
        }

    @staticmethod
    def _document_id(document_name):
        return document_name.rstrip("/").split("/")[-1]

    def collection(self, name):
        return FirestoreRESTCollection(self, name)

    def _create_document(self, collection, data):
        # Firestore createDocument endpoint.
        url = f"{self.base_url}/{collection}"
        return self._request(
            "POST",
            url,
            params={"documentId": uuid.uuid4().hex},
            json={"fields": self._encode_fields(data)},
        )

    def _patch_document(self, collection, document_id, data):
        url = (
            f"{self.base_url}/{collection}/"
            f"{document_id}"
        )

        return self._request(
            "PATCH",
            url,
            json={
                "fields": self._encode_fields(data)
            },
        )

    def _run_query(self, collection, filters=None):
        parent = self.base_url

        structured = {
            "from": [
                {
                    "collectionId": collection
                }
            ]
        }

        if filters:
            filter_items = []

            for field_path, operator, value in filters:
                filter_items.append(
                    {
                        "fieldFilter": {
                            "field": {
                                "fieldPath": field_path
                            },
                            "op": operator,
                            "value": self._encode_value(value),
                        }
                    }
                )

            if len(filter_items) == 1:
                structured["where"] = filter_items[0]
            else:
                structured["where"] = {
                    "compositeFilter": {
                        "op": "AND",
                        "filters": filter_items,
                    }
                }

        url = f"{parent}:runQuery"

        response = self._request(
            "POST",
            url,
            json={
                "structuredQuery": structured
            },
        )

        # requests returns the JSON response. For RunQuery, the
        # service can return a sequence; support both a list and
        # a single object.
        rows = response if isinstance(response, list) else [response]

        results = []

        for row in rows:
            document = row.get("document")

            if not document:
                continue

            results.append(
                FirestoreRESTDocumentSnapshot(
                    self._document_id(document["name"]),
                    self._decode_document(document),
                )
            )

        return results


class FirestoreRESTCollection:
    def __init__(self, client, name):
        self.client = client
        self.name = name

    def add(self, data):
        document = self.client._create_document(
            self.name,
            data,
        )

        doc_id = self.client._document_id(
            document["name"]
        )

        return (
            FirestoreRESTDocumentReference(
                self.client,
                self.name,
                doc_id,
            ),
            document.get("createTime"),
        )

    def document(self, document_id):
        return FirestoreRESTDocumentReference(
            self.client,
            self.name,
            document_id,
        )

    def where(self, field_path=None, op_string=None, value=None, *, filter=None):
        if filter is not None:
            field_path = getattr(
                filter,
                "field_path",
                getattr(filter, "_field_path", None),
            )
            op_string = getattr(
                filter,
                "op_string",
                getattr(filter, "_op_string", None),
            )
            value = getattr(
                filter,
                "value",
                getattr(filter, "_value", None),
            )

        if field_path is None:
            raise ValueError("Firestore where() requires a field.")

        # Support the operators used by this application.
        operator_map = {
            "==": "EQUAL",
            "<": "LESS_THAN",
            "<=": "LESS_THAN_OR_EQUAL",
            ">": "GREATER_THAN",
            ">=": "GREATER_THAN_OR_EQUAL",
            "!=": "NOT_EQUAL",
            "array_contains": "ARRAY_CONTAINS",
            "in": "IN",
            "not-in": "NOT_IN",
            "array_contains_any": "ARRAY_CONTAINS_ANY",
        }

        if op_string not in operator_map:
            raise ValueError(
                f"Unsupported Firestore operator: {op_string}"
            )

        return FirestoreRESTQuery(
            self.client,
            self.name,
            [
                (
                    field_path,
                    operator_map[op_string],
                    value,
                )
            ],
        )

    def stream(self):
        return iter(
            self.client._run_query(
                self.name,
                [],
            )
        )


class FirestoreRESTQuery:
    def __init__(self, client, collection, filters):
        self.client = client
        self.collection = collection
        self.filters = filters

    def where(self, field_path=None, op_string=None, value=None, *, filter=None):
        return FirestoreRESTCollection(
            self.client,
            self.collection,
        ).where(
            field_path,
            op_string,
            value,
            filter=filter,
        )

    def stream(self):
        return iter(
            self.client._run_query(
                self.collection,
                self.filters,
            )
        )


class FirestoreRESTDocumentReference:
    def __init__(self, client, collection, document_id):
        self.client = client
        self.collection = collection
        self.document_id = document_id

    def set(self, data):
        return self.client._patch_document(
            self.collection,
            self.document_id,
            data,
        )


class FirestoreRESTDocumentSnapshot:
    def __init__(self, document_id, data):
        self.id = document_id
        self._data = data

    def to_dict(self):
        return dict(self._data)


db = FirestoreREST(
    EXPECTED_PROJECT,
    SERVICE_KEY,
)

print("========================================")
print("YIELDNEXT SMART FARMING APPLICATION")
print("========================================")
print("Firebase project   :", EXPECTED_PROJECT)
print("Firestore database : (default)")
print("Firestore transport: REST API")
print("Firestore client   : READY")
print("========================================")


# =========================================================
# TRANSLATIONS
# =========================================================

TRANSLATIONS = {
    "en": {
        "login":"Login","app_name":"YieldNext","register":"Register","dashboard":"Dashboard",
        "email":"Email","password":"Password","name":"Name","logout":"Logout","welcome":"Welcome",
        "farm_details":"Farm Details","prediction_result":"Prediction Result","crop":"Crop",
        "soil_type":"Soil Type","select_crop":"Select Crop","select_soil":"Select Soil","rainfall":"Rainfall",
        "temperature":"Temperature","humidity":"Humidity","soil_moisture":"Soil Moisture","soil_ph":"Soil pH",
        "area":"Area","acre":"acre","submit_prediction":"Submit Prediction","back_dashboard":"Back to Dashboard",
        "predicted_yield":"Predicted Yield","production":"Production","expected_profit":"Expected Profit",
        "drought_risk":"Drought Risk","pest_risk":"Pest Risk","overall_risk":"Overall Risk",
        "fertilizer":"Fertilizer","irrigation":"Irrigation","crop_advice":"Crop Advice",
        "new_prediction":"New Prediction","view_history":"View History",
    },
    "kn": {
        "login":"ಲಾಗಿನ್","app_name":"YieldNext","register":"ನೋಂದಣಿ","dashboard":"ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
        "email":"ಇಮೇಲ್","password":"ಪಾಸ್‌ವರ್ಡ್","name":"ಹೆಸರು","logout":"ಲಾಗ್‌ಔಟ್","welcome":"ಸ್ವಾಗತ",
        "farm_details":"ಹೊಲದ ವಿವರಗಳು","prediction_result":"ಇಳುವರಿ ಮುನ್ಸೂಚನೆ ಫಲಿತಾಂಶ","crop":"ಬೆಳೆ",
        "soil_type":"ಮಣ್ಣಿನ ವಿಧ","select_crop":"ಬೆಳೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ","select_soil":"ಮಣ್ಣಿನ ವಿಧವನ್ನು ಆಯ್ಕೆಮಾಡಿ",
        "rainfall":"ಮಳೆ ಪ್ರಮಾಣ","temperature":"ತಾಪಮಾನ","humidity":"ಆರ್ದ್ರತೆ","soil_moisture":"ಮಣ್ಣಿನ ತೇವಾಂಶ",
        "soil_ph":"ಮಣ್ಣಿನ pH","area":"ವಿಸ್ತೀರ್ಣ","acre":"ಎಕರೆ","submit_prediction":"ಮುನ್ಸೂಚನೆ ಪಡೆಯಿರಿ",
        "back_dashboard":"ಡ್ಯಾಶ್‌ಬೋರ್ಡ್‌ಗೆ ಹಿಂತಿರುಗಿ","predicted_yield":"ಅಂದಾಜು ಇಳುವರಿ","production":"ಒಟ್ಟು ಉತ್ಪಾದನೆ",
        "expected_profit":"ಅಂದಾಜು ಲಾಭ","drought_risk":"ಬರ ಅಪಾಯ","pest_risk":"ಕೀಟ ಅಪಾಯ","overall_risk":"ಒಟ್ಟು ಅಪಾಯ",
        "fertilizer":"ರಸಗೊಬ್ಬರ","irrigation":"ನೀರಾವರಿ","crop_advice":"ಬೆಳೆ ಸಲಹೆ",
        "new_prediction":"ಹೊಸ ಮುನ್ಸೂಚನೆ","view_history":"ಇತಿಹಾಸ ವೀಕ್ಷಿಸಿ",
    },
    "hi": {
        "login":"लॉगिन","app_name":"YieldNext","register":"पंजीकरण","dashboard":"डैशबोर्ड",
        "email":"ईमेल","password":"पासवर्ड","name":"नाम","logout":"लॉगआउट","welcome":"स्वागत है",
        "farm_details":"खेत का विवरण","prediction_result":"उपज पूर्वानुमान परिणाम","crop":"फसल",
        "soil_type":"मिट्टी का प्रकार","select_crop":"फसल चुनें","select_soil":"मिट्टी चुनें","rainfall":"वर्षा",
        "temperature":"तापमान","humidity":"आर्द्रता","soil_moisture":"मिट्टी की नमी","soil_ph":"मिट्टी का pH",
        "area":"क्षेत्रफल","acre":"एकड़","submit_prediction":"पूर्वानुमान प्राप्त करें",
        "back_dashboard":"डैशबोर्ड पर वापस जाएँ","predicted_yield":"अनुमानित उपज","production":"कुल उत्पादन",
        "expected_profit":"अनुमानित लाभ","drought_risk":"सूखा जोखिम","pest_risk":"कीट जोखिम","overall_risk":"कुल जोखिम",
        "fertilizer":"उर्वरक","irrigation":"सिंचाई","crop_advice":"फसल सलाह",
        "new_prediction":"नई भविष्यवाणी","view_history":"इतिहास देखें",
    },
}


def translate(key, default=None):
    lang = session.get("language", "en")

    if lang not in TRANSLATIONS:
        lang = "en"

    return TRANSLATIONS[lang].get(
        key,
        TRANSLATIONS["en"].get(
            key,
            default if default is not None else key,
        ),
    )


@app.context_processor
def inject_translation_helper():
    return {"t": translate}


# =========================================================
# MODEL LOADING
# =========================================================

yield_model_path = os.path.join(BASE, "model.pkl")
model = None

if os.path.exists(yield_model_path):
    try:
        with open(yield_model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print("Warning: could not load model.pkl:", e)


disease_model_path = os.path.join(BASE, "disease_model.keras")
disease_classes_path = os.path.join(BASE, "disease_classes.json")

disease_model = None
classes = []

if os.path.exists(disease_model_path):
    try:
        disease_model = tf.keras.models.load_model(
            disease_model_path
        )
    except Exception as e:
        print("Warning: could not load disease_model.keras:", e)

if os.path.exists(disease_classes_path):
    try:
        with open(
            disease_classes_path,
            "r",
            encoding="utf-8",
        ) as f:
            classes = json.load(f)

        if not isinstance(classes, list):
            raise ValueError("disease_classes.json must contain a JSON list.")

        print(f"Disease classes loaded: {len(classes)}")

    except Exception as e:
        print("Warning: could not load disease_classes.json:", e)


# =========================================================
# DISEASE MODEL / CLASS COUNT CHECK
# =========================================================

if disease_model is not None and classes:
    try:
        model_class_count = int(disease_model.output_shape[-1])
        json_class_count = len(classes)

        print("\n========================================")
        print("DISEASE MODEL CHECK")
        print("========================================")
        print("Model output classes :", model_class_count)
        print("JSON classes         :", json_class_count)

        if model_class_count != json_class_count:
            raise RuntimeError(
                "disease_model.keras and disease_classes.json do not match: "
                f"model={model_class_count}, json={json_class_count}"
            )

        print("Disease model check  : PASSED")
        print("========================================")
    except Exception as e:
        print("Disease model check failed:", e)

# =========================================================
# PROJECT DATA
# =========================================================

UPLOAD = os.path.join(
    BASE,
    "static",
    "uploads",
)

os.makedirs(UPLOAD, exist_ok=True)

WATER = {
    "Maize": "Medium",
    "Jowar": "Low",
    "Sugarcane": "High",
    "Bajra": "Low",
    "Paddy": "High",
}

ADVICE = {
    "Maize": "Monitor soil moisture during critical growth stages and avoid waterlogging.",
    "Jowar": "Maintain moisture during critical growth stages.",
    "Sugarcane": "Maintain regular irrigation and balanced crop management.",
    "Bajra": "Avoid unnecessary irrigation.",
    "Paddy": "Monitor field moisture and excess water.",
}

DISEASE = {
    "Apple___Apple_scab": (
        "Apple Scab",
        "A fungal disease that commonly causes olive, brown or dark lesions on apple leaves and fruit.",
        "Remove affected plant material where appropriate, maintain orchard sanitation and follow local disease-management guidance.",
    ),
    "Apple___Black_rot": (
        "Apple Black Rot",
        "A fungal disease that can cause dark leaf spots, fruit rot and cankers on apple plants.",
        "Remove diseased plant material, maintain sanitation and follow locally recommended management practices.",
    ),
    "Apple___Cedar_apple_rust": (
        "Apple Cedar Apple Rust",
        "A fungal disease that produces yellow-orange leaf lesions and may affect apple fruit and foliage.",
        "Monitor symptoms, maintain good orchard sanitation and follow local agricultural recommendations.",
    ),
    "Apple___healthy": (
        "Healthy Apple Leaf",
        "The model detected a healthy apple leaf pattern.",
        "Continue regular monitoring and maintain good orchard management.",
    ),
    "Blueberry___healthy": (
        "Healthy Blueberry Leaf",
        "The model detected a healthy blueberry leaf pattern.",
        "Continue regular monitoring and maintain good crop management.",
    ),
    "Cherry_(including_sour)___Powdery_mildew": (
        "Cherry Powdery Mildew",
        "A fungal disease that can produce a white powdery growth on leaves and young plant tissue.",
        "Improve airflow, avoid excessive humidity around foliage and follow local disease-management guidance.",
    ),
    "Cherry_(including_sour)___healthy": (
        "Healthy Cherry Leaf",
        "The model detected a healthy cherry leaf pattern.",
        "Continue regular monitoring and good orchard management.",
    ),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": (
        "Corn Cercospora Leaf Spot / Gray Leaf Spot",
        "A fungal leaf disease that can produce gray or elongated lesions on maize leaves.",
        "Monitor the crop, maintain good field management and follow locally recommended disease-control practices.",
    ),
    "Corn_(maize)___Common_rust_": (
        "Corn Common Rust",
        "A fungal disease that can produce reddish-brown rust-like pustules on maize leaves.",
        "Monitor rust symptoms and follow local agricultural recommendations for disease management.",
    ),
    "Corn_(maize)___Northern_Leaf_Blight": (
        "Corn Northern Leaf Blight",
        "A fungal disease that can produce long gray-green to brown lesions on maize leaves.",
        "Monitor affected leaves, maintain crop hygiene and follow local disease-management guidance.",
    ),
    "Corn_(maize)___healthy": (
        "Healthy Corn Leaf",
        "The model detected a healthy maize leaf pattern.",
        "Continue regular monitoring and maintain good crop management.",
    ),
    "Grape___Black_rot": (
        "Grape Black Rot",
        "A fungal disease that can cause brown leaf lesions and dark fruit rot in grapevines.",
        "Remove affected material where appropriate, improve sanitation and follow local grape disease-management guidance.",
    ),
    "Grape___Esca_(Black_Measles)": (
        "Grape Esca / Black Measles",
        "A grapevine disease complex that can cause leaf discoloration, spotting and fruit symptoms.",
        "Monitor affected vines and consult local agricultural guidance for appropriate disease management.",
    ),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": (
        "Grape Leaf Blight / Isariopsis Leaf Spot",
        "A fungal leaf disease that can produce dark or brown leaf spots and blight symptoms.",
        "Maintain vineyard sanitation, monitor foliage and follow local disease-management recommendations.",
    ),
    "Grape___healthy": (
        "Healthy Grape Leaf",
        "The model detected a healthy grape leaf pattern.",
        "Continue regular monitoring and maintain good vineyard management.",
    ),
    "Orange___Haunglongbing_(Citrus_greening)": (
        "Citrus Greening / Huanglongbing",
        "A serious citrus disease associated with yellowing and uneven leaf coloration and decline of citrus plants.",
        "Use certified healthy planting material, monitor for disease and follow local citrus-management guidance.",
    ),
    "Peach___Bacterial_spot": (
        "Peach Bacterial Spot",
        "A bacterial disease that can cause dark spots on peach leaves and fruit.",
        "Maintain good orchard sanitation, monitor symptoms and follow locally recommended bacterial disease management.",
    ),
    "Peach___healthy": (
        "Healthy Peach Leaf",
        "The model detected a healthy peach leaf pattern.",
        "Continue regular monitoring and maintain good orchard management.",
    ),
    "Pepper,_bell___Bacterial_spot": (
        "Bell Pepper Bacterial Spot",
        "A bacterial disease that can cause dark leaf lesions and spots on pepper plants.",
        "Maintain field sanitation, avoid unnecessary leaf wetness and follow local agricultural recommendations.",
    ),
    "Pepper,_bell___healthy": (
        "Healthy Bell Pepper Leaf",
        "The model detected a healthy bell pepper leaf pattern.",
        "Continue regular monitoring and maintain good crop management.",
    ),
    "Potato___Early_blight": (
        "Potato Early Blight",
        "A fungal disease that commonly produces dark target-like lesions on potato leaves.",
        "Remove severely affected material where appropriate, maintain crop hygiene and follow local recommendations.",
    ),
    "Potato___Late_blight": (
        "Potato Late Blight",
        "A serious disease that can rapidly damage potato foliage under favorable cool and wet conditions.",
        "Monitor frequently, avoid prolonged foliage wetness and follow local late-blight management guidance.",
    ),
    "Potato___healthy": (
        "Healthy Potato Leaf",
        "The model detected a healthy potato leaf pattern.",
        "Continue regular monitoring and maintain good crop management.",
    ),
    "Raspberry___healthy": (
        "Healthy Raspberry Leaf",
        "The model detected a healthy raspberry leaf pattern.",
        "Continue regular monitoring and maintain good crop management.",
    ),
    "Soybean___healthy": (
        "Healthy Soybean Leaf",
        "The model detected a healthy soybean leaf pattern.",
        "Continue regular monitoring and maintain good crop management.",
    ),
    "Squash___Powdery_mildew": (
        "Squash Powdery Mildew",
        "A fungal disease that produces a characteristic white powdery growth on leaves.",
        "Improve airflow, manage humidity and follow local disease-management recommendations.",
    ),
    "Strawberry___Leaf_scorch": (
        "Strawberry Leaf Scorch",
        "A disease that can cause reddish-brown or scorched-looking areas on strawberry leaves.",
        "Remove severely affected leaves where appropriate, maintain sanitation and follow local recommendations.",
    ),
    "Strawberry___healthy": (
        "Healthy Strawberry Leaf",
        "The model detected a healthy strawberry leaf pattern.",
        "Continue regular monitoring and maintain good crop management.",
    ),
    "Tomato___Bacterial_spot": (
        "Tomato Bacterial Spot",
        "A bacterial disease that can produce small dark spots on tomato leaves and fruit.",
        "Maintain sanitation, avoid unnecessary foliage wetness and follow local agricultural guidance.",
    ),
    "Tomato___Early_blight": (
        "Tomato Early Blight",
        "A fungal disease that commonly causes dark concentric or target-like leaf lesions.",
        "Remove affected plant material where appropriate, maintain sanitation and follow local disease-management guidance.",
    ),
    "Tomato___Late_blight": (
        "Tomato Late Blight",
        "A serious disease that can cause rapidly expanding dark lesions, especially under cool and wet conditions.",
        "Monitor frequently, reduce prolonged foliage wetness and follow local disease-management recommendations.",
    ),
    "Tomato___Leaf_Mold": (
        "Tomato Leaf Mold",
        "A fungal disease that commonly causes pale or yellow upper-leaf areas with mold growth on the underside.",
        "Improve ventilation, reduce excess humidity and follow local disease-management guidance.",
    ),
    "Tomato___Septoria_leaf_spot": (
        "Tomato Septoria Leaf Spot",
        "A fungal leaf disease that produces numerous small spots, often with darker margins.",
        "Remove affected leaves where appropriate, maintain sanitation and avoid unnecessary foliage wetness.",
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": (
        "Tomato Two-Spotted Spider Mite",
        "A mite pest that can cause stippling, yellowing and loss of leaf vigor.",
        "Inspect the undersides of leaves, manage plant stress and follow local integrated pest-management guidance.",
    ),
    "Tomato___Target_Spot": (
        "Tomato Target Spot",
        "A fungal disease that can cause circular brown lesions with target-like patterns on leaves and fruit.",
        "Improve field sanitation, monitor foliage and follow local disease-management recommendations.",
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
        "Tomato Yellow Leaf Curl Virus",
        "A viral disease that can cause leaf curling, yellowing and reduced plant growth.",
        "Monitor and manage insect vectors according to local integrated pest-management guidance and remove severely affected plants where appropriate.",
    ),
    "Tomato___Tomato_mosaic_virus": (
        "Tomato Mosaic Virus",
        "A viral disease that can cause mosaic patterns, leaf distortion and reduced plant vigor.",
        "Maintain sanitation, avoid mechanical spread and use healthy planting material according to local recommendations.",
    ),
    "Tomato___healthy": (
        "Healthy Tomato Leaf",
        "The model detected a healthy tomato leaf pattern.",
        "Continue regular monitoring and maintain good crop management.",
    ),
}


def get_disease_info(class_name):
    """Return information for every class in disease_classes.json."""
    if class_name in DISEASE:
        return DISEASE[class_name]

    # Safe fallback for a future class added to the model.
    clean_name = (
        str(class_name)
        .replace("___", " - ")
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )

    if "healthy" in str(class_name).lower():
        return (
            clean_name.title(),
            "The model detected a healthy crop-leaf pattern.",
            "Continue regular monitoring and good crop management.",
        )

    return (
        clean_name.title(),
        "The AI model identified this disease or pest class, but a detailed profile is not yet available.",
        "Monitor the crop and consult a local agricultural expert before applying treatment.",
    )


# =========================================================
# MULTILINGUAL DISEASE RESULTS
# =========================================================
DISEASE_TRANSLATIONS = {
    "Tomato___Early_blight": {
        "en": ("Tomato Early Blight", "A fungal disease that commonly causes dark concentric or target-like leaf lesions.", "Remove affected plant material where appropriate, maintain sanitation and follow local disease-management guidance."),
        "kn": ("ಟೊಮೇಟೊ ಆರಂಭಿಕ ಅಂಗಮಾರಿ", "ಎಲೆಗಳಲ್ಲಿ ಗಾಢವಾದ ವೃತ್ತಾಕಾರದ ಅಥವಾ ಗುರಿಯಂತಹ ಕಲೆಗಳನ್ನು ಉಂಟುಮಾಡುವ ಶಿಲೀಂಧ್ರ ರೋಗ.", "ಬಾಧಿತ ಸಸ್ಯ ಭಾಗಗಳನ್ನು ಸೂಕ್ತವಾಗಿ ತೆಗೆದುಹಾಕಿ, ಸ್ವಚ್ಛತೆಯನ್ನು ಕಾಪಾಡಿ ಮತ್ತು ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆಯನ್ನು ಅನುಸರಿಸಿ."),
        "hi": ("टमाटर अर्ली ब्लाइट", "यह एक फफूंद रोग है जो पत्तियों पर गहरे गोलाकार या लक्ष्य जैसे धब्बे पैदा करता है।", "प्रभावित पौधों के हिस्सों को उचित तरीके से हटाएँ, स्वच्छता बनाए रखें और स्थानीय कृषि सलाह का पालन करें।"),
    },
    "Tomato___Late_blight": {
        "en": ("Tomato Late Blight", "A serious disease that can cause rapidly expanding dark lesions, especially under cool and wet conditions.", "Monitor frequently, reduce prolonged foliage wetness and follow local disease-management recommendations."),
        "kn": ("ಟೊಮೇಟೊ ತಡ ಅಂಗಮಾರಿ", "ತಂಪಾದ ಮತ್ತು ತೇವಾಂಶ ಹೆಚ್ಚಿರುವ ಪರಿಸ್ಥಿತಿಯಲ್ಲಿ ವೇಗವಾಗಿ ಹರಡುವ ಗಂಭೀರ ರೋಗ.", "ಬೆಳೆಯನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ, ಎಲೆಗಳು ದೀರ್ಘಕಾಲ ತೇವವಾಗುವುದನ್ನು ಕಡಿಮೆ ಮಾಡಿ ಮತ್ತು ಸ್ಥಳೀಯ ಕೃಷಿ ಸಲಹೆಯನ್ನು ಅನುಸರಿಸಿ."),
        "hi": ("टमाटर लेट ब्लाइट", "यह एक गंभीर रोग है जो ठंडे और नम मौसम में तेजी से फैल सकता है।", "फसल की नियमित निगरानी करें, पत्तियों पर लंबे समय तक नमी न रहने दें और स्थानीय कृषि सलाह का पालन करें।"),
    },
    "Potato___Early_blight": {
        "en": ("Potato Early Blight", "A fungal disease that commonly produces dark target-like lesions on potato leaves.", "Remove severely affected material where appropriate, maintain crop hygiene and follow local recommendations."),
        "kn": ("ಆಲೂಗಡ್ಡೆ ಆರಂಭಿಕ ಅಂಗಮಾರಿ", "ಆಲೂಗಡ್ಡೆ ಎಲೆಗಳ ಮೇಲೆ ಗಾಢವಾದ ಗುರಿಯಂತಹ ಕಲೆಗಳನ್ನು ಉಂಟುಮಾಡುವ ಶಿಲೀಂಧ್ರ ರೋಗ.", "ಬಾಧಿತ ಭಾಗಗಳನ್ನು ಸೂಕ್ತವಾಗಿ ತೆಗೆದುಹಾಕಿ, ಬೆಳೆ ಸ್ವಚ್ಛತೆಯನ್ನು ಕಾಪಾಡಿ ಮತ್ತು ಸ್ಥಳೀಯ ಸಲಹೆಯನ್ನು ಅನುಸರಿಸಿ."),
        "hi": ("आलू अर्ली ब्लाइट", "यह एक फफूंद रोग है जो आलू की पत्तियों पर गहरे लक्ष्य जैसे धब्बे पैदा करता है।", "प्रभावित भागों को उचित तरीके से हटाएँ, खेत की स्वच्छता बनाए रखें और स्थानीय सलाह का पालन करें।"),
    },
    "Potato___Late_blight": {
        "en": ("Potato Late Blight", "A serious disease that can rapidly damage potato foliage under cool and wet conditions.", "Monitor frequently, avoid prolonged foliage wetness and follow local disease-management guidance."),
        "kn": ("ಆಲೂಗಡ್ಡೆ ತಡ ಅಂಗಮಾರಿ", "ತಂಪಾದ ಮತ್ತು ತೇವಾಂಶ ಹೆಚ್ಚಿರುವ ಪರಿಸ್ಥಿತಿಯಲ್ಲಿ ಆಲೂಗಡ್ಡೆ ಎಲೆಗಳಿಗೆ ವೇಗವಾಗಿ ಹಾನಿ ಮಾಡುವ ಗಂಭೀರ ರೋಗ.", "ಬೆಳೆಯನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ, ಎಲೆಗಳು ದೀರ್ಘಕಾಲ ತೇವವಾಗುವುದನ್ನು ತಪ್ಪಿಸಿ ಮತ್ತು ಸ್ಥಳೀಯ ಕೃಷಿ ಸಲಹೆಯನ್ನು ಅನುಸರಿಸಿ."),
        "hi": ("आलू लेट ब्लाइट", "यह एक गंभीर रोग है जो ठंडे और नम मौसम में आलू की पत्तियों को तेजी से नुकसान पहुँचा सकता है।", "फसल की नियमित निगरानी करें, पत्तियों पर लंबे समय तक नमी न रहने दें और स्थानीय कृषि सलाह का पालन करें।"),
    },
}

def get_multilingual_disease_info(class_name, language="en"):
    """
    Return the SAME model class in the selected language.

    Important:
    - class_name is always the original class returned by the model.
    - Changing language never runs the model again.
    - Explicit translations are used where available.
    - Every remaining class gets a safe localized fallback instead of
      silently changing the prediction.
    """
    language = language if language in ("en", "kn", "hi") else "en"

    if class_name in DISEASE_TRANSLATIONS:
        translated = DISEASE_TRANSLATIONS[class_name].get(language)
        if translated:
            return translated

    info = get_disease_info(class_name)
    english_name = info[0]

    # Localized display names for all 38 supported model classes.
    # The model class itself remains unchanged internally.
    localized_names = {
        "Apple___Apple_scab": {
            "kn": "ಆಪಲ್ ಸ್ಕ್ಯಾಬ್",
            "hi": "सेब स्कैब",
        },
        "Apple___Black_rot": {
            "kn": "ಆಪಲ್ ಕಪ್ಪು ಕೊಳೆ ರೋಗ",
            "hi": "सेब ब्लैक रॉट",
        },
        "Apple___Cedar_apple_rust": {
            "kn": "ಆಪಲ್ ಸೀಡರ್ ಆಪಲ್ ರಸ್ಟ್",
            "hi": "सेब सीडर एप्पल रस्ट",
        },
        "Apple___healthy": {
            "kn": "ಆರೋಗ್ಯಕರ ಆಪಲ್ ಎಲೆ",
            "hi": "स्वस्थ सेब का पत्ता",
        },
        "Blueberry___healthy": {
            "kn": "ಆರೋಗ್ಯಕರ ಬ್ಲೂಬೆರಿ ಎಲೆ",
            "hi": "स्वस्थ ब्लूबेरी का पत्ता",
        },
        "Cherry_(including_sour)___Powdery_mildew": {
            "kn": "ಚೆರ್ರಿ ಪುಡಿ ಶಿಲೀಂಧ್ರ ರೋಗ",
            "hi": "चेरी पाउडरी मिल्ड्यू",
        },
        "Cherry_(including_sour)___healthy": {
            "kn": "ಆರೋಗ್ಯಕರ ಚೆರ್ರಿ ಎಲೆ",
            "hi": "स्वस्थ चेरी का पत्ता",
        },
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
            "kn": "ಜೋಳ ಸೆರ್ಕೋಸ್ಪೋರಾ / ಗ್ರೇ ಲೀಫ್ ಸ್ಪಾಟ್",
            "hi": "मक्का सेर्कोस्पोरा / ग्रे लीफ स्पॉट",
        },
        "Corn_(maize)___Common_rust_": {
            "kn": "ಜೋಳ ಸಾಮಾನ್ಯ ರಸ್ಟ್",
            "hi": "मक्का कॉमन रस्ट",
        },
        "Corn_(maize)___Northern_Leaf_Blight": {
            "kn": "ಜೋಳ ಉತ್ತರ ಎಲೆ ಅಂಗಮಾರಿ",
            "hi": "मक्का नॉर्दर्न लीफ ब्लाइट",
        },
        "Corn_(maize)___healthy": {
            "kn": "ಆರೋಗ್ಯಕರ ಜೋಳದ ಎಲೆ",
            "hi": "स्वस्थ मक्का का पत्ता",
        },
        "Grape___Black_rot": {
            "kn": "ದ್ರಾಕ್ಷಿ ಕಪ್ಪು ಕೊಳೆ ರೋಗ",
            "hi": "अंगूर ब्लैक रॉट",
        },
        "Grape___Esca_(Black_Measles)": {
            "kn": "ದ್ರಾಕ್ಷಿ ಎಸ್ಕಾ / ಬ್ಲ್ಯಾಕ್ ಮೀಸಲ್ಸ್",
            "hi": "अंगूर एस्का / ब्लैक मीज़ल्स",
        },
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
            "kn": "ದ್ರಾಕ್ಷಿ ಎಲೆ ಅಂಗಮಾರಿ / ಇಸಾರಿಯೋಪ್ಸಿಸ್ ಕಲೆ",
            "hi": "अंगूर लीफ ब्लाइट / इसारियोप्सिस लीफ स्पॉट",
        },
        "Grape___healthy": {
            "kn": "ಆರೋಗ್ಯಕರ ದ್ರಾಕ್ಷಿ ಎಲೆ",
            "hi": "स्वस्थ अंगूर का पत्ता",
        },
        "Orange___Haunglongbing_(Citrus_greening)": {
            "kn": "ಸಿಟ್ರಸ್ ಗ್ರೀನಿಂಗ್ / ಹುವಾಂಗ್‌ಲಾಂಗ್‌ಬಿಂಗ್",
            "hi": "सिट्रस ग्रीनिंग / हुआंगलोंगबिंग",
        },
        "Peach___Bacterial_spot": {
            "kn": "ಪೀಚ್ ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಸ್ಪಾಟ್",
            "hi": "आड़ू बैक्टीरियल स्पॉट",
        },
        "Peach___healthy": {
            "kn": "ಆರೋಗ್ಯಕರ ಪೀಚ್ ಎಲೆ",
            "hi": "स्वस्थ आड़ू का पत्ता",
        },
        "Pepper,_bell___Bacterial_spot": {
            "kn": "ಬೆಲ್ ಪೆಪ್ಪರ್ ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಸ್ಪಾಟ್",
            "hi": "शिमला मिर्च बैक्टीरियल स्पॉट",
        },
        "Pepper,_bell___healthy": {
            "kn": "ಆರೋಗ್ಯಕರ ಬೆಲ್ ಪೆಪ್ಪರ್ ಎಲೆ",
            "hi": "स्वस्थ शिमला मिर्च का पत्ता",
        },
        "Potato___Early_blight": {
            "kn": "ಆಲೂಗಡ್ಡೆ ಆರಂಭಿಕ ಅಂಗಮಾರಿ",
            "hi": "आलू अर्ली ब्लाइट",
        },
        "Potato___Late_blight": {
            "kn": "ಆಲೂಗಡ್ಡೆ ತಡ ಅಂಗಮಾರಿ",
            "hi": "आलू लेट ब्लाइट",
        },
        "Potato___healthy": {
            "kn": "ಆರೋಗ್ಯಕರ ಆಲೂಗಡ್ಡೆ ಎಲೆ",
            "hi": "स्वस्थ आलू का पत्ता",
        },
        "Raspberry___healthy": {
            "kn": "ಆರೋಗ್ಯಕರ ರಾಸ್ಪ್ಬೆರಿ ಎಲೆ",
            "hi": "स्वस्थ रास्पबेरी का पत्ता",
        },
        "Soybean___healthy": {
            "kn": "ಆರೋಗ್ಯಕರ ಸೋಯಾಬೀನ್ ಎಲೆ",
            "hi": "स्वस्थ सोयाबीन का पत्ता",
        },
        "Squash___Powdery_mildew": {
            "kn": "ಸ್ಕ್ವಾಶ್ ಪುಡಿ ಶಿಲೀಂಧ್ರ ರೋಗ",
            "hi": "स्क्वैश पाउडरी मिल्ड्यू",
        },
        "Strawberry___Leaf_scorch": {
            "kn": "ಸ್ಟ್ರಾಬೆರಿ ಎಲೆ ಸುಡುವಿಕೆ",
            "hi": "स्ट्रॉबेरी लीफ स्कॉर्च",
        },
        "Strawberry___healthy": {
            "kn": "ಆರೋಗ್ಯಕರ ಸ್ಟ್ರಾಬೆರಿ ಎಲೆ",
            "hi": "स्वस्थ स्ट्रॉबेरी का पत्ता",
        },
        "Tomato___Bacterial_spot": {
            "kn": "ಟೊಮೇಟೊ ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಸ್ಪಾಟ್",
            "hi": "टमाटर बैक्टीरियल स्पॉट",
        },
        "Tomato___Early_blight": {
            "kn": "ಟೊಮೇಟೊ ಆರಂಭಿಕ ಅಂಗಮಾರಿ",
            "hi": "टमाटर अर्ली ब्लाइट",
        },
        "Tomato___Late_blight": {
            "kn": "ಟೊಮೇಟೊ ತಡ ಅಂಗಮಾರಿ",
            "hi": "टमाटर लेट ब्लाइट",
        },
        "Tomato___Leaf_Mold": {
            "kn": "ಟೊಮೇಟೊ ಎಲೆ ಮೋಲ್ಡ್",
            "hi": "टमाटर लीफ मोल्ड",
        },
        "Tomato___Septoria_leaf_spot": {
            "kn": "ಟೊಮೇಟೊ ಸೆಪ್ಟೋರಿಯಾ ಎಲೆ ಕಲೆ",
            "hi": "टमाटर सेप्टोरिया लीफ स्पॉट",
        },
        "Tomato___Spider_mites Two-spotted_spider_mite": {
            "kn": "ಟೊಮೇಟೊ ಎರಡು-ಕಲೆ ಸ್ಪೈಡರ್ ಮೈಟ್",
            "hi": "टमाटर टू-स्पॉटेड स्पाइडर माइट",
        },
        "Tomato___Target_Spot": {
            "kn": "ಟೊಮೇಟೊ ಟಾರ್ಗೆಟ್ ಸ್ಪಾಟ್",
            "hi": "टमाटर टार्गेट स्पॉट",
        },
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
            "kn": "ಟೊಮೇಟೊ ಹಳದಿ ಎಲೆ ಮುದುರಿಕೆ ವೈರಸ್",
            "hi": "टमाटर येलो लीफ कर्ल वायरस",
        },
        "Tomato___Tomato_mosaic_virus": {
            "kn": "ಟೊಮೇಟೊ ಮೊಸಾಯಿಕ್ ವೈರಸ್",
            "hi": "टमाटर मोज़ेक वायरस",
        },
        "Tomato___healthy": {
            "kn": "ಆರೋಗ್ಯಕರ ಟೊಮೇಟೊ ಎಲೆ",
            "hi": "स्वस्थ टमाटर का पत्ता",
        },
    }

    display_name = localized_names.get(class_name, {}).get(language, english_name)

    if language == "kn":
        return (
            display_name,
            f"AI ಮಾದರಿಯು {display_name} ಅನ್ನು ಪತ್ತೆಹಚ್ಚಿದೆ. ವಿವರವಾದ ದೃಢೀಕರಣಕ್ಕಾಗಿ ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
            "ಬೆಳೆಯನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ ಮತ್ತು ಚಿಕಿತ್ಸೆ ನೀಡುವ ಮೊದಲು ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆಯನ್ನು ಪಡೆಯಿರಿ.",
        )

    if language == "hi":
        return (
            display_name,
            f"AI मॉडल ने {display_name} की पहचान की है। विस्तृत पुष्टि के लिए स्थानीय कृषि विशेषज्ञ से सलाह लें।",
            "फसल की नियमित निगरानी करें और उपचार से पहले स्थानीय कृषि विशेषज्ञ की सलाह लें।",
        )

    return info

# =========================================================
# HELPERS
# =========================================================

def logged_in():
    return "user" in session


def utc_now():
    return datetime.now(timezone.utc)


def error_page(title, message, error):
    try:
        return render_template(
            "error.html",
            title=title,
            message=message,
            error=error,
        )
    except Exception:
        return (
            f"<h2>{title}</h2>"
            f"<p>{message}</p>"
            f"<pre>{error}</pre>"
        )


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def home():
    session.setdefault("language", "en")
    return render_template("login.html")


@app.route("/set-language/<lang>", methods=["GET"])
def set_language(lang):
    if lang not in TRANSLATIONS:
        lang = "en"

    session["language"] = lang

    if session.get("disease_model_class"):
        session["disease_language"] = lang

    next_url = request.args.get("next")
    if next_url and next_url.startswith("/"):
        if next_url.startswith("/disease-result"):
            separator = "&" if "?" in next_url else "?"
            return redirect(f"{next_url}{separator}lang={lang}")
        return redirect(next_url)

    referrer = request.referrer
    if referrer and referrer.startswith("/"):
        return redirect(referrer)

    return redirect(url_for("home"))


@app.route("/register", methods=["GET", "POST"])
def register():
    # GET: open the registration page.
    if request.method == "GET":
        session.setdefault("language", "en")
        return render_template("register.html")

    # POST: process the registration form.
    try:
        name = request.form.get(
            "name",
            "",
        ).strip()

        email = request.form.get(
            "email",
            "",
        ).strip().lower()

        password = request.form.get(
            "password",
            "",
        )

        if not name or not email or not password:
            return (
                '<script>'
                'alert("Please fill all registration fields.");'
                'history.back();'
                '</script>'
            )

        users = db.collection("users").where(
            filter=FieldFilter(
                "email",
                "==",
                email,
            )
        ).stream()

        if any(True for _ in users):
            return (
                '<script>'
                'alert("Email already registered");'
                'location="/";'
                '</script>'
            )

        db.collection("users").add(
            {
                "name": name,
                "email": email,
                "password": password,
                "created_at": utc_now(),
            }
        )

        return (
            '<script>'
            'alert("Registration successful");'
            'location="/";'
            '</script>'
        )

    except Exception as e:
        print("REGISTRATION ERROR:", repr(e))

        return error_page(
            "Registration Error",
            "Database registration request failed.",
            str(e),
        )


@app.route("/login", methods=["POST"])
def login():
    try:
        email = request.form.get(
            "email",
            "",
        ).strip().lower()

        password = request.form.get(
            "password",
            "",
        )

        if not email or not password:
            return (
                '<script>'
                'alert("Please enter email and password.");'
                'history.back();'
                '</script>'
            )

        users = db.collection("users").where(
            filter=FieldFilter(
                "email",
                "==",
                email,
            )
        ).stream()

        for user_doc in users:
            data = user_doc.to_dict()

            if data.get("password") == password:
                selected_language = session.get("language", "en")

                session.clear()
                session["user"] = email
                session["name"] = data.get(
                    "name",
                    "User",
                )
                session["language"] = (
                    selected_language
                    if selected_language in ("en", "kn", "hi")
                    else "en"
                )

                return redirect(
                    url_for("dashboard")
                )

        return (
            '<script>'
            'alert("Invalid email or password");'
            'location="/";'
            '</script>'
        )

    except Exception as e:
        print("LOGIN ERROR:", repr(e))

        return error_page(
            "Login Error",
            "Database login request failed.",
            str(e),
        )


@app.route("/dashboard")
def dashboard():
    if not logged_in():
        return redirect(url_for("home"))

    return render_template(
        "dashboard.html",
        name=session.get(
            "name",
            "User",
        ),
    )


@app.route("/farm-details")
def farm_details():
    if not logged_in():
        return redirect(url_for("home"))

    return render_template(
        "farm_details.html"
    )


@app.route("/predict", methods=["POST"])
def predict():
    if not logged_in():
        return redirect(url_for("home"))

    try:
        vals = {
            "crop": request.form["crop"],
            "soil_type": request.form["soil_type"],
        }

        nums = [
            "rainfall",
            "temperature",
            "humidity",
            "soil_moisture",
            "soil_ph",
            "area",
        ]

        vals.update(
            {
                k: float(request.form[k])
                for k in nums
            }
        )

        if model is None:
            return (
                "<h2>Prediction model not found</h2>"
                "<p>Please train the yield model first.</p>"
            )

        x = pd.DataFrame(
            [
                {
                    "Crop": vals["crop"],
                    "Soil_Type": vals["soil_type"],
                    "Rainfall_mm": vals["rainfall"],
                    "Temperature_C": vals["temperature"],
                    "Humidity_percent": vals["humidity"],
                    "Soil_Moisture_percent": vals["soil_moisture"],
                    "Soil_pH": vals["soil_ph"],
                    "Area_acre": vals["area"],
                }
            ]
        )

        try:
            pred = float(
                model.predict(x)[0]
            )
        except Exception as model_error:
            return (
                "<h2>Prediction model input mismatch</h2>"
                f"<p>{model_error}</p>"
                "<p>Retrain model.pkl using the current input fields.</p>"
            )

        pred = round(pred, 3)

        rain = vals["rainfall"]
        temp = vals["temperature"]
        hum = vals["humidity"]
        sm = vals["soil_moisture"]

        # =====================================================
        # RISK CALCULATION
        # =====================================================
        # These are rule-based agricultural risk indicators.
        # They are intentionally calculated from the same values
        # entered by the farmer, independently of the yield model.

        drought_score = 0

        # Rainfall
        if rain < 250:
            drought_score += 3
        elif rain < 450:
            drought_score += 2
        elif rain < 600:
            drought_score += 1

        # Soil moisture
        if sm < 15:
            drought_score += 3
        elif sm < 25:
            drought_score += 2
        elif sm < 35:
            drought_score += 1

        # High temperature increases water stress
        if temp >= 38:
            drought_score += 2
        elif temp >= 35:
            drought_score += 1

        # Low humidity increases evaporation/water stress
        if hum < 30:
            drought_score += 2
        elif hum < 45:
            drought_score += 1

        if drought_score >= 5:
            drought = "High"
        elif drought_score >= 2:
            drought = "Medium"
        else:
            drought = "Low"

        # Pest/fungal pressure
        pest_score = 0

        if hum >= 85:
            pest_score += 3
        elif hum >= 75:
            pest_score += 2
        elif hum >= 65:
            pest_score += 1

        if rain >= 800:
            pest_score += 2
        elif rain >= 650:
            pest_score += 1

        if 20 <= temp <= 32:
            pest_score += 1

        if pest_score >= 5:
            pest = "High"
        elif pest_score >= 2:
            pest = "Medium"
        else:
            pest = "Low"

        # Overall risk uses the stronger of drought and pest
        # risk, while also considering combined moderate risks.
        if drought == "High" or pest == "High":
            overall = "High"
        elif drought == "Medium" and pest == "Medium":
            overall = "High"
        elif drought == "Medium" or pest == "Medium":
            overall = "Medium"
        else:
            overall = "Low"

        crop = vals["crop"]
        need = WATER.get(
            crop,
            "Medium",
        )

        irrigation = (
            f"{crop} has {need} water requirement. "
        )

        if rain < 450:
            irrigation += (
                "Rainfall is low; irrigate according "
                "to soil moisture and crop growth stage."
            )
        elif rain > 750:
            irrigation += (
                "Rainfall is high; avoid unnecessary "
                "irrigation and monitor waterlogging."
            )
        else:
            irrigation += (
                "Use irrigation according to soil moisture "
                "and crop growth stage."
            )

        # Prototype profit estimate.
        # Market price values below are ₹ per quintal.
        # Production is in tonnes, so convert ₹/quintal -> ₹/tonne.
        base_price_per_quintal = {
            "Maize": 2200,
            "Jowar": 3000,
            "Sugarcane": 350,
            "Bajra": 2500,
            "Paddy": 2300,
        }.get(
            crop,
            2200,
        )

        QUINTAL_TO_TON = 10
        base_price_per_ton = (
            base_price_per_quintal
            * QUINTAL_TO_TON
        )

        ACRES_TO_HECTARES = 0.40468564224
        area_acres = max(vals["area"], 0)
        area_hectares = area_acres * ACRES_TO_HECTARES

        estimated_total_yield = (
            max(pred, 0)
            * area_hectares
        )

        estimated_revenue = (
            estimated_total_yield
            * base_price_per_ton
        )

        estimated_cost_per_acre = {
            "Maize": 18000,
            "Jowar": 12000,
            "Sugarcane": 45000,
            "Bajra": 11000,
            "Paddy": 20000,
        }.get(
            crop,
            15000,
        )

        estimated_cost = (
            estimated_cost_per_acre
            * max(vals["area"], 0)
        )

        base_profit = (


            estimated_revenue - estimated_cost


        )



        # One-year future profit estimate.


        future_years = 1


        future_price_growth = {


            "Maize": 0.08,


            "Jowar": 0.08,


            "Sugarcane": 0.06,


            "Bajra": 0.08,


            "Paddy": 0.07,


        }.get(crop, 0.07)



        future_price_per_ton = (


            base_price_per_ton


            * ((1 + future_price_growth) ** future_years)


        )



        future_revenue = (


            estimated_total_yield


            * future_price_per_ton


        )



        future_cost_inflation = 0.05


        future_cost = (


            estimated_cost


            * ((1 + future_cost_inflation) ** future_years)


        )



        future_base_profit = (


            future_revenue - future_cost


        )



        # Risk makes future profit more conservative.


        risk_profit_factor = {


            "High": 0.55,


            "Medium": 0.80,


            "Low": 1.00,


        }.get(overall, 0.80)



        estimated_profit = (


            future_base_profit * risk_profit_factor


        )

        fertilizer_recommendation = (
            f"For {crop}, use soil-test-based fertilizer "
            "management. Avoid excessive fertilizer application "
            "and follow local agricultural recommendations."
        )

        db.collection("predictions").add(
            {
                "user": session["user"],
                "crop": crop,
                "soil_type": vals["soil_type"],
                "rainfall": vals["rainfall"],
                "temperature": vals["temperature"],
                "humidity": vals["humidity"],
                "soil_moisture": vals["soil_moisture"],
                "soil_ph": vals["soil_ph"],
                "area_acre": vals["area"],
                "predicted_yield": pred,
                "estimated_total_yield": round(
                    estimated_total_yield,
                    3,
                ),
                "estimated_revenue": round(
                    estimated_revenue,
                    2,
                ),
                "price_unit": "INR per tonne",
                "estimated_cost": round(
                    estimated_cost,
                    2,
                ),
                "estimated_profit": round(
                    estimated_profit,
                    2,
                ),
                "expected_profit": round(
                    estimated_profit,
                    2,
                ),
                "expected_future_profit": round(
                    estimated_profit,
                    2,
                ),
                "future_price_per_ton": round(
                    future_price_per_ton,
                    2,
                ),
                "future_revenue": round(
                    future_revenue,
                    2,
                ),
                "future_cost": round(
                    future_cost,
                    2,
                ),
                "base_profit": round(
                    base_profit,
                    2,
                ),
                "risk_profit_factor": risk_profit_factor,
                "profit_risk_adjusted": True,
                "drought_risk": drought,
                "pest_risk": pest,
                "overall_risk": overall,
                "created_at": utc_now(),
            }
        )

        return render_template(
            "recommendations.html",
            crop=crop,
            prediction=pred,
            area=vals["area"],
            drought_risk=drought,
            pest_risk=pest,
            overall_risk=overall,
            estimated_total_yield=round(
                estimated_total_yield,
                3,
            ),
            estimated_revenue=round(
                estimated_revenue,
                2,
            ),
            estimated_cost=round(
                estimated_cost,
                2,
            ),
            estimated_profit=round(
                estimated_profit,
                2,
            ),
            production=round(
                estimated_total_yield,
                3,
            ),
            expected_profit=round(
                estimated_profit,
                2,
            ),
            expected_future_profit=round(
                estimated_profit,
                2,
            ),
            future_price_per_ton=round(
                future_price_per_ton,
                2,
            ),
            future_revenue=round(
                future_revenue,
                2,
            ),
            future_cost=round(
                future_cost,
                2,
            ),
            area_hectares=round(
                area_hectares,
                3,
            ),
            fertilizer_recommendation=fertilizer_recommendation,
            irrigation_recommendation=irrigation,
            crop_advice=ADVICE.get(
                crop,
                "Monitor the crop regularly.",
            ),
        )

    except Exception as e:
        print("PREDICTION ERROR:", repr(e))

        return (
            "<h2>Prediction error</h2>"
            f"<p>{e}</p>"
        )


@app.route("/history")
def history():
    if not logged_in():
        return redirect(url_for("home"))

    try:
        # Read the prediction collection for the logged-in user.
        # We intentionally filter in Python after reading the collection so
        # older/newer Firestore records remain compatible even if a field
        # index/query configuration changes.
        all_docs = db.collection("predictions").stream()

        records = []

        for doc in all_docs:
            data = doc.to_dict() or {}

            # Only show this user's prediction history.
            if str(data.get("user", "")).strip().lower() != str(
                session.get("user", "")
            ).strip().lower():
                continue

            # ---------------------------------------------------------
            # NORMALIZE FIELD NAMES
            # ---------------------------------------------------------
            # Current prediction route saves these names:
            # area_acre, predicted_yield, estimated_total_yield,
            # expected_future_profit, future_price_per_ton,
            # future_revenue, future_cost, drought_risk,
            # pest_risk, overall_risk.
            #
            # Older records may contain the shorter/older names.
            data["crop"] = data.get(
                "crop",
                data.get("Crop", "N/A"),
            )

            data["area_acre"] = data.get(
                "area_acre",
                data.get("area", 0),
            )

            data["predicted_yield"] = data.get(
                "predicted_yield",
                data.get("prediction", 0),
            )

            data["estimated_total_yield"] = data.get(
                "estimated_total_yield",
                data.get("production", 0),
            )

            # Expected Future Profit:
            # newest field first, then older compatible fields.
            future_profit = data.get("expected_future_profit")

            if future_profit is None:
                future_profit = data.get("expected_profit")

            if future_profit is None:
                future_profit = data.get("estimated_profit")

            if future_profit is None:
                future_profit = 0

            try:
                data["expected_future_profit"] = float(future_profit)
            except (TypeError, ValueError):
                data["expected_future_profit"] = 0.0

            # Future financial fields.
            for field in (
                "future_price_per_ton",
                "future_revenue",
                "future_cost",
                "estimated_revenue",
                "estimated_cost",
                "base_profit",
            ):
                value = data.get(field, 0)
                try:
                    data[field] = float(value or 0)
                except (TypeError, ValueError):
                    data[field] = 0.0

            # Make sure risks are always available to the template.
            data["drought_risk"] = data.get(
                "drought_risk",
                "N/A",
            )

            data["pest_risk"] = data.get(
                "pest_risk",
                "N/A",
            )

            data["overall_risk"] = data.get(
                "overall_risk",
                "N/A",
            )

            # ---------------------------------------------------------
            # DATE DISPLAY + REAL SORT KEY
            # ---------------------------------------------------------
            created = data.get("created_at")

            if isinstance(created, datetime):
                if created.tzinfo is None:
                    created = created.replace(
                        tzinfo=timezone.utc
                    )

                created_utc = created.astimezone(timezone.utc)

                data["created_at_display"] = created_utc.strftime(
                    "%d %b %Y, %I:%M %p"
                )

                # Keep a real datetime for correct newest-first sorting.
                data["_history_sort"] = created_utc

            elif created:
                data["created_at_display"] = str(created)

                try:
                    raw_created = str(created).replace(
                        "Z",
                        "+00:00",
                    )
                    parsed_created = datetime.fromisoformat(
                        raw_created
                    )

                    if parsed_created.tzinfo is None:
                        parsed_created = parsed_created.replace(
                            tzinfo=timezone.utc
                        )

                    data["_history_sort"] = parsed_created

                except Exception:
                    data["_history_sort"] = datetime.min.replace(
                        tzinfo=timezone.utc
                    )

            else:
                data["created_at_display"] = "N/A"
                data["_history_sort"] = datetime.min.replace(
                    tzinfo=timezone.utc
                )

            # Firestore document ID is useful for debugging and ensures
            # every saved prediction can be uniquely identified.
            data["_document_id"] = getattr(
                doc,
                "id",
                "",
            )

            records.append(data)

        # Newest prediction first.
        records.sort(
            key=lambda item: item.get(
                "_history_sort",
                datetime.min.replace(tzinfo=timezone.utc),
            ),
            reverse=True,
        )

        # Do not expose internal helper fields to Jinja.
        for record in records:
            record.pop("_history_sort", None)

        response = render_template(
            "history.html",
            records=records,
            current_lang=session.get(
                "language",
                "en",
            ),
        )

        # Prevent the browser/proxy from displaying an old history page.
        response = app.make_response(response)
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

        return response

    except Exception as e:
        print("HISTORY ERROR:", repr(e))

        return error_page(
            "History Error",
            "Could not load prediction history.",
            str(e),
        )


@app.route("/disease-detection")
def disease_detection():
    if not logged_in():
        return redirect(url_for("home"))

    return render_template(
        "disease_detection.html"
    )


@app.route("/analyze-disease", methods=["POST"])
def analyze():
    if not logged_in():
        return redirect(url_for("home"))

    if disease_model is None:
        return (
            "Disease model not found. "
            "Run the disease training script first."
        )

    f = request.files.get(
        "crop_image"
    )

    if not f or not f.filename:
        return "Please select an image."

    ext = f.filename.rsplit(
        ".",
        1,
    )[-1].lower()

    if ext not in {
        "jpg",
        "jpeg",
        "png",
    }:
        return (
            "Only JPG, JPEG and PNG are allowed."
        )

    name = (
        f"{uuid.uuid4().hex}.{ext}"
    )

    path = os.path.join(
        UPLOAD,
        name,
    )

    f.save(path)

    try:
        img = (
            Image.open(path)
            .convert("RGB")
            .resize((224, 224))
        )

        arr = np.expand_dims(
            np.asarray(
                img,
                dtype=np.float32,
            ),
            axis=0,
        )

        probs = disease_model.predict(
            arr,
            verbose=0,
        )[0]

        # The trained disease_model.keras already contains:
        # Rescaling(1/127.5, offset=-1)
        # Therefore the image must be passed as raw 0-255 float32 pixels.
        if not classes:
            raise RuntimeError(
                "disease_classes.json is missing or empty."
            )

        if len(probs) != len(classes):
            raise RuntimeError(
                f"Model output has {len(probs)} classes, but "
                f"disease_classes.json has {len(classes)} classes."
            )

        idx = int(np.argmax(probs))

        # Keep the exact class order produced by the training script.
        cls = classes[idx]
        conf = round(float(probs[idx]) * 100, 2)

        # Terminal debugging: useful for verifying that different leaf
        # images are actually producing different predictions.
        print("\n========================================")
        print("DISEASE PREDICTION DEBUG")
        print("========================================")
        print("Input shape       :", arr.shape)
        print("Input pixel range :", float(arr.min()), "to", float(arr.max()))
        print("Model outputs     :", len(probs))
        print("Class count       :", len(classes))
        print("Predicted index   :", idx)
        print("Predicted class   :", cls)
        print("Confidence        :", f"{conf:.2f}%")
        print("\nTop 5 predictions:")

        top_indices = np.argsort(probs)[-5:][::-1]
        for top_i in top_indices:
            top_i = int(top_i)
            print(
                f"{top_i:02d} | {classes[top_i]} | "
                f"{float(probs[top_i]) * 100:.2f}%"
            )

        print("========================================\n")

        language = request.form.get(
            "language",
            session.get("language", "en")
        )
        if language not in ("en", "kn", "hi"):
            language = "en"

        session["language"] = language
        info = get_multilingual_disease_info(cls, language)

        # Save the exact AI result in the session.
        # Language changes reuse this result and do NOT run the model again.
        session["disease_model_class"] = cls
        session["disease_image_file"] = name
        session["disease_confidence"] = conf
        session["disease_language"] = language

        # Store the original model class so language changes never alter
        # the actual AI prediction.
        db.collection(
            "disease_predictions"
        ).add(
            {
                "user": session["user"],
                "image": name,
                "disease": get_disease_info(cls)[0],
                "model_class": cls,
                "language": language,
                "confidence": conf,
                "created_at": utc_now(),
            }
        )

        return redirect(
            url_for("disease_result", lang=language)
        )

    except Exception as e:
        print(
            "DISEASE ANALYSIS ERROR:",
            repr(e),
        )

        return (
            "<h2>Disease analysis error</h2>"
            f"<p>{e}</p>"
        )



# =========================================================
# DISEASE RESULT PAGE
# =========================================================
@app.route("/disease-result", methods=["GET"])
def disease_result():
    """Display the exact saved disease prediction in the selected language."""
    if not logged_in():
        return redirect(url_for("home"))

    language = request.args.get(
        "lang",
        session.get(
            "disease_language",
            session.get("language", "en"),
        ),
    )

    if language not in ("en", "kn", "hi"):
        language = "en"

    model_class = session.get("disease_model_class")
    image_file = session.get("disease_image_file")
    confidence = session.get("disease_confidence")

    # Never run the model again when the language changes.
    if not model_class or not image_file or confidence is None:
        return redirect(url_for("disease_detection"))

    session["language"] = language
    session["disease_language"] = language

    disease_name, cause, advice = get_multilingual_disease_info(
        model_class,
        language,
    )

    return render_template(
        "disease_result.html",
        image_file=image_file,
        disease=disease_name,
        confidence=float(confidence),
        cause=cause,
        advice=advice,
        language=language,
        model_class=model_class,
    )

@app.route("/logout")
def logout():
    session.clear()
    return redirect(
        url_for("home")
    )


@app.errorhandler(404)
def page_not_found(error):
    return (
        "<h2>Page not found</h2>"
        "<p>The requested page does not exist.</p>"
    ), 404


@app.errorhandler(500)
def internal_server_error(error):
    return (
        "<h2>Internal server error</h2>"
        "<p>Check the Flask terminal for the exact error.</p>"
    ), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
