import json
import requests
import numpy as np
import os
import hashlib


# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)


class DiagnosticLLM:
    """
    Diagnostic LLM interface using Ollama.

    Generates beliefs from retrieved documents alone.
    Ground truth is NEVER passed during inference.

    Falls back to document-content evidence scoring when Ollama
    is unavailable or returns unparseable output.
    """

    def __init__(self, model_name="llama3", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.ollama_available = self._check_connection()

        if self.ollama_available:
            print(f"--- Engine: Ollama '{model_name}' at {base_url} ---")
        else:
            print(f"--- Engine: Ollama UNAVAILABLE — using evidence scoring ---")

    def _check_connection(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except (requests.RequestException, ValueError):
            return False

    def _cache_key(self, docs, hypotheses, query):
        doc_ids = "_".join(sorted([d['id'] for d in docs]))
        raw = f"{doc_ids}_{query}_{','.join(hypotheses)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _load_cache(self, key):
        path = os.path.join(CACHE_DIR, f"{key}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def _save_cache(self, key, result):
        path = os.path.join(CACHE_DIR, f"{key}.json")
        try:
            with open(path, 'w') as f:
                json.dump(result, f)
        except OSError:
            pass

    def _build_prompt(self, docs, hypotheses, query):
        context = "\n".join([f"[{d['id']}] {d['text']}" for d in docs])
        hyp_list = ", ".join(hypotheses)

        prompt = f"""You are a UAV diagnostic analyst. Based ONLY on the documents below, select the most likely diagnosis.

Documents:
{context}

Candidate diagnoses: {hyp_list}

Symptom: {query}

Respond with JSON ONLY:
{{
  "diagnosis": "<selected diagnosis or null>",
  "confidence": <0.0-1.0>,
  "evidence": "<quote from documents>",
  "document_ids": ["<supporting doc IDs>"],
  "reasoning": "<one sentence>"
}}
"""
        return prompt

    def _parse_json_response(self, text):
        text = text.strip()

        # Strip code fences
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    def _score_evidence(self, docs, hypotheses, query):
        """
        Score each hypothesis by how well its diagnostic concepts
        are supported by the SPECIFIC retrieved documents.

        Different documents → different scores → different beliefs.
        """
        # Combine all retrieved document text
        combined_text = " ".join([d['text'] for d in docs]).lower()
        combined_words = set(combined_text.split())

        # Diagnostic concept map: hypothesis → key diagnostic terms
        # Each term is weighted by how specific/diagnostic it is
        concept_map = {
            "voltage_sag": {"voltage": 0.3, "sag": 0.3, "throttle": 0.2, "battery": 0.1, "internal resistance": 0.1},
            "propeller_loose": {"propeller": 0.4, "loose": 0.3, "nut": 0.3},
            "esc_thermal": {"esc": 0.3, "thermal": 0.3, "mosfet": 0.2, "temperature": 0.1, "shutdown": 0.1},
            "pitot_ice": {"pitot": 0.4, "ice": 0.3, "airspeed": 0.2, "frozen": 0.1},
            "sensor_failure": {"sensor": 0.3, "failure": 0.2, "fault": 0.2, "malfunction": 0.2, "freeze": 0.1},
            "software_bug": {"software": 0.4, "bug": 0.3, "firmware": 0.2, "glitch": 0.1},
            "connector_corrosion": {"connector": 0.3, "corrosion": 0.3, "corroded": 0.2, "intermittent": 0.1, "ground": 0.1},
            "servo_potentiometer": {"servo": 0.3, "potentiometer": 0.4, "jitter": 0.2, "actuator": 0.1},
            "emi": {"emi": 0.3, "interference": 0.3, "radio": 0.2, "tower": 0.2},
            "vtx_interference": {"vtx": 0.3, "interference": 0.2, "video": 0.2, "feed": 0.1, "5.8ghz": 0.2},
            "power_sag": {"power": 0.3, "voltage": 0.2, "sag": 0.2, "battery": 0.1, "throttle": 0.2},
            "vibration": {"vibration": 0.3, "resonance": 0.2, "oscillation": 0.2, "imu": 0.2, "clipping": 0.1},
            "misaligned_motor": {"misaligned": 0.3, "mount": 0.2, "motor": 0.2, "yaw": 0.2, "twisted": 0.1},
            "imu_drift": {"imu": 0.3, "drift": 0.3, "attitude": 0.2, "vibration": 0.2},
            "wind": {"wind": 0.4, "gust": 0.3, "turbulence": 0.3},
            "baro_turbulence": {"barometric": 0.3, "baro": 0.3, "pressure": 0.2, "turbulence": 0.2},
            "gps_glitch": {"gps": 0.4, "glitch": 0.3, "satellite": 0.3},
            "ground_effect": {"ground": 0.3, "effect": 0.3, "hover": 0.2, "altitude": 0.2},
            "cell_imbalance": {"cell": 0.3, "imbalance": 0.4, "voltage": 0.2, "cutoff": 0.1},
            "high_resistance": {"resistance": 0.3, "internal": 0.2, "aging": 0.2, "cell": 0.1, "battery": 0.2},
            "sensor_error": {"sensor": 0.3, "error": 0.3, "wrong": 0.2, "incorrect": 0.2},
            "emi_tower": {"emi": 0.3, "compass": 0.2, "tower": 0.2, "metallic": 0.2, "variance": 0.1},
            "calibration_error": {"calibration": 0.4, "error": 0.3, "calibrated": 0.3},
            "metal_structure": {"metallic": 0.3, "structure": 0.3, "metal": 0.2, "compass": 0.2},
            "vtx_bleed": {"vtx": 0.3, "bleed": 0.3, "video": 0.2, "transmitter": 0.2},
            "antenna_failure": {"antenna": 0.3, "failure": 0.3, "signal": 0.2, "frame": 0.2},
            "gps_fault": {"gps": 0.4, "fault": 0.3, "satellite": 0.3},
            "carbon_dust_short": {"carbon": 0.2, "dust": 0.2, "short": 0.3, "reboot": 0.2, "pins": 0.1},
            "voltage_spike": {"voltage": 0.3, "spike": 0.3, "surge": 0.2, "capacitor": 0.2},
            "bec_failure": {"bec": 0.4, "servo": 0.2, "power": 0.2, "5v": 0.2},
            "structural_resonance": {"resonance": 0.3, "structural": 0.2, "vibration": 0.2, "imu": 0.2, "rpm": 0.1},
            "unbalanced_prop": {"unbalanced": 0.3, "propeller": 0.3, "prop": 0.2, "vibration": 0.2},
            "loose_bolt": {"loose": 0.3, "bolt": 0.4, "fastener": 0.3},
            "hydraulic_leak": {"hydraulic": 0.3, "leak": 0.3, "fluid": 0.2, "pressure": 0.2},
            "mechanical_jam": {"mechanical": 0.3, "jam": 0.3, "obstruction": 0.2, "stuck": 0.2},
            "sensor_fault": {"sensor": 0.3, "fault": 0.3, "failure": 0.2, "error": 0.2},
            "pyro_resistance": {"pyro": 0.3, "resistance": 0.3, "igniter": 0.2, "parachute": 0.2},
            "logic_error": {"logic": 0.3, "error": 0.3, "command": 0.2, "software": 0.2},
            "dead_battery": {"dead": 0.2, "battery": 0.3, "charge": 0.2, "low": 0.2, "power": 0.1},
            "antenna_blocking": {"antenna": 0.2, "blocking": 0.3, "carbon": 0.2, "frame": 0.2, "telemetry": 0.1},
            "range_limit": {"range": 0.3, "distance": 0.2, "limit": 0.2, "link": 0.2, "loss": 0.1},
            "radio_failure": {"radio": 0.3, "failure": 0.3, "2.4ghz": 0.2, "latency": 0.2},
            "bearing_failure": {"bearing": 0.4, "failing": 0.2, "rpm": 0.2, "motor": 0.2},
            "overcurrent": {"current": 0.3, "draw": 0.3, "thrust": 0.2, "obstruction": 0.2},
            "bad_esc": {"esc": 0.3, "bad": 0.2, "failure": 0.2, "mosfet": 0.2, "thermal": 0.1},
            "airspeed_mismatch": {"airspeed": 0.3, "mismatch": 0.3, "speed": 0.2, "difference": 0.2},
            "wind_gust": {"wind": 0.3, "gust": 0.4, "turbulence": 0.3},
            "pitot_clog": {"pitot": 0.3, "clog": 0.3, "blocked": 0.2, "tube": 0.2},
            "aggressive_pids": {"pid": 0.3, "aggressive": 0.3, "d-term": 0.2, "gain": 0.2},
            "low_voltage": {"low": 0.2, "voltage": 0.3, "cutoff": 0.2, "battery": 0.2, "power": 0.1},
            "heavy_payload": {"payload": 0.3, "heavy": 0.3, "weight": 0.2, "load": 0.2},
            "internal_resistance": {"internal": 0.3, "resistance": 0.3, "cell": 0.2, "aging": 0.2},
            "cell_failure": {"cell": 0.3, "failure": 0.3, "imbalance": 0.2, "voltage": 0.2},
            "charger_error": {"charger": 0.4, "error": 0.3, "charge": 0.3},
            "fog_scattering": {"fog": 0.3, "scattering": 0.3, "dust": 0.2, "lidar": 0.2},
            "lidar_fault": {"lidar": 0.3, "fault": 0.3, "failure": 0.2, "distance": 0.2},
            "dirty_lens": {"dirty": 0.3, "lens": 0.3, "optical": 0.2, "flow": 0.2},
            "short_circuit": {"short": 0.3, "circuit": 0.2, "capacitor": 0.2, "pdb": 0.2, "heating": 0.1},
            "overvoltage": {"overvoltage": 0.3, "voltage": 0.2, "regulator": 0.2, "5v": 0.2, "rail": 0.1},
            "bad_cap": {"capacitor": 0.3, "cap": 0.3, "heating": 0.2, "smoke": 0.2},
            "high_pids": {"pid": 0.3, "d-term": 0.3, "oscillation": 0.2, "high": 0.2},
            "low_pids": {"pid": 0.3, "low": 0.3, "gain": 0.2, "mushy": 0.2},
            "gcs_throttling": {"cpu": 0.2, "throttling": 0.3, "ground station": 0.2, "fan": 0.2, "frozen": 0.1},
            "link_failure": {"link": 0.3, "failure": 0.3, "feed": 0.2, "connection": 0.2},
            "app_crash": {"app": 0.3, "crash": 0.3, "screen": 0.2, "frozen": 0.2},
            "motor_order_wrong": {"motor": 0.2, "order": 0.3, "wrong": 0.2, "sequence": 0.2, "takeoff": 0.1},
            "reversed_prop": {"reversed": 0.3, "propeller": 0.3, "prop": 0.2, "flip": 0.2},
            "gyro_fault": {"gyro": 0.3, "fault": 0.3, "imu": 0.2, "failure": 0.2},
            "firmware_mismatch": {"firmware": 0.3, "mismatch": 0.3, "update": 0.2, "hardware": 0.1, "id": 0.1},
            "bad_cable": {"cable": 0.3, "bad": 0.2, "connector": 0.2, "wire": 0.2, "loose": 0.1},
            "eeprom_fault": {"eeprom": 0.4, "fault": 0.3, "memory": 0.3},
        }

        query_words = set(query.lower().split())

        beliefs = {}
        for hyp in hypotheses:
            terms = concept_map.get(hyp, {})
            if not terms:
                # Unknown hypothesis → tiny score
                beliefs[hyp] = 0.01
                continue

            score = 0.0
            for term, weight in terms.items():
                # Multi-word term check
                if " " in term:
                    if term in combined_text:
                        score += weight * 1.5  # Exact phrase match is strong
                else:
                    if term in combined_words:
                        score += weight * 1.0

            # Query-document overlap bonus
            query_overlap = len(query_words.intersection(combined_words))
            score += query_overlap * 0.02

            # Normalize score to [0.01, 0.99] range using sigmoid-like function
            # This ensures differentiation without any hypothesis hitting 0 or 1
            score = min(0.95, max(0.01, score))

            beliefs[hyp] = score

        # Softmax normalization to proper distribution
        # Using temperature to control sharpness
        temperature = 0.5
        keys = list(beliefs.keys())
        scores = np.array([beliefs[k] for k in keys])

        # Softmax
        exp_scores = np.exp((scores - np.max(scores)) / temperature)
        probs = exp_scores / exp_scores.sum()

        beliefs = {k: float(p) for k, p in zip(keys, probs)}

        # Shannon entropy
        prob_array = np.array(list(beliefs.values()))
        entropy = float(-np.sum(prob_array * np.log2(prob_array + 1e-12)))

        return {
            "beliefs": beliefs,
            "entropy": entropy,
            "raw_response": "[evidence-based scoring]",
            "abstain": False,
            "parsed": None,
        }

    def generate_beliefs(self, docs, hypotheses, query):
        """
        Generate diagnostic beliefs from retrieved documents.
        No ground truth is passed.
        """
        # Cache check
        ck = self._cache_key(docs, hypotheses, query)
        cached = self._load_cache(ck)
        if cached is not None:
            return cached

        # Ollama unavailable → evidence scoring
        if not self.ollama_available:
            result = self._score_evidence(docs, hypotheses, query)
            self._save_cache(ck, result)
            return result

        prompt = self._build_prompt(docs, hypotheses, query)

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 500}
                },
                timeout=60
            )
            response.raise_for_status()
            result_json = response.json()
            raw_text = result_json.get("response", "")
        except (requests.RequestException, json.JSONDecodeError) as e:
            result = self._score_evidence(docs, hypotheses, query)
            self._save_cache(ck, result)
            return result

        parsed = self._parse_json_response(raw_text)

        if parsed is None:
            result = self._score_evidence(docs, hypotheses, query)
            self._save_cache(ck, result)
            return result

        diagnosis = parsed.get("diagnosis")
        model_confidence = parsed.get("confidence", 0.0)

        # Build belief distribution
        beliefs = {h: 0.01 for h in hypotheses}

        if diagnosis is None or str(diagnosis).upper() == "ABSTAIN" or model_confidence < 0.1:
            beliefs = {h: 1.0 / len(hypotheses) for h in hypotheses}
            result = {
                "beliefs": beliefs,
                "entropy": float(np.log2(len(hypotheses))),
                "raw_response": raw_text,
                "abstain": True,
                "parsed": parsed,
            }
            self._save_cache(ck, result)
            return result

        if diagnosis in beliefs:
            beliefs[diagnosis] = max(model_confidence, 0.1)

        total = sum(beliefs.values())
        if total > 0:
            beliefs = {k: v / total for k, v in beliefs.items()}
        else:
            beliefs = {h: 1.0 / len(hypotheses) for h in hypotheses}

        prob_array = np.array(list(beliefs.values()))
        entropy = float(-np.sum(prob_array * np.log2(prob_array + 1e-12)))

        result = {
            "beliefs": beliefs,
            "entropy": entropy,
            "raw_response": raw_text,
            "abstain": False,
            "parsed": parsed,
        }
        self._save_cache(ck, result)
        return result
