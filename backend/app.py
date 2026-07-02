"""
Flask API — Thin wrapper over the multi-agent orchestrator.
Endpoints:
  POST /api/rank    — Run the full pipeline, return ranked results
  GET  /api/status  — Return current pipeline stage (for frontend progress)
  GET  /api/health  — Health check
"""

import os
import time
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import json
import uuid
import logging
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_ROLE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
else:
    supabase_client = None

# ------------------------------------------------------------------
# Orchestrator setup
# ------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import orchestrator
from agents.resume_analyzer import analyze_resume

app = Flask(__name__)
CORS(app)

# Shared progress state — one slot (single-user demo; extend with session IDs for multi-user)
_pipeline_progress = {'stage': 'idle'}
_progress_lock = threading.Lock()

print('Loading FAISS index and metadata...')
try:
    orchestrator.load_resources()
    print('Resources loaded successfully.')
except Exception as e:
    print(f'WARNING: Failed to load resources: {e}')
    print('The /api/rank endpoint will return an error until resources are available.')


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'timestamp': time.time()})

@app.route('/api/analyze_resume', methods=['POST'])
def handle_analyze_resume():
    data = request.get_json(silent=True)
    app.logger.warning(f"DEBUG: mimetype={request.mimetype}, data={data}")
    app.logger.warning(f"DEBUG: form={request.form}, files={request.files}")
    if data is not None:
        resume_text = data.get('resume', '').strip()
        api_key = data.get('api_key', '').strip()
    else:
        resume_text = request.form.get('resume', '').strip()
        api_key = request.form.get('api_key', '').strip()
        
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.lower().endswith('.pdf'):
                try:
                    import pypdf
                    reader = pypdf.PdfReader(file)
                    extracted = [page.extract_text() for page in reader.pages if page.extract_text()]
                    file_text = "\n".join(extracted)
                    resume_text = (resume_text + "\n\n" + file_text).strip()
                except Exception as e:
                    app.logger.exception("Failed to parse PDF")
            elif file.filename.lower().endswith('.txt'):
                try:
                    file_text = file.read().decode('utf-8')
                    resume_text = (resume_text + "\n\n" + file_text).strip()
                except Exception as e:
                    app.logger.exception("Failed to read text file")
    
    if not resume_text:
        return jsonify({'error': 'No resume provided'}), 400
        
    try:
        result = analyze_resume(resume_text, api_key=api_key)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Failed to analyze resume: {e}'}), 500


@app.route('/api/status', methods=['GET'])
def status():
    with _progress_lock:
        return jsonify(_pipeline_progress.copy())


@app.route('/api/rank', methods=['POST'])
def rank_candidates():
    data = request.json or {}
    jd_text = data.get('jd', '').strip()
    batch_id = data.get('batch_id')

    if not jd_text:
        return jsonify({'error': 'No JD provided'}), 400

    with _progress_lock:
        _pipeline_progress['stage'] = 'starting'

    try:
        result = orchestrator.run(
            jd_text  = jd_text,
            progress = _pipeline_progress,
            batch_id = batch_id,
        )
        with _progress_lock:
            _pipeline_progress['stage'] = 'idle'
        return jsonify(result)

    except RuntimeError as e:
        with _progress_lock:
            _pipeline_progress['stage'] = 'error'
        return jsonify({'error': str(e)}), 503
    except Exception as e:  # noqa: BLE001
        with _progress_lock:
            _pipeline_progress['stage'] = 'error'
        return jsonify({'error': f'Pipeline error: {e}'}), 500

@app.route('/api/upload_candidates', methods=['POST'])
def upload_candidates():
    if not supabase_client:
        return jsonify({'error': 'Supabase not configured in backend'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    try:
        raw_bytes = file.read().decode('utf-8')
        if file.filename.endswith('.csv'):
            import csv
            import io
            reader = csv.DictReader(io.StringIO(raw_bytes))
            data = []
            for row in reader:
                skills = []
                skills_raw = row.get('skills', '[]')
                try:
                    parsed = json.loads(skills_raw)
                    skills = [{'name': s, 'proficiency': 'expert'} for s in parsed]
                except Exception:
                    skills = [{'name': s.strip(), 'proficiency': 'expert'} for s in skills_raw.split(',') if s.strip()]

                cand = {
                    'candidate_id': row.get('candidate_id', str(uuid.uuid4())),
                    'profile': {
                        'current_title': row.get('current_role', ''),
                        'years_of_experience': int(row.get('experience_years', 0)) if str(row.get('experience_years', '')).isdigit() else 0,
                        'summary': row.get('summary', '')
                    },
                    'skills': skills,
                    'career_history': []
                }
                data.append(cand)
        else:
            try:
                data = json.loads(raw_bytes)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                # Try JSONL
                data = [json.loads(line) for line in raw_bytes.strip().split('\n') if line.strip()]
            
        from agents.stage2_prefilter import _get_model, build_candidate_text
        model = _get_model()
        
        batch_id = request.form.get('batch_id', str(uuid.uuid4()))
        inserted_count = 0
        
        for cand in data:
            candidate_id = cand.get('candidate_id', str(uuid.uuid4()))
            cand['candidate_id'] = candidate_id
            cand['batch_id'] = batch_id
            
            text = build_candidate_text(cand)
            vec = model.encode([text], convert_to_numpy=True)[0].tolist()
            
            supabase_client.table('candidates_vector').upsert({
                'candidate_id': candidate_id,
                'metadata': cand,
                'embedding': vec
            }, on_conflict='candidate_id').execute()
            inserted_count += 1
            
        return jsonify({
            'message': f'Successfully uploaded and embedded {inserted_count} candidates',
            'batch_id': batch_id
        })
    except Exception as e:
        app.logger.exception("Failed to upload candidates")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_attachment', methods=['POST'])
def upload_attachment():
    if not supabase_client:
        return jsonify({'error': 'Supabase not configured'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    file_path = request.form.get('file_path')
    if not file_path:
        return jsonify({'error': 'No file path provided'}), 400
    try:
        file_bytes = file.read()
        supabase_client.storage.from_('chat-attachments').upload(
            file_path, 
            file_bytes,
            {"content-type": file.content_type}
        )
        pub_url = supabase_client.storage.from_('chat-attachments').get_public_url(file_path)
        return jsonify({'url': pub_url, 'path': file_path})
    except Exception as e:
        app.logger.exception("Failed to upload attachment")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5001)
