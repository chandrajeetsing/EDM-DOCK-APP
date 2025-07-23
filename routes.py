from flask import Blueprint, request, send_from_directory, url_for, render_template, session, redirect
from werkzeug.utils import secure_filename
import os
import subprocess
import uuid
import zipfile
import re
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

upload = Blueprint('upload', __name__)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@upload.route('/', methods=['GET'], endpoint='index')
def index():
    output_message = session.pop('output_message', None)
    pdb_url = None
    run_id = session.get('current_run_id', None)
    
    # Only generate pdb_url if docking was completed
    if run_id and session.get('docking_completed', False):
        # Sanitize run_id to prevent path traversal
        if not re.match(r'^[a-zA-Z0-9_-]+$', run_id):
            return render_template('index.html', output_message="Invalid run ID", pdb_url=None)
        pdb_path = os.path.join(UPLOAD_FOLDER, run_id, "sys1", "sys1_min.pdb")
        if os.path.exists(pdb_path):
            pdb_url = url_for('upload.serve_pdb', run_id=run_id, _external=True)
        else:
            logging.warning(f"sys1_min.pdb not found at {pdb_path} for run_id {run_id}")
    
    # Clear docking_completed flag after rendering to prevent reuse
    session.pop('docking_completed', None)
    
    return render_template('index.html', output_message=output_message, pdb_url=pdb_url)

@upload.route('/visualization', methods=['GET'], endpoint='visualization')
def visualization():
    pdb_url = request.args.get('pdb')
    return render_template('visualization.html', pdb_url=pdb_url)

@upload.route('/upload', methods=['POST'], endpoint='upload_files')
def upload_files():
    protein = request.files.get('protein')
    ligand = request.files.get('ligand')
    box = request.files.get('box')

    if not protein or not ligand or not box:
        session['output_message'] = "❌ All three files (protein, ligand, box) are required."
        return redirect(url_for('upload.index'))

    # Generate unique ID and store in session
    run_id = str(uuid.uuid4())[:8]
    session['current_run_id'] = run_id
    session.pop('docking_completed', None)  # Clear any previous docking flag
    logging.info(f"Starting upload for run_id {run_id}")

    # Create session folders
    upload_session_folder = os.path.join(UPLOAD_FOLDER, run_id, "sys1")
    result_session_folder = os.path.join(RESULTS_FOLDER, run_id)
    os.makedirs(upload_session_folder, exist_ok=True)
    os.makedirs(result_session_folder, exist_ok=True)

    # Save uploaded files
    protein_path = os.path.join(upload_session_folder, secure_filename("protein.pdb"))
    ligand_path = os.path.join(upload_session_folder, secure_filename("ligand.sdf"))
    box_path = os.path.join(upload_session_folder, secure_filename("box.csv"))
    protein.save(protein_path)
    ligand.save(ligand_path)
    box.save(box_path)
    logging.debug(f"Uploaded files: {protein_path}, {ligand_path}, {box_path}")

    try:
        # Run preparation script
        logging.info(f"Running prepare.py for run_id {run_id}")
        prep_result = subprocess.run(
            ["python", "scripts/prepare.py", 
             "--dataset_path", os.path.join(UPLOAD_FOLDER, run_id)],
            capture_output=True, text=True, check=True
        )
        logging.debug(f"prepare.py stdout: {prep_result.stdout}")
        logging.debug(f"prepare.py stderr: {prep_result.stderr}")

        # Run docking script
        logging.info(f"Running dock.py for run_id {run_id}")
        dock_result = subprocess.run(
            ["python", "scripts/dock.py",
             "--run_path", "runs/paper_baseline",
             "--dataset_path", os.path.join(UPLOAD_FOLDER, run_id),
             "--run_id", run_id],
            capture_output=True, text=True, check=True
        )
        logging.debug(f"dock.py stdout: {dock_result.stdout}")
        logging.debug(f"dock.py stderr: {dock_result.stderr}")

        # Paths
        run_results_folder = os.path.join("runs", "paper_baseline", "results", f"run_{run_id}")
        run_results_sys1 = os.path.join(run_results_folder, "sys1")
        uploads_sys1_min = os.path.join(UPLOAD_FOLDER, run_id, "sys1", "sys1_min.pdb")
        zip_filename = f"{run_id}_results.zip"
        zip_path = os.path.join(result_session_folder, zip_filename)

        # Check if sys1_min.pdb exists in uploads folder
        if not os.path.exists(uploads_sys1_min):
            logging.error(f"sys1_min.pdb not found at {uploads_sys1_min}")
            session['output_message'] = (
                f"❌ Error: sys1_min.pdb not found at {uploads_sys1_min}. "
                "Please ensure dock.py generates the file. Check logs for details."
            )
            return redirect(url_for('upload.index'))

        # Check existence of results folder
        if not os.path.exists(run_results_sys1):
            logging.warning(f"Results directory not found at {run_results_sys1}")
            session['output_message'] = f"❌ Error: Results directory not found at {run_results_sys1}"
            return redirect(url_for('upload.index'))

        # Create the zip file
        logging.info(f"Creating ZIP file at {zip_path}")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files from sys1 results folder
            for root, _, files in os.walk(run_results_sys1):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join("sys1", os.path.relpath(file_path, run_results_sys1))
                    zipf.write(file_path, arcname=arcname)
            # Add results.csv if present
            results_csv = os.path.join(run_results_folder, "results.csv")
            if os.path.exists(results_csv):
                zipf.write(results_csv, arcname="results.csv")
            # Add sys1_min.pdb from uploads folder
            zipf.write(uploads_sys1_min, arcname=os.path.join("sys1", "sys1_min.pdb"))

        # Set docking completed flag
        session['docking_completed'] = True
        download_link = url_for('upload.download_result_folder', run_id=run_id, _external=True)
        session['output_message'] = f'✅ Processing complete! <a href="{download_link}" class="download-link">Download Results</a>'
        logging.info(f"Docking completed for run_id {run_id}")
        return redirect(url_for('upload.index'))

    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess error: {e.stderr or 'Unknown error'}")
        session['output_message'] = f"❌ Error during processing: {e.stderr or 'Unknown error'}"
        return redirect(url_for('upload.index'))

@upload.route('/download/<run_id>', methods=['GET'], endpoint='download_result_folder')
def download_result_folder(run_id):
    # Sanitize run_id to prevent path traversal
    if not re.match(r'^[a-zA-Z0-9_-]+$', run_id):
        logging.warning(f"Invalid run_id attempted: {run_id}")
        return "Invalid run ID", 400
    result_folder = os.path.join(RESULTS_FOLDER, run_id)
    zip_file = os.path.join(result_folder, f"{run_id}_results.zip")
    if not os.path.exists(zip_file):
        logging.error(f"ZIP file not found at {zip_file}")
        return "Result file not found.", 404
    return send_from_directory(result_folder, f"{run_id}_results.zip", as_attachment=True)

@upload.route('/pdb/<run_id>', methods=['GET'], endpoint='serve_pdb')
def serve_pdb(run_id):
    # Sanitize run_id to prevent path traversal
    if not re.match(r'^[a-zA-Z0-9_-]+$', run_id):
        logging.warning(f"Invalid run_id attempted: {run_id}")
        return "Invalid run ID", 400
    pdb_dir = os.path.join(UPLOAD_FOLDER, run_id, "sys1")
    pdb_file = "sys1_min.pdb"
    pdb_path = os.path.join(pdb_dir, pdb_file)
    if not os.path.exists(pdb_path):
        logging.error(f"PDB file not found at {pdb_path}")
        return "PDB file not found", 404
    return send_from_directory(pdb_dir, pdb_file, mimetype='chemical/x-pdb')