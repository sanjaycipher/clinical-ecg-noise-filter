import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from beatly_pipeline import load_edf, classify_signal_windows, analyze_clean_segments, generate_pdf_report

app = FastAPI(title="Clinical Noise Filter Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok", "service": "Clinical Noise Filter Pipeline API"}


@app.post("/api/analyze")
async def analyze_ecg(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".edf"):
        raise HTTPException(status_code=400, detail="Only EDF files are supported.")

    temp_file = f"temp_upload_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Run pipeline
        signal, fs = load_edf(temp_file)
        windows = classify_signal_windows(signal, fs)
        updated_windows, metrics = analyze_clean_segments(signal, fs, windows)

        # Generate Two-Tier PDF Report
        report_filename = "Beatly_ECG_Report.pdf"
        generate_pdf_report(signal, fs, updated_windows, output_filename=report_filename)

        total = len(updated_windows)
        clean_sec = metrics.get("clean_seconds", 0)
        brady_sec = metrics.get("bradycardia_seconds", 0)
        noise_sec = metrics.get("noise_seconds", 0)
        ld_sec = metrics.get("ld_seconds", 0)

        return {
            "success": True,
            "filename": file.filename,
            "duration_seconds": total,
            "duration_hours": round(total / 3600.0, 2),
            "total_beats": metrics.get("total_r_peaks", 0),
            "metrics": {
                "clean_pct": round((clean_sec / total) * 100, 1) if total > 0 else 0.0,
                "brady_pct": round((brady_sec / total) * 100, 1) if total > 0 else 0.0,
                "noise_pct": round((noise_sec / total) * 100, 1) if total > 0 else 0.0,
                "ld_pct": round((ld_sec / total) * 100, 1) if total > 0 else 0.0,
                "avg_hr": round(metrics.get("avg_heart_rate", 0.0), 1),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass


@app.get("/api/download")
async def download_report():
    pdf_path = "Beatly_ECG_Report.pdf"
    if os.path.exists(pdf_path):
        return FileResponse(pdf_path, filename="Beatly_Clinical_Report.pdf", media_type="application/pdf")
    raise HTTPException(status_code=404, detail="Report not found.")
