from pathlib import Path

# Data
data_path = Path(Path(__file__).resolve().parent, "Data")
raw_data_path = Path(data_path, "raw")
parsed_data_path = Path(data_path, "parsed")
processed_data_path = Path(data_path, "processed")

# Raw data
ptb_path = Path(raw_data_path, "ptb")
ptb_xl_path = Path(raw_data_path, "ptb-xl")
mimic_note_path = Path(raw_data_path, "mimic-iv-note", "note")
mimic_wdb_path = Path(raw_data_path, "mimic-iv-wdb", "0.1.0", "waves")
mimic_ecg_path = Path(raw_data_path, "mimic-iv-ecg")

# Parsed data
mimic_ecg_parsed_path = Path(parsed_data_path, "mimic-iv-ecg")

# Processed data
ptb_processed_path = Path(processed_data_path, "ptb")
ptb_xl_processed_path = Path(processed_data_path, "ptb-xl")

# Result data
result_path = Path(Path(__file__).resolve().parent, "Results")
recon_path = Path(result_path, "x01_reconstruction_results")
disc_path = Path(result_path, "x02_discriminative_models")
gen_path = Path(result_path, "x03_generative_models")
morphed_path = Path(result_path, "x04_morphed_ecgs")
residual_path = Path(result_path, "x05_linproj_residuals")

code_path = Path(Path(__file__).resolve().parent, "Code")
exp_path = Path(code_path, "experiments")