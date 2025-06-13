import pandas as pd
import re

def remove_empty_reports(df, report_key: str = "report_page1", verbose: bool = False, inplace: bool = False):
    target_df = df if inplace else df.copy()
    
    indices = []

    for idx, report in df[report_key].items():
        if pd.isna(report) or report == "b''" or report == "" or report.strip() == "":
            indices.append(idx)

    target_df.drop(index=indices, axis=0, inplace=True)
    target_df.reset_index(drop=True, inplace=True)

    if verbose:
        print(f'no. of empty reports: {len(indices)}, patient_indices: {indices}')

    if not inplace:
        return target_df

def extract_findings_section(df, report_key: str = "report_page1", verbose: bool = False, inplace: bool = False):
    target_df = df if inplace else df.copy() 

    empty_extractions = 0
    extracted_findings_list = []
    bad_indices = []

    for i, report in enumerate(target_df[report_key]):
        match = re.search(r'(.+)(?:Measurements and Calculations:)', report, re.DOTALL | re.IGNORECASE)
        if match:
            findings = match.group(1)
            
            # Remove study info line
            match = re.search(r'Study Info:.*', findings, re.IGNORECASE)
            if match:
                findings = findings[:match.start()] + findings[match.end():]

            # Remove comparison to previous exam line
            match = re.search(r'Comparison to Previous Exam.*|Compared to prior exam.*|Compared to previous study.*', findings, re.IGNORECASE)
            if match:
                findings = findings[:match.start()] + findings[match.end():]

            cleaned_lines = [line.strip() for line in findings.splitlines() if line.strip()]
            
            extracted_findings_list.append('\n'.join(cleaned_lines))
        else:
            match2 = re.search(r'(.+)(?:Comparison to Previous Exam)|(.+)(?:Compared to prior exam)|(.+)(?:Compared to previous study)', report, re.DOTALL | re.IGNORECASE)

            if match2:
                findings = match2.group(1)
                # Remove study info line
                match = re.search(r'Study Info:.*', findings, re.IGNORECASE)
                if match:
                    findings = findings[:match.start()] + findings[match.end():]

                cleaned_lines = [line.strip() for line in findings.splitlines() if line.strip()]
                
                extracted_findings_list.append('\n'.join(cleaned_lines))
            else:
                empty_extractions += 1
                extracted_findings_list.append(-1)
                bad_indices.append(target_df[report_key].index[i])
    
    if verbose:
        print(f"Empty extractions: {empty_extractions} Extraction percentage: {empty_extractions/len(target_df)*100:.2f}%")
        print(f"Bad reports: {bad_indices}")

    target_df["extracted_findings"] = extracted_findings_list

    if not inplace:
        return target_df

def extract_calculations_section(df, report_key: str = "report_page1", verbose: bool = False, inplace: bool = True):
    target_df = df if inplace else df.copy() 
    
    empty_extractions = 0
    extracted_calcs_list = []
    bad_indices = []

    for i, report in enumerate(target_df[report_key]):
        match = re.search(r'(?:Measurements and Calculations:)(.+?)(?:Sonographer|$)', report, re.DOTALL | re.IGNORECASE)
        
        if match:
            findings = match.group(1)
            cleaned_lines = [line.strip() for line in findings.splitlines() if line.strip()]
            extracted_calcs_list.append('\n'.join(cleaned_lines))
        else:
            match2 = re.search(r'(RVd\s+A4C:.+?)(?:Sonographer|$)', report, re.DOTALL | re.IGNORECASE)

            if match2:
                findings = match2.group(1)
                cleaned_lines = [line.strip() for line in findings.splitlines() if line.strip()]
                extracted_calcs_list.append('\n'.join(cleaned_lines))
            else:
                empty_extractions += 1
                extracted_calcs_list.append(-1)
                bad_indices.append(target_df[report_key].index[i])
    
    if verbose:
        print(f"Empty extractions: {empty_extractions} Extraction percentage: {empty_extractions/len(target_df)*100:.2f}%")
        print(f"Bad reports: {bad_indices}")

    target_df["extracted_calcs"] = extracted_calcs_list

    if not inplace:
        return target_df

def drop_unusable_reports(df, verbose: bool = False, inplace: bool = False):
    target_df = df if inplace else df.copy()

    bad_indices =[]

    for idx, row in target_df.iterrows():
        if row["extracted_calcs"] == -1 or row["extracted_findings"] == -1:
            bad_indices.append(idx)

    target_df.drop(index = bad_indices, axis=0, inplace=True)
    target_df.reset_index(drop=True, inplace=True) 

    if verbose:
        print(f'no. of unusable reports: {len(bad_indices)}, table indices: {bad_indices}')
    
    if not inplace:
        return target_df
    
def process_dataframe(df, verbose: bool = False, inplace: bool = False):
    target_df = df if inplace else df.copy()

    if verbose:
        print("Processing dataframe.")

    remove_empty_reports(target_df, verbose=verbose, inplace = True)
    extract_findings_section(target_df, verbose=verbose, inplace=True)
    extract_calculations_section(target_df, verbose=verbose, inplace=True)
    drop_unusable_reports(target_df, verbose=verbose, inplace=True)

    if verbose:
        print("Finish processing dataframe.")
    
    if not inplace:
        return target_df

def split_data(df, train_fraction, val_fraction, random_state:int = 42, verbose: bool = False):
    assert train_fraction + val_fraction <= 1

    grouped = df.groupby("patient_id").first().reset_index()
    patients = grouped["patient_id"].sample(frac=1, random_state=random_state).reset_index(drop=True)

    train_index = int(train_fraction * len(patients))
    val_index = int(val_fraction * len(patients))

    train_patients = patients[:train_index]
    val_patients = patients[train_index:train_index + val_index]
    test_patients = patients[train_index + val_index:]

    if verbose:
        print(f"Train Patients: {len(train_patients)}, Val Patients: {len(val_patients)}, Test Patients: {len(test_patients)}")

    df_train = df[df["patient_id"].isin(train_patients)]
    df_val = df[df["patient_id"].isin(val_patients)]
    df_test = df[df["patient_id"].isin(test_patients)]

    return df_train, df_val, df_test


def run_calc_searches(report_text):
    la_mm = re.search(r"left\s+atrium:\s+[*]?(\d+)\s+mm", report_text, flags = re.IGNORECASE)
    ivsd = re.search(r"ivsd:\s+[*]?(\d+)\s+mm", report_text, flags = re.IGNORECASE)
    lvpwd = re.search(r"lvpwd:\s+[*]?(\d+)\s+mm", report_text, flags = re.IGNORECASE)
    la_vol_bpidx = re.search(r"la\s?(?:volume|vol)\s?index\s?(?:\(Biplane\))?:\s+[*]?([\d\.]+)\s+ml/m", report_text, flags = re.IGNORECASE)
    lvidd = re.search(r"lvidd:\s+[*]?(\d+)", report_text, flags = re.IGNORECASE)
    lvidd_idx = re.search(r"lvidd\sindex:\s+[*]?(\d+)\s+mm/m", report_text, flags = re.IGNORECASE)
    rvd_a4c = re.search(r"rvd\sa4c:\s+[*]?(\d+)\s+mm", report_text, flags = re.IGNORECASE)
    lvids = re.search(r"lvids:\s+[*]?(\d+)\s+mm", report_text, flags = re.IGNORECASE)
    rv_s = re.search(r"rv\sS.\s+[*]?([\d\.]+)\s+cm/s", report_text, flags = re.IGNORECASE)
    lvef = re.search(r"lv\s?ef\s?.(?:Biplane|Visual).:\s+[*]?(\d+)\s?%?", report_text, flags = re.IGNORECASE)
    tapse = re.search(r"tapse:\s+[*]?(\d+)\s+mm", report_text, flags = re.IGNORECASE)
    lv_mass_idx = re.search(r"lv\smass\sindex:\s+[*]?([\d\.]+)\s+g/m", report_text, flags = re.IGNORECASE)   
    lv_rwt = re.search(r"lv\s?rwt:\s+[*]?([\d\.]+)", report_text, flags = re.IGNORECASE)
    ra_vol_idx = re.search(r"ra\s?vol[\w|\s]+index:\s+[*]?(\d+)\s+ml/m", report_text, flags = re.IGNORECASE)
    lv_edv_idx = re.search(r"lv\s?edv\s?index:\s+[*]?([\d\.]+)\s+ml/m", report_text, flags = re.IGNORECASE)
    lv_esv_idx = re.search(r"lv\s?esv\s?index:\s+[*]?([\d\.]+)\s+ml/m", report_text, flags = re.IGNORECASE)
    aorta_sinuses = re.search(r"aorta\ssinuses:\s+[*]?(\d+)\s+mm", report_text, flags = re.IGNORECASE)
    lvot_diam = re.search(r"lvot\sdiam:\s+[*]?(\d+)\s+mm", report_text, flags = re.IGNORECASE)
    aorta_sinuses_idx = re.search(r"aorta\ssinuses\sindex:\s+[*]?([\d\.]+)\s+mm/m", report_text, flags = re.IGNORECASE)
    prox_asc_aorta = re.search(r"prox\sascending\saorta:\s+[*]?(\d+)\s+mm", report_text, flags = re.IGNORECASE)
    prox_asc_aorta_idx = re.search(r"prox\sasc\saorta\sindex:\s+[*]?([\d\.]+)\s+mm/m", report_text, flags = re.IGNORECASE)
    mv_peak_e = re.search(r"mv\speak\se:\s+[*]?([\d\.]+)\s+cm/s", report_text, flags = re.IGNORECASE)
    mv_peak_a = re.search(r"mv\speak\sa:\s+[*]?([\d\.]+)\s+cm/s", report_text, flags = re.IGNORECASE)
    mv_ea_ratio = re.search(r"mv\se/a\sratio:\s+[*]?([\d\.]+)", report_text, flags = re.IGNORECASE)
    decel_time = re.search(r"decel\stime:\s+[*]?(\d+)\s+msec", report_text, flags = re.IGNORECASE)
    lateral_e = re.search(r"lateral\se\s?:\s+[*]?([\d\.]+)\s+cm/s", report_text, flags = re.IGNORECASE)
    septal_e = re.search(r"septal\se\s?:\s+[*]?([\d\.]+)\s+cm/s", report_text, flags = re.IGNORECASE)
    avg_ee_ratio = re.search(r"average\se/e\sratio:\s+[*]?([\d\.]+)", report_text, flags = re.IGNORECASE)
    tr_max_velocity = re.search(r"TR\s+max\s+velocity:\s+[*]?([\d\.]+)\s+m/s", report_text, flags = re.IGNORECASE)
    ra_pressure = re.search(r"ra\spressure:?\s+[*]?([\d\.]+)\s+mmHg", report_text, flags = re.IGNORECASE)
    pasp = re.search(r"pasp:\s+[*]?(\d+)\s+mmHg", report_text, flags = re.IGNORECASE)


    measurements = [la_mm, la_vol_bpidx,
                   lvpwd, lvidd, lvidd_idx, lvef, lv_mass_idx, lv_rwt, lv_edv_idx, lv_esv_idx, lvot_diam,
                   ivsd, lvids,
                   rvd_a4c, rv_s,
                    tapse,
                   ra_vol_idx,
                    aorta_sinuses, aorta_sinuses_idx, prox_asc_aorta, prox_asc_aorta_idx,
                   mv_peak_e, mv_peak_a, mv_ea_ratio,
                    decel_time,
                    lateral_e, septal_e,
                    avg_ee_ratio,
                    tr_max_velocity,
                    ra_pressure,
                    pasp]
    
    measurements_ls = [m.group(1) if m !=None else -1 for m in measurements]
    
    return measurements_ls