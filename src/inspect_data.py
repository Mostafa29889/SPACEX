import os
import csv
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'spacex_web_scraped.csv')
REPORT_FILE = os.path.join(BASE_DIR, 'reports', 'step_1_2_report.md')

def run_analysis():
    print("Reading and analyzing dataset using built-in Python modules...")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset file not found at {DATA_FILE}")

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = [row for row in reader if row]

    total_rows = len(rows)
    total_cols = len(headers)
    
    # 1. Column Names and Missing Values
    missing_counts = {header: 0 for header in headers}
    for row in rows:
        for idx, val in enumerate(row):
            if idx < len(headers):
                if val.strip() == '' or val.strip().lower() == 'none' or val.strip() == 'U':
                    missing_counts[headers[idx]] += 1

    # 2. Duplicate Records
    seen = set()
    duplicates = 0
    for row in rows:
        row_tuple = tuple(row)
        if row_tuple in seen:
            duplicates += 1
        seen.add(row_tuple)

    # 3. Target Class Derivation & Class Distribution
    # Landing outcome index is 10 (Booster landing)
    success_count = 0
    fail_count = 0
    
    # Success rates by Launch Site (index 1) and Orbit (index 4)
    launch_sites = {}
    orbits = {}
    
    # Payload Mass (index 3)
    payload_masses = []
    
    for row in rows:
        if len(row) < 11:
            continue
            
        landing = row[10]
        site = row[1]
        orbit = row[4]
        payload_mass_str = row[3]
        
        # Determine Success (Class = 1 if 'Success' in landing else 0)
        is_success = 'Success' in landing
        if is_success:
            success_count += 1
        else:
            fail_count += 1
            
        # Success by Launch Site
        if site not in launch_sites:
            launch_sites[site] = {'total': 0, 'success': 0}
        launch_sites[site]['total'] += 1
        if is_success:
            launch_sites[site]['success'] += 1
            
        # Success by Orbit
        if orbit not in orbits:
            orbits[orbit] = {'total': 0, 'success': 0}
        orbits[orbit]['total'] += 1
        if is_success:
            orbits[orbit]['success'] += 1
            
        # Parse payload mass (remove commas, 'kg' suffix, ignore U)
        clean_mass = payload_mass_str.replace(',', '').replace('~', '').replace('kg', '').strip()
        try:
            val = float(clean_mass)
            payload_masses.append(val)
        except ValueError:
            pass

    # Summarize Payload Mass
    if payload_masses:
        payload_masses.sort()
        avg_mass = sum(payload_masses) / len(payload_masses)
        min_mass = min(payload_masses)
        max_mass = max(payload_masses)
        median_mass = payload_masses[len(payload_masses)//2]
    else:
        avg_mass = min_mass = max_mass = median_mass = 0

    # Write Markdown Report
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("# SpaceX Falcon 9 Landing Prediction - Step 1 & 2 Analysis Report\n\n")
        
        f.write("## 1. Dataset Understanding\n")
        f.write("- **Dataset Name:** `spacex_web_scraped.csv`\n")
        f.write(f"- **Shape:** {total_rows} rows, {total_cols} columns\n")
        f.write("- **Target Variable:** derived from `Booster landing` (Landing Outcome)\n")
        f.write("  - **Class 1 (Success):** Booster landing contains the string `Success`\n")
        f.write("  - **Class 0 (Failure):** All other landing outcomes (including no attempts, failures, crashes)\n\n")
        
        f.write("### Feature Classification & Descriptions:\n")
        f.write("| Feature Name | Data Type | Description |\n")
        f.write("| --- | --- | --- |\n")
        f.write("| `Flight No.` | Numerical (Discrete) | The sequential number of the Falcon 9 launch. |\n")
        f.write("| `Launch site` | Categorical (Nominal) | The spaceport launch pad (e.g. Cape Canaveral, VAFB, KSC). |\n")
        f.write("| `Payload` | Categorical (Nominal) | Name of the satellite or spacecraft being launched. |\n")
        f.write("| `Payload mass` | Numerical (Continuous) | Mass of the payload in kilograms. |\n")
        f.write("| `Orbit` | Categorical (Nominal) | The target destination orbit (e.g. LEO, GTO, ISS). |\n")
        f.write("| `Customer` | Categorical (Nominal) | The entity paying for the launch service. |\n")
        f.write("| `Launch outcome` | Categorical (Binary) | Success or failure of the ascent phase. |\n")
        f.write("| `Date` | Date/Time | The calendar date of the launch. |\n")
        f.write("| `Time` | Date/Time | The launch time (UTC). |\n")
        f.write("| `Version Booster` | Categorical (Nominal) | Specific rocket model/variant version. |\n")
        f.write("| `Booster landing` | Categorical (Nominal) | Landing pad method and outcome details. |\n\n")
        
        f.write("## 2. Exploratory Data Analysis (EDA) Summary\n\n")
        f.write("### Data Quality Metrics:\n")
        f.write(f"- **Duplicates:** {duplicates} duplicate records found.\n")
        f.write("- **Missing Values (or Unknowns) per Column:**\n")
        for col, count in missing_counts.items():
            f.write(f"  - `{col}`: {count} missing/unknown entries\n")
        f.write("\n")
        
        f.write("### Target Class Distribution:\n")
        success_rate = (success_count / total_rows) * 100 if total_rows > 0 else 0
        f.write(f"- **Successful Landings (Class 1):** {success_count} ({success_rate:.2f}%)\n")
        f.write(f"- **Unsuccessful Landings (Class 0):** {fail_count} ({100 - success_rate:.2f}%)\n\n")
        
        f.write("### Summary Statistics for Payload Mass (kg):\n")
        f.write(f"- **Mean Mass:** {avg_mass:.2f} kg\n")
        f.write(f"- **Median Mass:** {median_mass:.2f} kg\n")
        f.write(f"- **Min Mass:** {min_mass:.2f} kg\n")
        f.write(f"- **Max Mass:** {max_mass:.2f} kg\n\n")
        
        f.write("### Landing Success Rate by Launch Site:\n")
        f.write("| Launch Site | Total Launches | Landing Successes | Landing Success Rate |\n")
        f.write("| --- | --- | --- | --- |\n")
        for site, stats in sorted(launch_sites.items()):
            rate = (stats['success'] / stats['total']) * 100
            f.write(f"| {site} | {stats['total']} | {stats['success']} | {rate:.2f}% |\n")
        f.write("\n")
        
        f.write("### Landing Success Rate by Orbit Type:\n")
        f.write("| Orbit Type | Total Launches | Landing Successes | Landing Success Rate |\n")
        f.write("| --- | --- | --- | --- |\n")
        for orbit, stats in sorted(orbits.items()):
            rate = (stats['success'] / stats['total']) * 100
            f.write(f"| {orbit} | {stats['total']} | {stats['success']} | {rate:.2f}% |\n")
        f.write("\n")
        
    print(f"Analysis complete. Report written to {REPORT_FILE}")

if __name__ == '__main__':
    run_analysis()
