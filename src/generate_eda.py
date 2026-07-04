import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'spacex_web_scraped.csv')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

def run_full_eda():
    print("Running Full EDA...")
    df = pd.read_csv(DATA_FILE)
    
    # 1. Basic properties
    shape = df.shape
    dtypes = df.dtypes
    missing = df.isnull().sum()
    duplicates = df.duplicated().sum()
    
    # Clean features
    df['Class'] = df['Booster landing'].apply(lambda x: 1 if 'Success' in str(x) else 0)
    df['Payload mass'] = pd.to_numeric(
        df['Payload mass'].astype(str).str.replace('~', '').str.replace(',', '').str.replace('kg', '').str.strip(), 
        errors='coerce'
    )
    df['Payload mass'] = df['Payload mass'].fillna(df['Payload mass'].median())
    df['Flight No.'] = pd.to_numeric(df['Flight No.'], errors='coerce')
    df['Year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year
    
    # Summary stats
    stats = df.describe(include='all').to_markdown()

    # Visualizations
    sns.set_theme(style="darkgrid", palette="muted")
    
    # 1. Countplot (Class distribution)
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Class', hue='Class', legend=False)
    plt.title('Landing Class Distribution')
    plt.xlabel('Class (0 = Failure/No Attempt, 1 = Success)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'class_distribution.png'))
    plt.close()

    # 2. Correlation Matrix Heatmap
    plt.figure(figsize=(6, 4))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'correlation_matrix.png'))
    plt.close()

    # 3. Payload Mass Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='Payload mass', kde=True, bins=20)
    plt.title('Distribution of Payload Mass')
    plt.xlabel('Payload Mass (kg)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'payload_distribution.png'))
    plt.close()

    # 4. Boxplot of Payload Mass by Class
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x='Class', y='Payload mass', hue='Class', legend=False)
    plt.title('Payload Mass by Landing Outcome')
    plt.xlabel('Landing Class')
    plt.ylabel('Payload Mass (kg)')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'payload_boxplot.png'))
    plt.close()

    # 5. Bar chart (Landing success by Launch Site)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x='Launch site', y='Class', errorbar=None, hue='Launch site', legend=False)
    plt.title('Landing Success Rate by Launch Site')
    plt.ylabel('Landing Success Rate')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'success_launch_site.png'))
    plt.close()

    # 6. Bar chart (Landing success by Orbit)
    plt.figure(figsize=(12, 5))
    sns.barplot(data=df, x='Orbit', y='Class', errorbar=None, hue='Orbit', legend=False)
    plt.title('Landing Success Rate by Orbit Type')
    plt.ylabel('Landing Success Rate')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'success_orbit.png'))
    plt.close()

    # 7. Bar chart (Landing success by Booster Version - Top 10)
    plt.figure(figsize=(12, 6))
    top_boosters = df['Version Booster'].value_counts().index[:10]
    sns.barplot(data=df[df['Version Booster'].isin(top_boosters)], x='Version Booster', y='Class', errorbar=None, hue='Version Booster', legend=False)
    plt.title('Landing Success Rate of Top 10 Booster Versions')
    plt.ylabel('Landing Success Rate')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'success_booster.png'))
    plt.close()

    # 8. Line plot (Landing success rate by Launch Year)
    yearly_stats = df.groupby('Year')['Class'].mean().reset_index()
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=yearly_stats, x='Year', y='Class', marker='o', linewidth=2.5)
    plt.title('SpaceX Landing Success Rate Trend Over Years')
    plt.xlabel('Year')
    plt.ylabel('Success Rate')
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'success_year.png'))
    plt.close()

    # 9. Scatter plot (Flight number vs Payload mass colored by Class)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Flight No.', y='Payload mass', hue='Class', style='Class', s=100)
    plt.title('Flight Number vs Payload Mass by Landing Class')
    plt.xlabel('Flight Number')
    plt.ylabel('Payload Mass (kg)')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'flight_payload_scatter.png'))
    plt.close()

    # Compile Markdown Report
    report_path = os.path.join(REPORTS_DIR, 'full_eda_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# SpaceX Falcon 9 Landing Prediction - Full EDA Report\n\n")
        f.write("This report presents the full exploratory data analysis (EDA) as specified in the project requirements.\n\n")
        
        f.write("## 1. Dataset Profile\n")
        f.write(f"- **Shape:** {shape[0]} rows, {shape[1]} columns\n")
        f.write(f"- **Duplicates:** {duplicates} duplicate records\n\n")
        
        f.write("### Data Types & Missing Values:\n")
        f.write("| Column Name | Non-Null Count | Null Count | Data Type |\n")
        f.write("| --- | --- | --- | --- |\n")
        for col in df.columns:
            non_null = df[col].notnull().sum()
            null_val = df[col].isnull().sum()
            f.write(f"| `{col}` | {non_null} | {null_val} | {df[col].dtype} |\n")
        f.write("\n")
        
        f.write("### Summary Statistics:\n")
        f.write(stats + "\n\n")
        
        f.write("## 2. Visualizations and Key Insights\n\n")
        
        f.write("### 2.1 Landing Class Distribution\n")
        f.write("We derived the landing outcome from the raw `Booster landing` column. Success is designated as 1, Failure/No Attempt as 0.\n\n")
        f.write("![Class Distribution](class_distribution.png)\n\n")
        
        f.write("### 2.2 Numerical Relationships (Correlation Heatmap)\n")
        f.write("![Correlation Matrix](correlation_matrix.png)\n\n")
        
        f.write("### 2.3 Payload Mass Distribution\n")
        f.write("![Payload Distribution](payload_distribution.png)\n")
        f.write("![Payload Boxplot](payload_boxplot.png)\n\n")
        
        f.write("### 2.4 Landing Success Rate Analyses\n")
        f.write("#### By Launch Site:\n")
        f.write("![Success Launch Site](success_launch_site.png)\n\n")
        f.write("#### By Orbit Type:\n")
        f.write("![Success Orbit](success_orbit.png)\n\n")
        f.write("#### By Booster Version (Top 10):\n")
        f.write("![Success Booster Version](success_booster.png)\n\n")
        f.write("#### Yearly Trend:\n")
        f.write("![Success Trend Over Time](success_year.png)\n\n")
        
        f.write("### 2.5 Flight Number vs Payload Mass Scatter Plot\n")
        f.write("![Flight No vs Payload Mass](flight_payload_scatter.png)\n\n")
        
    print(f"EDA successfully completed. Full report and plots saved to {REPORTS_DIR}")

if __name__ == '__main__':
    run_full_eda()
