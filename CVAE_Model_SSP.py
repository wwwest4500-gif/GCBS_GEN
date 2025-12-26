# -*- coding: utf-8 -*-
"""
CVAE Climate Scenario Generation System v3.4
Updates:
1. Fixed missing function definitions 
2. Resolved torch.load warnings 
3. Added physical constraint post-processing 
4. Enhanced validation with all 22 variables visualization
5. Added CSV export of validation statistics 
Last updated: 2025-11-25 14:13 
"""

import pandas as pd 
import numpy as np 
import torch 
import torch.nn as nn 
from torch.optim import Adam 
from datetime import datetime 
import re 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os 

# System initialization 
print(f"\n{'='*50}")
print(f"CVAE Climate Scenario System Starting @ {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*50}\n")

# ==================== 1. Data Loading and Preprocessing ====================
def load_data(data_path):
    """Intelligent data loading and validation - for Excel format"""
    try:
        # Read Excel file
        df = pd.read_excel(data_path) 
        print(f"Data dimensions: {df.shape} | Variables: {len(df.columns)-1}")  
        
        # Key column detection
        required_columns = {
            'year': 'Year',
            'temperature': 'Global mean temperature',
            'co2': 'Atmospheric Carbon Dioxide (CO2)'
        }
        
        # Column name validation 
        missing_cols = [col for col in required_columns.values() if col not in df.columns]  
        if missing_cols:
            raise ValueError(f"Missing key columns: {missing_cols}\nExisting columns: {df.columns.tolist()}")  
        
        # Data extraction - 修正这里：移除，只保留列索引计算
        data_values = df.drop(columns=['Year']).values.astype(float)  
        temp_col = df.columns.get_loc(required_columns['temperature']) - 1  # 移除
        co2_col = df.columns.get_loc(required_columns['co2']) - 1 
        
        # Data separation
        c = data_values[:, temp_col]  # Temperature 
        co2 = data_values[:, co2_col]  # CO2 
        
        # Special standardization
        stats = {
            'mean_x': np.mean(np.delete(data_values, [temp_col, co2_col], axis=1), axis=0),
            'std_x': np.std(np.delete(data_values, [temp_col, co2_col], axis=1), axis=0),
            'mean_c': np.mean(c),  
            'std_c': np.std(c),  
            'mean_co2': np.mean(co2),  
            'std_co2': np.std(co2),  
            'base_temperature': 13.5,  # 添加基温信息
            'feature_names': [col for col in df.columns if col not in ['Year', 'Global mean temperature', 'Atmospheric Carbon Dioxide (CO2)']]
        }
        
        return df, stats 
    
    except Exception as e:
        raise ValueError(f"Excel file reading failed: {str(e)}")

# ==================== 2. Physical Constraint Functions ====================
def apply_physical_constraints(scenario_df, stats):
    """
    Apply physical constraint post-processing 
    1. Non-negative emission constraint 
    2. Fossil fuel emission consistency: 'fossil.emissions.excluding.carbonation' = 'Coal' + 'Oil' + 'Gas' + 'Cement.emission' + 'Flaring' + 'Other'
    3. Carbon conservation check 
    4. Temperature-emission consistency validation
    """
    print("\n" + "="*50)
    print("Applying Physical Constraint Post-processing")
    print("="*50)
    
    # Deep copy to avoid modifying original data 
    df_processed = scenario_df.copy() 
    
    # 1. Non-negative emission constraint: ReLU truncation
    print("1. Applying non-negative emission constraint...")
    emission_columns = [
        'Coal', 'Oil', 'Gas', 'Cement.emission', 'Flaring', 'Other',
        'fossil.emissions.excluding.carbonation', 'Per.Capita'
    ]
    
    # Keep only existing columns 
    existing_emission_cols = [col for col in emission_columns if col in df_processed.columns] 
    
    for col in existing_emission_cols:
        original_min = df_processed[col].min()
        df_processed[col] = np.maximum(df_processed[col], 0)
        adjusted_count = (df_processed[col] <= 0).sum() if original_min < 0 else 0
        if adjusted_count > 0:
            print(f"  {col}: Adjusted {adjusted_count} negative values to 0 (original min: {original_min:.4f})")
    
    # 2. Fossil fuel emission consistency constraint
    print("\n2. Applying fossil fuel emission consistency constraint...")
    fossil_components = ['Coal', 'Oil', 'Gas', 'Cement.emission', 'Flaring', 'Other']
    fossil_total_col = 'fossil.emissions.excluding.carbonation' 
    
    # Check if all required columns exist
    if fossil_total_col in df_processed.columns and all(col in df_processed.columns for col in fossil_components):
        # Calculate the sum of components
        component_sum = df_processed[fossil_components].sum(axis=1)
        
        # Calculate the difference between total and sum of components
        difference = df_processed[fossil_total_col] - component_sum 
        max_diff = difference.abs().max() 
        mean_diff = difference.abs().mean() 
        
        print(f"  Fossil emission consistency check:")
        print(f"    Maximum difference: {max_diff:.6f}")
        print(f"    Mean absolute difference: {mean_diff:.6f}")
        
        if max_diff > 1e-6:  # If there's a significant difference
            print(f"  Adjusting {fossil_total_col} to match sum of components...")
            
            # Option 1: Adjust total to match sum of components (more physically consistent)
            original_total = df_processed[fossil_total_col].copy()
            df_processed[fossil_total_col] = component_sum
            
            # Calculate adjustment statistics
            adjustment_pct = ((df_processed[fossil_total_col] - original_total) / original_total * 100).abs()
            max_adjustment_pct = adjustment_pct.max() 
            mean_adjustment_pct = adjustment_pct.mean() 
            
            print(f"  Adjustment statistics:")
            print(f"    Maximum adjustment: {max_adjustment_pct:.2f}%")
            print(f"    Mean adjustment: {mean_adjustment_pct:.2f}%")
            
            # Verify the adjustment 
            new_difference = df_processed[fossil_total_col] - component_sum 
            new_max_diff = new_difference.abs().max() 
            print(f"  After adjustment - Maximum difference: {new_max_diff:.6f}")
            
            if new_max_diff < 1e-10:
                print(f"  ✓ Success: Fossil emission consistency constraint satisfied")
            else:
                print(f"   ⚠ Warning: Small residual differences remain after adjustment")
        else:
            print(f"  ✓ Pass: Fossil emission consistency already satisfied")
    else:
        missing_cols = [col for col in fossil_components + [fossil_total_col] if col not in df_processed.columns] 
        print(f"  Skipping fossil emission consistency: Missing columns {missing_cols}")
    
    # 3. Carbon conservation check: Total emissions ≈ Ocean sink + Land sink + Atmospheric increase 
    print("\n3. Carbon conservation check...")
    
    # Check if necessary columns exist 
    required_carbon_cols = [
        'fossil.emissions.excluding.carbonation', 
        'deforestation (total)',
        'Ocean_Sink',
        'land_Sink',
        'Cement Carbonation Sink',
        'Atmospheric Carbon Dioxide (CO2)'
    ]
    
    missing_carbon_cols = [col for col in required_carbon_cols if col not in df_processed.columns] 
    
    if not missing_carbon_cols:
        # Calculate carbon budget 
        # Estimate atmospheric increase (simplified: using CO2 change)
        df_processed['CO2_change'] = df_processed.groupby('Scenario')['Atmospheric Carbon Dioxide (CO2)'].diff().fillna(0)
        # CO2 ppm to GtC conversion: 1 ppm CO2 ≈ 2.12 GtC 
        df_processed['Atmospheric_increase'] = df_processed['CO2_change'] * 2.12
        
        # Calculate carbon balance residual 
        df_processed['Carbon_residual'] = (
            df_processed['fossil.emissions.excluding.carbonation'] / 1000 +  # Convert to GtC 
            df_processed['deforestation (total)'] -
            df_processed['Ocean_Sink'] -
            df_processed['land_Sink'] -
            df_processed['Cement Carbonation Sink'] -
            df_processed['Atmospheric_increase']
        )
        
        residual_mean = df_processed['Carbon_residual'].abs().mean()
        residual_std = df_processed['Carbon_residual'].std()
        print(f"  Carbon balance residual statistics:")
        print(f"    Absolute mean: {residual_mean:.4f} GtC/yr")
        print(f"    Standard deviation: {residual_std:.4f} GtC/yr")
        print(f"    Maximum: {df_processed['Carbon_residual'].max():.4f} GtC/yr")
        print(f"    Minimum: {df_processed['Carbon_residual'].min():.4f} GtC/yr")
        
        if residual_mean > 0.5:
            print(f"  Warning: Large carbon balance residual, may need adjustment")
        else:
            print(f"  Pass: Carbon balance residual within acceptable range")
    else:
        print(f"  Skipping carbon conservation check: Missing required columns {missing_carbon_cols}")
    
    # 4. Temperature-emission consistency: Validate using simplified climate model 
    print("\n4. Temperature-emission consistency validation...")
    
    def simple_climate_model(co2_series, base_temp, tcr=1.8, base_co2=280):
        """
        Simplified climate model: Estimate temperature using Transient Climate Response (TCR)
        co2_series: CO2 concentration series (ppm)
        base_temp: Base temperature (°C)
        tcr: Transient Climate Response (°C per 2xCO2)
        base_co2: Pre-industrial CO2 baseline concentration (ppm)
        """
        # Temperature change corresponding to CO2 doubling 
        delta_temp = tcr * np.log2(co2_series / base_co2)
        return base_temp + delta_temp
    
    # Validate each scenario
    scenarios = df_processed['Scenario'].unique()
    temp_errors = []
    co2_temperatures = []
    
    for scenario in scenarios[:10]:  # Check first 10 scenarios
        scenario_data = df_processed[df_processed['Scenario'] == scenario]
        if len(scenario_data) > 0:
            # Estimate temperature using model-generated CO2
            estimated_temp = simple_climate_model(
                scenario_data['Atmospheric Carbon Dioxide (CO2)'].values,
                base_temp=scenario_data['Global mean temperature'].iloc[0],
                tcr=1.8,
                base_co2=280 
            )
            
            # Calculate difference from model-generated temperature 
            actual_temp = scenario_data['Global mean temperature'].values
            mae = np.mean(np.abs(estimated_temp - actual_temp))
            rmse = np.sqrt(np.mean((estimated_temp - actual_temp) ** 2))
            temp_errors.append({'mae': mae, 'rmse': rmse})
            
            # Record CO2-temperature relationship
            co2_temperatures.append({ 
                'scenario': scenario,
                'co2_range': [scenario_data['Atmospheric Carbon Dioxide (CO2)'].min(), 
                            scenario_data['Atmospheric Carbon Dioxide (CO2)'].max()],
                'temp_range': [scenario_data['Global mean temperature'].min(),
                             scenario_data['Global mean temperature'].max()]
            })
    
    if temp_errors:
        avg_mae = np.mean([err['mae'] for err in temp_errors])
        avg_rmse = np.mean([err['rmse'] for err in temp_errors])
        print(f"  Temperature-emission consistency validation:")
        print(f"    Mean Absolute Error (MAE): {avg_mae:.4f} °C")
        print(f"    Root Mean Square Error (RMSE): {avg_rmse:.4f} °C")
        
        # Check CO2-temperature relationship reasonableness
        if len(co2_temperatures) > 0:
            avg_co2_change = np.mean([t['co2_range'][1] - t['co2_range'][0] for t in co2_temperatures])
            avg_temp_change = np.mean([t['temp_range'][1] - t['temp_range'][0] for t in co2_temperatures])
            co2_temp_sensitivity = avg_temp_change / (avg_co2_change / 280)  # Temperature change per CO2 doubling 
            
            print(f"  CO2-temperature relationship statistics:")
            print(f"    Average CO2 change: {avg_co2_change:.1f} ppm")
            print(f"    Average temperature change: {avg_temp_change:.2f} °C")
            print(f"    Estimated CO2 sensitivity: {co2_temp_sensitivity:.2f} °C/2xCO2")
            
            if 1.0 <= co2_temp_sensitivity <= 3.0:
                print(f"    Pass: CO2 sensitivity within reasonable range")
            else:
                print(f"    Warning: CO2 sensitivity ({co2_temp_sensitivity:.2f} °C/2xCO2) may exceed typical range (1.5-4.5°C/2xCO2)")
        
        if avg_mae > 0.5:
            print(f"  Warning: Large temperature-emission consistency deviation")
        else:
            print(f"  Pass: Temperature-emission consistency is good")
    
    # Remove temporary calculation columns
    cols_to_drop = ['CO2_change', 'Atmospheric_increase', 'Carbon_residual']
    for col in cols_to_drop:
        if col in df_processed.columns: 
            df_processed = df_processed.drop(columns=[col]) 
    
    print(f"\nPhysical constraint post-processing completed!")
    print(f"Original data rows: {len(scenario_df)}")
    print(f"Processed data rows: {len(df_processed)}")
    
    # Generate quality report
    generate_quality_report(df_processed)
    
    return df_processed

def generate_quality_report(df_processed):
    """Generate data quality report"""
    print("\n" + "="*50)
    print("Data Quality Report")
    print("="*50)
    
    report = []
    
    # 1. Basic statistics 
    report.append("1. Basic Statistics:")
    report.append(f"    Total data rows: {len(df_processed)}")
    report.append(f"    Number of scenarios: {df_processed['Scenario'].nunique()}")
    report.append(f"    Year range: {df_processed['Year'].min()} - {df_processed['Year'].max()}")
    
    # 2. Key variable ranges 
    report.append("\n2. Key Variable Ranges:")
    key_vars = ['Global mean temperature', 'Atmospheric Carbon Dioxide (CO2)', 
                'Coal', 'Oil', 'Gas', 'Ocean_Sink', 'land_Sink']
    
    for var in key_vars:
        if var in df_processed.columns: 
            min_val = df_processed[var].min()
            max_val = df_processed[var].max()
            mean_val = df_processed[var].mean()
            report.append(f"    {var}: {min_val:.2f} to {max_val:.2f} (mean: {mean_val:.2f})")
    
    # 3. Fossil emission consistency check 
    report.append("\n3. Fossil Emission Consistency:")
    fossil_components = ['Coal', 'Oil', 'Gas', 'Cement.emission', 'Flaring', 'Other']
    fossil_total_col = 'fossil.emissions.excluding.carbonation' 
    
    if fossil_total_col in df_processed.columns and all(col in df_processed.columns for col in fossil_components):
        component_sum = df_processed[fossil_components].sum(axis=1)
        difference = df_processed[fossil_total_col] - component_sum
        max_diff = difference.abs().max() 
        mean_diff = difference.abs().mean() 
        
        report.append(f"    Fossil emission consistency:")
        report.append(f"      Maximum difference: {max_diff:.6f}")
        report.append(f"      Mean absolute difference: {mean_diff:.6f}")
        
        if max_diff < 1e-10:
            report.append(f"      ✓ PASS: Fossil emission consistency satisfied")
        else:
            report.append(f"       ⚠ WARNING: Small residual differences remain")
    else:
        missing_cols = [col for col in fossil_components + [fossil_total_col] if col not in df_processed.columns] 
        report.append(f"    Missing columns for fossil emission check: {missing_cols}")
    
    # 4. Outlier detection
    report.append("\n4. Outlier Detection:")
    
    # Check temperature outliers
    if 'Global mean temperature' in df_processed.columns: 
        temp_data = df_processed['Global mean temperature']
        temp_mean = temp_data.mean() 
        temp_std = temp_data.std() 
        outliers = temp_data[(temp_data < temp_mean - 3*temp_std) | (temp_data > temp_mean + 3*temp_std)]
        report.append(f"    Temperature outlier count (3σ rule): {len(outliers)}")
    
    # 5. Temporal continuity check
    report.append("\n5. Temporal Continuity Check:")
    scenarios = df_processed['Scenario'].unique()[:5]  # Check first 5 scenarios
    all_continuous = True 
    
    for scenario in scenarios:
        years = df_processed[df_processed['Scenario'] == scenario]['Year'].values 
        if len(years) > 1:
            year_diff = np.diff(years) 
            if not np.all(year_diff == 1):
                all_continuous = False 
                report.append(f"    Scenario {scenario}: Time series not continuous")
                break
    
    if all_continuous:
        report.append("    All checked scenarios have continuous time series")
    
    # Print report
    for line in report:
        print(line)
    
    # Save report to file
    report_text = "\n".join(report)
    with open('CVAE_quality_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text) 
    
    print(f"\nQuality report saved: CVAE_quality_report.txt") 

def check_scenario_quality(scenario_df):
    """Check scenario data quality"""
    print("\n" + "="*50)
    print("Scenario Data Quality Check")
    print("="*50)
    
    checks_passed = 0
    total_checks = 5  # Increased from 4 to 5 to include fossil emission consistency
    
    # 1. Check for NaN values
    nan_count = scenario_df.isna().sum().sum() 
    if nan_count == 0:
        print(f"✓ Pass: No NaN values")
        checks_passed += 1
    else:
        print(f"✗ Fail: Found {nan_count} NaN values")
    
    # 2. Check for negative emission values 
    emission_cols = ['Coal', 'Oil', 'Gas', 'Cement.emission', 'Flaring', 'Other']
    emission_cols = [col for col in emission_cols if col in scenario_df.columns] 
    
    negative_count = 0 
    for col in emission_cols:
        negative_count += (scenario_df[col] < 0).sum()
    
    if negative_count == 0:
        print(f"✓ Pass: All emission values are non-negative")
        checks_passed += 1 
    else:
        print(f"✗ Fail: Found {negative_count} negative emission values")
    
    # 3. Check fossil emission consistency
    fossil_components = ['Coal', 'Oil', 'Gas', 'Cement.emission', 'Flaring', 'Other']
    fossil_total_col = 'fossil.emissions.excluding.carbonation' 
    
    if fossil_total_col in scenario_df.columns and all(col in scenario_df.columns for col in fossil_components):
        component_sum = scenario_df[fossil_components].sum(axis=1)
        difference = scenario_df[fossil_total_col] - component_sum
        max_diff = difference.abs().max() 
        
        if max_diff < 1e-10:
            print(f"✓ Pass: Fossil emission consistency satisfied")
            checks_passed += 1 
        else:
            print(f"✗ Fail: Fossil emission inconsistency (max diff: {max_diff:.6f})")
    else:
        print(f"⚠ Skip: Missing columns for fossil emission consistency check")
        total_checks -= 1  # Reduce total since we're skipping this check
    
    # 4. Check data range reasonableness 
    print("\nData Range Check:")
    check_columns = ['Global mean temperature', 'Atmospheric Carbon Dioxide (CO2)', 'Coal', 'Oil']
    for col in check_columns:
        if col in scenario_df.columns: 
            min_val = scenario_df[col].min()
            max_val = scenario_df[col].max()
            mean_val = scenario_df[col].mean()
            print(f"  {col}: {min_val:.2f} to {max_val:.2f} (mean: {mean_val:.2f})")
    
    # Temperature range reasonableness check
    if 'Global mean temperature' in scenario_df.columns: 
        temp_min = scenario_df['Global mean temperature'].min()
        temp_max = scenario_df['Global mean temperature'].max()
        if 0 <= temp_min <= 20 and 0 <= temp_max <= 20:
            print(f"✓ Pass: Temperature range is reasonable")
            checks_passed += 1
        else:
            print(f"✗ Fail: Abnormal temperature range ({temp_min:.2f} to {temp_max:.2f} °C)")
    
    # 5. Check temporal continuity
    scenarios = scenario_df['Scenario'].unique()
    time_continuous = True 
    problematic_scenarios = []
    
    for scenario in scenarios[:5]:  # Check first 5 scenarios
        years = scenario_df[scenario_df['Scenario'] == scenario]['Year'].values 
        if len(years) > 1:
            year_diff = np.diff(years) 
            if not np.all(year_diff == 1):
                time_continuous = False 
                problematic_scenarios.append(scenario) 
                break
    
    if time_continuous:
        print(f"✓ Pass: Time series are continuous")
        checks_passed += 1
    else:
        print(f"✗ Fail: Scenario {problematic_scenarios[0]} time series not continuous")
    
    print(f"\nQuality Check Summary: {checks_passed}/{total_checks} items passed")
    
    return checks_passed == total_checks

# ==================== 3. CVAE Model ====================
class ClimateCVAE(nn.Module):
    def __init__(self, input_dim, cond_dim=2, latent_dim=8):  # cond_dim changed to 2 for both temperature and CO2
        super().__init__()
        self.latent_dim = latent_dim 
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2)
        )
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        
        # Decoder - now outputs include temperature and CO2
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, input_dim + 2)  # +2 for temperature and CO2 output 
        )

    def encode(self, x, conditions):
        h = self.encoder(torch.cat((x, conditions), dim=1))
        return self.mu(h), self.logvar(h)  
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  
        return mu + eps * std 
    
    def decode(self, z, conditions):
        return self.decoder(torch.cat((z, conditions), dim=1))
    
    def forward(self, x, conditions):
        mu, logvar = self.encode(x, conditions)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, conditions), mu, logvar 

# ==================== 4. Training and Generation ====================
def train_model(x_norm, temp_norm, co2_norm, epochs=2000, batch_size=32):
    """Enhanced training process"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClimateCVAE(x_norm.shape[1]).to(device)  
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100)
    
    # Ensure batch size is at least 2
    batch_size = max(batch_size, 2)
    
    # Merge condition variables (temperature and CO2)
    conditions = np.column_stack([temp_norm, co2_norm])
    
    # Prepare training target - includes original features and temperature, CO2 
    target_data = np.column_stack([x_norm, temp_norm, co2_norm])
    
    dataset = torch.utils.data.TensorDataset(  
        torch.FloatTensor(x_norm).to(device),
        torch.FloatTensor(conditions).to(device),
        torch.FloatTensor(target_data).to(device)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_loss = float('inf')
    for epoch in range(1, epochs+1):
        model.train()  
        epoch_loss = 0 
        for x_batch, cond_batch, target_batch in dataloader:
            # Ensure batch size is at least 2
            if x_batch.size(0) < 2:
                continue 
                
            optimizer.zero_grad()  
            recon, mu, logvar = model(x_batch, cond_batch)
            
            # Composite loss function - now includes temperature and CO2 reconstruction 
            mse = nn.MSELoss()(recon, target_batch)
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  
            loss = mse + 0.1 * kld 
            
            loss.backward()  
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  
            epoch_loss += loss.item()  
        
        epoch_loss /= len(dataloader)
        scheduler.step(epoch_loss)  
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss 
            torch.save(model.state_dict(), 'CVAE_best_model.pth')  
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")  
    
    # Load best model - fix torch.load warnings
    model.load_state_dict(torch.load('CVAE_best_model.pth', weights_only=True)) 
    return model 

def get_ssp_scenarios(base_year, future_years, current_co2):
    """Define SSP scenario temperature and CO2 paths"""
    scenarios = {}
    
    n_years = len(future_years)
    
    # SSP1-1.9: Strong mitigation scenario, warming limited to 1.5°C 
    scenarios['SSP1-1.9'] = {
        'temperature': np.linspace(base_year, 1.5, n_years),
        'co2': np.concatenate([ 
            np.linspace(current_co2, 440, n_years//3),  # First rise to peak
            np.linspace(440, 430, n_years//3),          # Then decline 
            np.linspace(430, 420, n_years - 2*(n_years//3))  # Continue decline
        ])
    }
    
    # SSP1-2.6: Mitigation scenario, warming limited to 2.0°C 
    scenarios['SSP1-2.6'] = {
        'temperature': np.linspace(base_year, 2.0, n_years),
        'co2': np.concatenate([ 
            np.linspace(current_co2, 520, n_years//2),  # First rise
            np.linspace(520, 480, n_years - n_years//2) # Then slow decline 
        ])
    }
    
    # SSP2-4.5: Middle pathway
    scenarios['SSP2-4.5'] = {
        'temperature': np.linspace(base_year, 2.7, n_years),
        'co2': np.linspace(current_co2, 650, n_years)
    }
    
    # SSP3-7.0: Regional rivalry pathway 
    scenarios['SSP3-7.0'] = {
        'temperature': np.linspace(base_year, 3.6, n_years),
        'co2': np.linspace(current_co2, 800, n_years)
    }
    
    # SSP4-6.0: Inequality pathway
    scenarios['SSP4-6.0'] = {
        'temperature': np.linspace(base_year, 3.0, n_years),
        'co2': np.linspace(current_co2, 700, n_years)
    }
    
    # SSP5-8.5: Fossil-fuel intensive development pathway 
    scenarios['SSP5-8.5'] = {
        'temperature': np.linspace(base_year, 4.5, n_years),
        'co2': np.linspace(current_co2, 1000, n_years)
    }
    
    return scenarios 

def generate_scenarios(model, temp_path, co2_path, years, stats, num_scenarios=1000):
    """Physically constrained scenario generation - ensure complete temperature and CO2 data output"""
    device = next(model.parameters()).device   
    
    # Parameter unpacking 
    mean_x, std_x, mean_c, std_c, mean_co2, std_co2 = (
        stats['mean_x'], stats['std_x'],
        stats['mean_c'], stats['std_c'],
        stats['mean_co2'], stats['std_co2']
    )
    
    # Normalize condition variables
    temp_norm = (temp_path - mean_c) / std_c 
    co2_norm = (co2_path - mean_co2) / std_co2
    
    # Merge condition variables
    conditions_norm = np.column_stack([temp_norm, co2_norm])
    conditions_tensor = torch.FloatTensor(conditions_norm).unsqueeze(0).repeat(num_scenarios, 1, 1)
    conditions_tensor = conditions_tensor.reshape(num_scenarios * len(years), 2).to(device)
    
    # Generate data 
    model.eval()  
    with torch.no_grad():  
        z = torch.randn(num_scenarios * len(years), model.latent_dim).to(device)  
        gen_norm = model.decode(z, conditions_tensor).cpu().numpy()
    
    # Separate generated variables 
    # First n features are original variables, last 2 are temperature and CO2 
    n_features = len(stats['feature_names'])
    gen_x = gen_norm[:, :n_features] * std_x + mean_x 
    gen_temp = gen_norm[:, -2] * std_c + mean_c  # Second to last is temperature
    gen_co2 = gen_norm[:, -1] * std_co2 + mean_co2  # Last one is CO2
    
    # Assemble DataFrame - using model-generated temperature and CO2 data 
    scenario_df = pd.DataFrame({
        'Scenario': np.repeat(np.arange(num_scenarios), len(years)),
        'Year': np.tile(years, num_scenarios),
        'Global mean temperature': gen_temp,
        'Atmospheric Carbon Dioxide (CO2)': gen_co2
    })
    
    # Add other climate variables
    for i, var_name in enumerate(stats['feature_names']):
        scenario_df[var_name] = gen_x[:, i]
    
    # Verify output data completeness
    print(f"  Output verification: {len(scenario_df)} rows of data")
    print(f"  Temperature range: {scenario_df['Global mean temperature'].min():.2f} - {scenario_df['Global mean temperature'].max():.2f} °C")
    print(f"  CO2 range: {scenario_df['Atmospheric Carbon Dioxide (CO2)'].min():.1f} - {scenario_df['Atmospheric Carbon Dioxide (CO2)'].max():.1f} ppm")
    print(f"  Year range: {scenario_df['Year'].min()} - {scenario_df['Year'].max()}")
    
    return scenario_df 

def save_validation_statistics(validation_results, filename="CVAE_validation_statistics.csv"): 
    """Save validation statistics to CSV file"""
    # Convert validation results to DataFrame
    stats_data = []
    for var_name, metrics in validation_results.items(): 
        stats_data.append({ 
            'Variable': var_name,
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'R2': metrics['r2'],
            'Bias': metrics['bias']
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Calculate summary statistics
    summary_stats = {
        'Variable': ['Summary Statistics'],
        'RMSE': [stats_df['RMSE'].mean()],
        'MAE': [stats_df['MAE'].mean()],
        'R2': [stats_df['R2'].mean()],
        'Bias': [stats_df['Bias'].mean()]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Combine detailed and summary statistics
    combined_df = pd.concat([stats_df, summary_df], ignore_index=True)
    
    # Save to CSV
    combined_df.to_csv(filename, index=False)
    print(f"\nValidation statistics saved to: {filename}")
    
    # Print summary 
    print(f"\nSummary Statistics:")
    print(f"Average R²: {summary_df['R2'].iloc[0]:.4f}")
    print(f"Average RMSE: {summary_df['RMSE'].iloc[0]:.4f}")
    print(f"Average MAE: {summary_df['MAE'].iloc[0]:.4f}")
    print(f"Average Bias: {summary_df['Bias'].iloc[0]:.4f}")
    
    return combined_df

def historical_validation(model, df, stats, num_simulations=100):
    """Historical data validation and performance assessment"""
    print("\n" + "="*60)
    print("Historical Simulation Validation Results")
    print("="*60)
    
    device = next(model.parameters()).device  
    
    # Prepare historical data 
    years = df['Year'].values
    x_data = df.drop(columns=['Year', 'Global mean temperature', 'Atmospheric Carbon Dioxide (CO2)']).values
    actual_temp = df['Global mean temperature'].values
    actual_co2 = df['Atmospheric Carbon Dioxide (CO2)'].values 
    
    # Normalize condition variables
    temp_norm = (actual_temp - stats['mean_c']) / stats['std_c']
    co2_norm = (actual_co2 - stats['mean_co2']) / stats['std_co2']
    conditions_norm = np.column_stack([temp_norm, co2_norm])
    
    # Generate historical simulations 
    model.eval() 
    all_simulations = []
    all_temps = []
    all_co2s = []
    
    with torch.no_grad(): 
        for _ in range(num_simulations):
            z = torch.randn(len(years), model.latent_dim).to(device) 
            conditions_tensor = torch.FloatTensor(conditions_norm).to(device)
            gen_norm = model.decode(z, conditions_tensor).cpu().numpy()
            
            # Separate generated variables 
            n_features = len(stats['feature_names'])
            gen_data = gen_norm[:, :n_features] * stats['std_x'] + stats['mean_x']
            gen_temp = gen_norm[:, -2] * stats['std_c'] + stats['mean_c']
            gen_co2 = gen_norm[:, -1] * stats['std_co2'] + stats['mean_co2']
            
            all_simulations.append(gen_data) 
            all_temps.append(gen_temp) 
            all_co2s.append(gen_co2) 
    
    all_simulations = np.array(all_simulations) 
    all_temps = np.array(all_temps) 
    all_co2s = np.array(all_co2s) 
    
    # Calculate validation metrics 
    validation_results = {}
    feature_names = stats['feature_names']
    
    print(f"\nValidation Metrics (based on {num_simulations} simulations):")
    print("-" * 90)
    print(f"{'Variable':<30} {'RMSE':<12} {'MAE':<12} {'R²':<12} {'Bias':<12}")
    print("-" * 90)
    
    for i, var_name in enumerate(feature_names):
        actual_values = x_data[:, i]
        simulated_means = np.mean(all_simulations[:, :, i], axis=0)
        
        rmse = np.sqrt(mean_squared_error(actual_values, simulated_means))
        mae = mean_absolute_error(actual_values, simulated_means)
        r2 = r2_score(actual_values, simulated_means)
        bias = np.mean(simulated_means - actual_values)
        
        validation_results[var_name] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'bias': bias
        }
        
        print(f"{var_name:<30} {rmse:<12.4f} {mae:<12.4f} {r2:<12.4f} {bias:<12.4f}")
    
    # Validate temperature and CO2 
    temp_rmse = np.sqrt(mean_squared_error(actual_temp, np.mean(all_temps, axis=0)))
    temp_r2 = r2_score(actual_temp, np.mean(all_temps, axis=0))
    co2_rmse = np.sqrt(mean_squared_error(actual_co2, np.mean(all_co2s, axis=0)))
    co2_r2 = r2_score(actual_co2, np.mean(all_co2s, axis=0))
    
    print(f"\nTemperature and CO2 Validation:")
    print(f"Temperature - RMSE: {temp_rmse:.4f}, R²: {temp_r2:.4f}")
    print(f"CO2 - RMSE: {co2_rmse:.4f}, R²: {co2_r2:.4f}")
    
    # Add temperature and CO2 to validation results 
    validation_results['Global mean temperature'] = {
        'rmse': temp_rmse,
        'mae': mean_absolute_error(actual_temp, np.mean(all_temps, axis=0)),
        'r2': temp_r2,
        'bias': np.mean(np.mean(all_temps, axis=0) - actual_temp)
    }
    
    validation_results['Atmospheric Carbon Dioxide (CO2)'] = {
        'rmse': co2_rmse,
        'mae': mean_absolute_error(actual_co2, np.mean(all_co2s, axis=0)),
        'r2': co2_r2,
        'bias': np.mean(np.mean(all_co2s, axis=0) - actual_co2)
    }
    
    # Save validation statistics to CSV 
    stats_df = save_validation_statistics(validation_results)
    
    # Generate validation plots for all 22 variables
    generate_comprehensive_validation_plots(years, x_data, all_simulations, feature_names, validation_results, all_temps, all_co2s, actual_temp, actual_co2)
    
    return validation_results, all_simulations, stats_df 

def generate_comprehensive_validation_plots(years, actual_data, simulations, feature_names, validation_results, all_temps, all_co2s, actual_temp, actual_co2):
    """Generate comprehensive validation plots for all 22 variables"""
    print("\nGenerating comprehensive validation plots...")
    
    # Create output directory for plots 
    output_dir = "CVAE_validation_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Performance metrics comparison plot for all variables 
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare data for plotting 
    variables = list(validation_results.keys()) 
    r2_values = [validation_results[var]['r2'] for var in variables]
    rmse_values = [validation_results[var]['rmse'] for var in variables]
    mae_values = [validation_results[var]['mae'] for var in variables]
    bias_values = [validation_results[var]['bias'] for var in variables]
    
    # R² bar plot 
    ax1 = axes[0, 0]
    colors = ['green' if val > 0.8 else 'orange' if val > 0.6 else 'red' for val in r2_values]
    ax1.bar(range(len(variables)), r2_values, color=colors, alpha=0.7)
    ax1.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Variables') 
    ax1.set_ylabel('R² Score')
    ax1.set_title('R² Scores for All Variables')
    ax1.set_xticks(range(len(variables))) 
    ax1.set_xticklabels(variables, rotation=45, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # RMSE bar plot
    ax2 = axes[0, 1]
    ax2.bar(range(len(variables)), rmse_values, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Variables') 
    ax2.set_ylabel('RMSE') 
    ax2.set_title('RMSE for All Variables')
    ax2.set_xticks(range(len(variables))) 
    ax2.set_xticklabels(variables, rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # MAE bar plot
    ax3 = axes[1, 0]
    ax3.bar(range(len(variables)), mae_values, color='coral', alpha=0.7)
    ax3.set_xlabel('Variables') 
    ax3.set_ylabel('MAE') 
    ax3.set_title('MAE for All Variables')
    ax3.set_xticks(range(len(variables))) 
    ax3.set_xticklabels(variables, rotation=45, ha='right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Bias bar plot 
    ax4 = axes[1, 1]
    colors_bias = ['red' if val < 0 else 'blue' for val in bias_values]
    ax4.bar(range(len(variables)), bias_values, color=colors_bias, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Variables') 
    ax4.set_ylabel('Bias') 
    ax4.set_title('Bias for All Variables')
    ax4.set_xticks(range(len(variables))) 
    ax4.set_xticklabels(variables, rotation=45, ha='right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout() 
    plt.savefig(f'{output_dir}/CVAE_all_variables_metrics.png', dpi=300, bbox_inches='tight')
    plt.close() 
    print(f"  All variables metrics plot saved: {output_dir}/CVAE_all_variables_metrics.png") 
    
    # 2. Time series plots for all variables (4x6 grid for 22 variables + 2 for temperature/CO2)
    n_vars = len(feature_names)
    n_rows = (n_vars + 5) // 6  # 6 columns per row 
    
    fig, axes = plt.subplots(n_rows, 6, figsize=(24, 4*n_rows))
    axes = axes.flatten() 
    
    for i, var_name in enumerate(feature_names):
        if i < len(axes):
            idx = feature_names.index(var_name) 
            ax = axes[i]
            
            # Actual data 
            ax.plot(years, actual_data[:, idx], 'k-', linewidth=2, label='Observed')
            
            # Simulation mean
            sim_mean = np.mean(simulations[:, :, idx], axis=0)
            ax.plot(years, sim_mean, 'r--', linewidth=1.5, label='Simulation Mean')
            
            # Simulation uncertainty interval 
            sim_lower = np.percentile(simulations[:, :, idx], 2.5, axis=0)
            sim_upper = np.percentile(simulations[:, :, idx], 97.5, axis=0)
            ax.fill_between(years, sim_lower, sim_upper, alpha=0.3, color='red', label='95% CI')
            
            ax.set_title(f'{var_name}\n(R² = {validation_results[var_name]["r2"]:.3f})', fontsize=10)
            ax.set_xlabel('Year', fontsize=8)
            ax.set_ylabel(var_name, fontsize=8)
            ax.legend(fontsize=6) 
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=7)
    
    # Hide unused subplots 
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout() 
    plt.savefig(f'{output_dir}/CVAE_all_variables_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close() 
    print(f"  All variables time series plot saved: {output_dir}/CVAE_all_variables_timeseries.png") 
    
    # 3. Temperature and CO2 specific plots 
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Temperature time series
    ax1 = axes[0, 0]
    ax1.plot(years, actual_temp, 'k-', linewidth=2, label='Observed')
    ax1.plot(years, np.mean(all_temps, axis=0), 'r--', linewidth=1.5, label='Simulation Mean')
    temp_lower = np.percentile(all_temps, 2.5, axis=0)
    temp_upper = np.percentile(all_temps, 97.5, axis=0)
    ax1.fill_between(years, temp_lower, temp_upper, alpha=0.3, color='red', label='95% CI')
    ax1.set_title(f'Global Mean Temperature\n(R² = {validation_results["Global mean temperature"]["r2"]:.3f})')
    ax1.set_xlabel('Year') 
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend() 
    ax1.grid(True, alpha=0.3)
    
    # CO2 time series
    ax2 = axes[0, 1]
    ax2.plot(years, actual_co2, 'k-', linewidth=2, label='Observed')
    ax2.plot(years, np.mean(all_co2s, axis=0), 'b--', linewidth=1.5, label='Simulation Mean')
    co2_lower = np.percentile(all_co2s, 2.5, axis=0)
    co2_upper = np.percentile(all_co2s, 97.5, axis=0)
    ax2.fill_between(years, co2_lower, co2_upper, alpha=0.3, color='blue', label='95% CI')
    ax2.set_title(f'Atmospheric CO₂ Concentration\n(R² = {validation_results["Atmospheric Carbon Dioxide (CO2)"]["r2"]:.3f})')
    ax2.set_xlabel('Year') 
    ax2.set_ylabel('CO₂ (ppm)')
    ax2.legend() 
    ax2.grid(True, alpha=0.3)
    
    # Temperature-CO2 relationship
    ax3 = axes[1, 0]
    # Plot for first 5 simulations
    for i in range(min(5, len(all_temps))):
        ax3.scatter(all_co2s[i], all_temps[i], alpha=0.3, s=10, label=f'Sim {i+1}' if i == 0 else None)
    ax3.scatter(actual_co2, actual_temp, color='black', s=30, label='Observed', zorder=5)
    ax3.set_xlabel('CO₂ Concentration (ppm)')
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_title('Temperature-CO₂ Relationship')
    ax3.legend() 
    ax3.grid(True, alpha=0.3)
    
    # Residuals plot 
    ax4 = axes[1, 1]
    residuals = []
    var_names_residuals = []
    
    # Collect residuals for key variables 
    key_vars = ['Coal', 'Oil', 'Gas', 'Cement.emission', 'Ocean_Sink', 'land_Sink']
    for var in key_vars:
        if var in feature_names:
            idx = feature_names.index(var) 
            actual_vals = actual_data[:, idx]
            sim_means = np.mean(simulations[:, :, idx], axis=0)
            residuals.extend(actual_vals - sim_means)
            var_names_residuals.extend([var] * len(actual_vals))
    
    # Create box plot of residuals
    import pandas as pd
    residuals_df = pd.DataFrame({'Variable': var_names_residuals, 'Residual': residuals})
    sns.boxplot(x='Variable', y='Residual', data=residuals_df, ax=ax4)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax4.set_title('Residual Distribution for Key Variables')
    ax4.set_xlabel('Variable') 
    ax4.set_ylabel('Residual (Observed - Simulated)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout() 
    plt.savefig(f'{output_dir}/CVAE_key_variables_validation.png', dpi=300, bbox_inches='tight')
    plt.close() 
    print(f"  Key variables validation plot saved: {output_dir}/CVAE_key_variables_validation.png") 
    
    # 4. Heatmap of correlation between variables
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Select key variables for correlation matrix 
    key_vars_for_corr = ['Global mean temperature', 'Atmospheric Carbon Dioxide (CO2)', 
                        'Coal', 'Oil', 'Gas', 'Cement.emission', 'Ocean_Sink', 'land_Sink']
    
    # Create correlation matrix 
    corr_data = []
    for var in key_vars_for_corr:
        if var in validation_results:
            # For temperature and CO2, use actual data 
            if var == 'Global mean temperature':
                corr_data.append(actual_temp) 
            elif var == 'Atmospheric Carbon Dioxide (CO2)':
                corr_data.append(actual_co2) 
            else:
                # For other variables, use simulation means 
                if var in feature_names:
                    idx = feature_names.index(var) 
                    corr_data.append(np.mean(simulations[:, :, idx], axis=0))
    
    if len(corr_data) > 1:
        corr_matrix = np.corrcoef(corr_data) 
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add text annotations
        for i in range(corr_matrix.shape[0]): 
            for j in range(corr_matrix.shape[1]): 
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        # Set labels 
        ax.set_xticks(range(len(key_vars_for_corr))) 
        ax.set_yticks(range(len(key_vars_for_corr))) 
        ax.set_xticklabels(key_vars_for_corr, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(key_vars_for_corr, fontsize=9)
        ax.set_title('Correlation Matrix of Key Variables')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout() 
        plt.savefig(f'{output_dir}/CVAE_variable_correlation.png', dpi=300, bbox_inches='tight')
        plt.close() 
        print(f"  Variable correlation heatmap saved: {output_dir}/CVAE_variable_correlation.png") 
    
    print(f"\nAll validation plots saved to directory: {output_dir}/")

def generate_validation_plots(years, actual_data, simulations, feature_names, validation_results):
    """Generate validation plots for key variables"""
    # Select key variables for visualization
    key_variables = ['Coal', 'Oil', 'Gas', 'Cement.emission'] 
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten() 
    
    for i, var_name in enumerate(key_variables):
        if var_name in feature_names:
            idx = feature_names.index(var_name) 
            ax = axes[i]
            
            # Actual data 
            ax.plot(years, actual_data[:, idx], 'k-', linewidth=2, label='Observed')
            
            # Simulation mean
            sim_mean = np.mean(simulations[:, :, idx], axis=0)
            ax.plot(years, sim_mean, 'r--', linewidth=1.5, label='Simulation Mean')
            
            # Simulation uncertainty interval 
            sim_lower = np.percentile(simulations[:, :, idx], 2.5, axis=0)
            sim_upper = np.percentile(simulations[:, :, idx], 97.5, axis=0)
            ax.fill_between(years, sim_lower, sim_upper, alpha=0.3, color='red', label='95% Confidence Interval')
            
            ax.set_title(f'{var_name} - Historical Validation (R² = {validation_results[var_name]["r2"]:.3f})')
            ax.set_xlabel('Year') 
            ax.set_ylabel(var_name) 
            ax.legend() 
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout() 
    plt.savefig('CVAE_historical_validation.png', dpi=300, bbox_inches='tight')
    plt.close() 
    
    print(f"\nValidation plots saved: CVAE_historical_validation.png") 

def generate_scenario_summary_plot(ssp_scenarios, future_years):
    """Generate scenario overview plot"""
    plt.figure(figsize=(12, 8))
    
    # Temperature scenario plot 
    plt.subplot(2, 1, 1)
    for name, scenario in ssp_scenarios.items(): 
        plt.plot(future_years, scenario['temperature'], label=name, linewidth=2)
    plt.ylabel('Global Mean Temperature (°C)')
    plt.title('SSP Scenarios - Temperature Pathways')
    plt.legend() 
    plt.grid(True, alpha=0.3)
    
    # CO2 scenario plot 
    plt.subplot(2, 1, 2)
    for name, scenario in ssp_scenarios.items(): 
        plt.plot(future_years, scenario['co2'], label=name, linewidth=2)
    plt.ylabel('Atmospheric CO₂ Concentration (ppm)')
    plt.xlabel('Year') 
    plt.title('SSP Scenarios - CO₂ Pathways')
    plt.legend() 
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout() 
    plt.savefig('CVAE_ssp_scenarios_overview.png', dpi=300, bbox_inches='tight')
    plt.close() 
    
    print(f"Scenario overview plot saved: CVAE_ssp_scenarios_overview.png") 

# ==================== Main Process ====================
if __name__ == "__main__":
    # 1. Data loading 
    try:
        print("Loading data...")
        df, stats = load_data('GCB_1959_2023_data.xlsx')  
        print(f"Data loaded successfully (1959-{int(df['Year'].max())})")
        print(f"Variables included: {len(stats['feature_names'])} climate indicators")
        print(f"Variable list: {stats['feature_names']}")
        
        # Display basic statistics 
        print(f"\nBasic Statistics:")
        print(f"Temperature range: {df['Global mean temperature'].min():.2f} - {df['Global mean temperature'].max():.2f} °C")
        print(f"CO2 range: {df['Atmospheric Carbon Dioxide (CO2)'].min():.2f} - {df['Atmospheric Carbon Dioxide (CO2)'].max():.2f} ppm")
        
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        exit()

    # 2. Model training
    print("\nStarting model training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare training data
    x_data = df.drop(columns=['Year', 'Global mean temperature', 'Atmospheric Carbon Dioxide (CO2)']).values 
    x_norm = (x_data - stats['mean_x']) / stats['std_x']
    temp_norm = (df['Global mean temperature'].values - stats['mean_c']) / stats['std_c']
    co2_norm = (df['Atmospheric Carbon Dioxide (CO2)'].values - stats['mean_co2']) / stats['std_co2']
    
    model = train_model(x_norm, temp_norm, co2_norm, epochs=2000)
    print("Model training completed")

    # 3. Historical data validation 
    print("\nPerforming historical data validation...")
    validation_results, historical_simulations, stats_df = historical_validation(model, df, stats, num_simulations=100)

    # 4. Scenario generation
    future_years = np.arange(int(df['Year'].max())+1, 2101)
    
    # Get current baseline values 
    current_temp = df['Global mean temperature'].iloc[-1]
    current_co2 = df['Atmospheric Carbon Dioxide (CO2)'].iloc[-1]
    
    print(f"\nCurrent baseline values - Temperature: {current_temp:.2f}°C, CO2: {current_co2:.2f} ppm")
    
    # Define SSP scenarios
    ssp_scenarios = get_ssp_scenarios(current_temp, future_years, current_co2)
    
    # Generate scenario overview plot 
    generate_scenario_summary_plot(ssp_scenarios, future_years)
    
    print("\nGenerating scenario data...")
    for name, scenario_data in ssp_scenarios.items():  
        print(f"\nGenerating {name} scenario...")
        print(f"  Temperature pathway: {scenario_data['temperature'][0]:.2f}°C → {scenario_data['temperature'][-1]:.2f}°C")
        print(f"  CO2 pathway: {scenario_data['co2'][0]:.0f} ppm → {scenario_data['co2'][-1]:.0f} ppm")
        
        # Generate original scenario data
        scenario_df = generate_scenarios(model, 
                                       scenario_data['temperature'], 
                                       scenario_data['co2'], 
                                       future_years, 
                                       stats, 
                                       num_scenarios=1000)
        
        # Apply physical constraint post-processing
        scenario_df_constrained = apply_physical_constraints(scenario_df, stats)
        
        # Check data quality
        quality_passed = check_scenario_quality(scenario_df_constrained)
        
        # Use CVAE prefix for filenames 
        current_date = datetime.now().strftime('%Y%m%d')  
        
        # Save original data
        fname_raw = f"CVAE_scenario_{name}_{current_date}_raw.csv"  
        scenario_df.to_csv(fname_raw, index=False)
        print(f"  Original data saved: {fname_raw}")
        
        # Save constrained data
        fname_constrained = f"CVAE_scenario_{name}_{current_date}_constrained.csv"  
        scenario_df_constrained.to_csv(fname_constrained, index=False)
        print(f"  Constrained data saved: {fname_constrained}")
        
        # Verify CSV file contains temperature and CO2 data
        verify_df = pd.read_csv(fname_constrained) 
        print(f"  File verification - Temperature column exists: {'Global mean temperature' in verify_df.columns}") 
        print(f"  File verification - CO2 column exists: {'Atmospheric Carbon Dioxide (CO2)' in verify_df.columns}") 
        print(f"  File verification - Data rows: {len(verify_df)}")
        print(f"  File verification - Temperature data range: {verify_df['Global mean temperature'].min():.2f} - {verify_df['Global mean temperature'].max():.2f} °C")
        print(f"  File verification - CO2 data range: {verify_df['Atmospheric Carbon Dioxide (CO2)'].min():.1f} - {verify_df['Atmospheric Carbon Dioxide (CO2)'].max():.1f} ppm")
        
        # Display sample data for first few years 
        sample_data = scenario_df_constrained[scenario_df_constrained['Scenario'] == 0].head()
        print(f"  Sample data (Scenario 0):")
        print(f"    Years: {list(sample_data['Year'].values)}")
        print(f"    Temperature: {[f'{x:.2f}' for x in sample_data['Global mean temperature'].values]}")
        print(f"    CO2: {[f'{x:.1f}' for x in sample_data['Atmospheric Carbon Dioxide (CO2)'].values]}")

    # 5. Output validation summary
    print(f"\n{'='*60}")
    print("Model Validation Summary")
    print(f"{'='*60}")
    
    # Calculate averages (excluding temperature and CO2 for comparison with other models)
    r2_values = [validation_results[var]['r2'] for var in stats['feature_names']]
    rmse_values = [validation_results[var]['rmse'] for var in stats['feature_names']]
    
    avg_r2 = np.mean(r2_values) 
    avg_rmse = np.mean(rmse_values) 
    
    print(f"Average R² (20 climate variables): {avg_r2:.4f}")
    print(f"Average RMSE (20 climate variables): {avg_rmse:.4f}")
    print(f"Temperature R²: {validation_results['Global mean temperature']['r2']:.4f}")
    print(f"CO2 R²: {validation_results['Atmospheric Carbon Dioxide (CO2)']['r2']:.4f}")
    print(f"Validation completed at: {datetime.now().strftime('%H:%M')}") 
    
    # Model performance rating
    if avg_r2 > 0.8:
        performance = "Excellent"
    elif avg_r2 > 0.6:
        performance = "Good"
    elif avg_r2 > 0.4:
        performance = "Fair"
    else:
        performance = "Needs Improvement"
    
    print(f"Model performance rating: {performance}")

    print(f"\nAll tasks completed @ {datetime.now().strftime('%H:%M')}") 
    print(f"Generated {len(ssp_scenarios)} SSP scenarios, time range: {future_years[0]}-{future_years[-1]}")
    print(f"Each CSV file contains complete Global mean temperature and Atmospheric Carbon Dioxide (CO2) data")
    print(f"Physical constraint post-processing successfully applied to all generated data")
    print(f"Comprehensive validation plots and statistics saved")