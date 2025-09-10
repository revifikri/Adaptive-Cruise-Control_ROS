#!/usr/bin/env python3
"""
Nav2 DWB Thesis Metrics Analyzer
Load and analyze the collected metrics data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from datetime import datetime

class Nav2MetricsAnalyzer:
    def __init__(self, metrics_dir="./nav2_thesis_metrics"):
        self.metrics_dir = metrics_dir
        self.realtime_data = None
        self.summary_data = None
        self.path_data = None
        
    def load_latest_experiment(self):
        """Load the most recent experiment data"""
        # Find latest files
        realtime_files = glob.glob(f"{self.metrics_dir}/*_realtime_data.csv")
        summary_files = glob.glob(f"{self.metrics_dir}/*_summary_metrics.csv")
        path_files = glob.glob(f"{self.metrics_dir}/*_path_comparison.csv")
        
        if not realtime_files:
            print("No experiment data found!")
            return False
        
        # Load most recent files
        latest_realtime = max(realtime_files, key=os.path.getctime)
        latest_summary = max(summary_files, key=os.path.getctime) if summary_files else None
        latest_path = max(path_files, key=os.path.getctime) if path_files else None
        
        print(f"Loading experiment data:")
        print(f"  Real-time: {os.path.basename(latest_realtime)}")
        
        try:
            self.realtime_data = pd.read_csv(latest_realtime)
            if latest_summary:
                self.summary_data = pd.read_csv(latest_summary)
                print(f"  Summary: {os.path.basename(latest_summary)}")
            if latest_path:
                self.path_data = pd.read_csv(latest_path)
                print(f"  Paths: {os.path.basename(latest_path)}")
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def load_all_experiments(self):
        """Load all experiment data for comparison"""
        summary_files = glob.glob(f"{self.metrics_dir}/*_summary_metrics.csv")
        
        if not summary_files:
            print("No experiment summary files found!")
            return None
        
        all_experiments = []
        for file in summary_files:
            try:
                df = pd.read_csv(file)
                all_experiments.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if all_experiments:
            combined_df = pd.concat(all_experiments, ignore_index=True)
            print(f"Loaded {len(combined_df)} experiments")
            return combined_df
        return None
    
    def generate_thesis_summary(self):
        """Generate comprehensive summary for thesis"""
        if self.realtime_data is None:
            print("No data loaded! Call load_latest_experiment() first.")
            return
        
        print("\n" + "="*80)
        print("NAV2 DWB THESIS METRICS SUMMARY")
        print("="*80)
        
        # Primary Thesis Metrics
        print("\n🎯 PRIMARY THESIS METRICS:")
        
        if self.summary_data is not None:
            distance = self.summary_data['total_distance_traveled_m'].iloc[0]
            duration = self.summary_data['mission_duration_s'].iloc[0]
            error = self.summary_data['goal_reach_error_m'].iloc[0]
            cpu_load = self.summary_data['avg_cpu_usage_percent'].iloc[0]
            
            print(f"   Distance Traveled:     {distance:.2f} m")
            print(f"   Mission Duration:      {duration:.2f} s")
            print(f"   Goal Reach Error:      {error:.2f} m")
            print(f"   Computational Load:    {cpu_load:.1f} %")
        
        # Real-time derived metrics
        avg_speed = self.realtime_data['speed_kmh'].mean()
        max_speed = self.realtime_data['speed_kmh'].max()
        avg_path_error = self.realtime_data['path_error'].mean()
        max_path_error = self.realtime_data['path_error'].max()
        
        print("\n🚀 PERFORMANCE METRICS:")
        print(f"   Average Speed:         {avg_speed:.2f} km/h")
        print(f"   Maximum Speed:         {max_speed:.2f} km/h")
        print(f"   Average Path Error:    {avg_path_error:.2f} m")
        print(f"   Maximum Path Error:    {max_path_error:.2f} m")
        print(f"   Data Points Collected: {len(self.realtime_data)}")
        
        # System Performance
        avg_cpu = self.realtime_data['cpu_usage'].mean()
        max_cpu = self.realtime_data['cpu_usage'].max()
        avg_memory = self.realtime_data['memory_usage'].mean()
        avg_control_freq = self.realtime_data['control_frequency'].mean()
        
        print("\n💻 SYSTEM PERFORMANCE:")
        print(f"   Average CPU Usage:     {avg_cpu:.1f} %")
        print(f"   Maximum CPU Usage:     {max_cpu:.1f} %")
        print(f"   Average Memory Usage:  {avg_memory:.1f} %")
        print(f"   Control Frequency:     {avg_control_freq:.1f} Hz")
        
        # Efficiency Metrics
        if self.summary_data is not None:
            efficiency = self.summary_data['path_efficiency_ratio'].iloc[0]
            print(f"\n📈 EFFICIENCY METRICS:")
            print(f"   Path Efficiency Ratio: {efficiency:.3f}")
        
        return {
            'distance_traveled_m': distance if self.summary_data is not None else 0,
            'mission_duration_s': duration if self.summary_data is not None else 0,
            'goal_reach_error_m': error if self.summary_data is not None else 0,
            'computational_load_percent': cpu_load if self.summary_data is not None else 0,
            'avg_speed_kmh': avg_speed,
            'max_speed_kmh': max_speed,
            'avg_path_error_m': avg_path_error,
            'max_path_error_m': max_path_error
        }
    
    def plot_mission_overview(self, save_plots=True):
        """Create comprehensive mission overview plots"""
        if self.realtime_data is None:
            print("No data loaded!")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Nav2 DWB Mission Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Speed over time
        axes[0, 0].plot(self.realtime_data['timestamp'], self.realtime_data['speed_kmh'], 'b-', linewidth=2)
        axes[0, 0].set_title('Vehicle Speed Over Time')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Speed (km/h)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distance to goal
        axes[0, 1].plot(self.realtime_data['timestamp'], self.realtime_data['distance_to_goal'], 'r-', linewidth=2)
        axes[0, 1].set_title('Distance to Goal Over Time')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Distance (m)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Path following error
        axes[0, 2].plot(self.realtime_data['timestamp'], self.realtime_data['path_error'], 'g-', linewidth=2)
        axes[0, 2].set_title('Path Following Error')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Error (m)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. CPU Usage
        axes[1, 0].plot(self.realtime_data['timestamp'], self.realtime_data['cpu_usage'], 'm-', linewidth=2)
        axes[1, 0].set_title('CPU Usage Over Time')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('CPU Usage (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Control frequency
        axes[1, 1].plot(self.realtime_data['timestamp'], self.realtime_data['control_frequency'], 'c-', linewidth=2)
        axes[1, 1].set_title('Control Frequency')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Frequency (Hz)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Vehicle trajectory
        if self.path_data is not None:
            # Plot planned vs actual path
            axes[1, 2].plot(self.path_data['planned_x'], self.path_data['planned_y'], 'b--', 
                           linewidth=2, label='Planned Path', alpha=0.7)
            axes[1, 2].plot(self.path_data['actual_x'], self.path_data['actual_y'], 'r-', 
                           linewidth=2, label='Actual Path')
            axes[1, 2].set_title('Path Comparison')
            axes[1, 2].set_xlabel('X Position (m)')
            axes[1, 2].set_ylabel('Y Position (m)')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].axis('equal')
        else:
            # Plot trajectory from real-time data
            axes[1, 2].plot(self.realtime_data['position_x'], self.realtime_data['position_y'], 'r-', linewidth=2)
            axes[1, 2].set_title('Vehicle Trajectory')
            axes[1, 2].set_xlabel('X Position (m)')
            axes[1, 2].set_ylabel('Y Position (m)')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].axis('equal')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"{self.metrics_dir}/nav2_analysis_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved: {plot_filename}")
        
        plt.show()
    
    def create_thesis_table(self):
        """Create a formatted table for thesis presentation"""
        summary = self.generate_thesis_summary()
        
        if not summary:
            return None
        
        # Create DataFrame for thesis table
        thesis_data = {
            'Metric': [
                'Distance Traveled',
                'Mission Duration', 
                'Goal Reach Error',
                'Computational Load',
                'Average Speed',
                'Maximum Speed',
                'Average Path Error',
                'Maximum Path Error'
            ],
            'Value': [
                f"{summary['distance_traveled_m']:.2f}",
                f"{summary['mission_duration_s']:.2f}",
                f"{summary['goal_reach_error_m']:.2f}",
                f"{summary['computational_load_percent']:.1f}",
                f"{summary['avg_speed_kmh']:.2f}",
                f"{summary['max_speed_kmh']:.2f}",
                f"{summary['avg_path_error_m']:.2f}",
                f"{summary['max_path_error_m']:.2f}"
            ],
            'Unit': [
                'm',
                's',
                'm', 
                '%',
                'km/h',
                'km/h',
                'm',
                'm'
            ]
        }
        
        df_thesis = pd.DataFrame(thesis_data)
        
        # Save thesis table
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        thesis_file = f"{self.metrics_dir}/thesis_table_{timestamp}.csv"
        df_thesis.to_csv(thesis_file, index=False)
        
        print("\n📋 THESIS TABLE:")
        print(df_thesis.to_string(index=False))
        print(f"\n💾 Thesis table saved: {thesis_file}")
        
        return df_thesis
    
    def export_for_latex(self):
        """Export data in LaTeX table format"""
        thesis_df = self.create_thesis_table()
        if thesis_df is None:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        latex_file = f"{self.metrics_dir}/thesis_latex_{timestamp}.tex"
        
        latex_content = """\\begin{table}[h]
\\centering
\\caption{Nav2 DWB Performance Metrics in CARLA Simulation}
\\label{tab:nav2_metrics}
\\begin{tabular}{|l|c|c|}
\\hline
\\textbf{Metric} & \\textbf{Value} & \\textbf{Unit} \\\\
\\hline
"""
        
        for _, row in thesis_df.iterrows():
            latex_content += f"{row['Metric']} & {row['Value']} & {row['Unit']} \\\\\n"
        
        latex_content += """\\hline
\\end{tabular}
\\end{table}"""
        
        with open(latex_file, 'w') as f:
            f.write(latex_content)
        
        print(f"📄 LaTeX table saved: {latex_file}")


def main():
    """Main analysis function"""
    analyzer = Nav2MetricsAnalyzer()
    
    print("Nav2 DWB Metrics Analyzer")
    print("="*40)
    
    # Load latest experiment
    if not analyzer.load_latest_experiment():
        return
    
    # Generate comprehensive analysis
    analyzer.generate_thesis_summary()
    
    # Create thesis table
    analyzer.create_thesis_table()
    
    # Export LaTeX format
    analyzer.export_for_latex()
    
    # Create plots
    print("\nGenerating plots...")
    analyzer.plot_mission_overview()
    
    print("\n✅ Analysis complete!")
    print(f"📁 All outputs saved in: {analyzer.metrics_dir}")


if __name__ == '__main__':
    main()
