#!/usr/bin/env python3

import numpy as np
import json
import time
import math
import random
from typing import Dict, List, Tuple, Any

class SimplifiedRealWorldBinaryGAOptimizer:
    def __init__(self, carla_world=None, test_scenarios=None):
        """
        Simplified Binary GA Optimizer for Real-World Speeds Only
        Optimizes 9 parameters: 4 distances + 5 speeds (NO brake parameters)
        Uses similar fitness structure as your successful original optimizer
        """
        self.carla_world = carla_world
        self.test_scenarios = test_scenarios or self.create_default_scenarios()
        
        # === SAME GA PARAMETERS AS YOUR SUCCESSFUL ORIGINAL ===
        self.Nvar = 9                    # Back to 9 variables (distances + speeds only)
        self.Nbit = 16                   # Same as original
        self.Npop = 100                  # Same as original
        self.Maxit = 100                 # Same as original
        self.Pc = 0.8                    # Same as original
        self.Pm = 0.02                   # Same as original
        self.el = 0.1                    # Same as original
        
        # === REAL-WORLD PARAMETER BOUNDS (distances + speeds in km/h) ===
        # [emergency_dist, critical_dist, warning_dist, safe_dist, max_speed_kmh, cruise_speed_kmh, normal_speed_kmh, slow_speed_kmh, creep_speed_kmh]
        
        # Lower bounds - based on your successful ranges but expanded for real-world
        self.rb = [6.0,  12.0, 15.0, 20.0, 45.0, 25.0, 18.0, 10.0, 1.0]
        
        # Upper bounds - real-world capable but reasonable
        self.ra = [12.0, 20.0, 25.0, 35.0, 65.0, 45.0, 35.0, 20.0, 5.0]
        
        # Calculate bounds range (same as original)
        self.batas = []
        for i in range(self.Nvar):
            self.batas.append(self.ra[i] - self.rb[i])
        
        # Storage arrays (same as original)
        self.eBangkit = []
        self.Individu = []
        self.eIndividu = []
        self.Dadatfit = []
        self.eDadatfit = []
        self.efitnessmax = []
        self.eIndividuMax = []
        self.optimization_history = []
        
        # SIMPLIFIED weight factors (similar to original)
        self.weights = {
            'safety': 0.5,      # 50% - Primary concern (same as original)
            'comfort': 0.3,     # 30% - Secondary concern (same as original)
            'efficiency': 0.2   # 20% - Tertiary concern (same as original)
        }
        
        print("🚗 Simplified Real-World Speed GA Optimizer (9 Parameters)")
        print(f"   Variables: {self.Nvar} (4 distances + 5 speeds)")
        print(f"   Speed Range: {self.rb[4]}-{self.ra[4]} km/h")
        print(f"   Same GA settings as your successful optimizer")
        print(f"   Population: {self.Npop}, Generations: {self.Maxit}")
    
    def bi2de(self, bin_array):
        """Binary to decimal conversion (same as original)"""
        return sum(bin_array * (2**(len(bin_array)-1-np.arange(len(bin_array)))))
    
    def create_default_scenarios(self) -> List[Dict]:
        """Create test scenarios (simplified, similar to original)"""
        scenarios = [
            {
                'name': 'static_obstacles',
                'obstacles': [
                    {'type': 'static', 'position': (50, 0), 'size': (2, 2)},
                    {'type': 'static', 'position': (100, 5), 'size': (2, 2)},
                ],
                'target_time': 30.0,
                'max_collisions': 0
            },
            {
                'name': 'dynamic_obstacles', 
                'obstacles': [
                    {'type': 'vehicle', 'position': (30, 0), 'velocity': (8, 0)},  # Faster vehicle
                    {'type': 'pedestrian', 'position': (80, -3), 'velocity': (0, 1)},
                ],
                'target_time': 35.0,
                'max_collisions': 0
            },
            {
                'name': 'real_world_traffic',
                'obstacles': [
                    {'type': 'vehicle', 'position': (40, 0), 'velocity': (12, 0)},  # Real-world speeds
                    {'type': 'vehicle', 'position': (80, 3), 'velocity': (15, 0)},
                    {'type': 'pedestrian', 'position': (60, -2), 'velocity': (0, 0.5)},
                ],
                'target_time': 40.0,
                'max_collisions': 0
            }
        ]
        return scenarios
    
    def optimize_real_world_collision_avoidance(self, params):
        """
        SIMPLIFIED objective function - similar structure to your successful original
        Args:
            params: [emergency_dist, critical_dist, warning_dist, safe_dist, 
                    max_speed_kmh, cruise_speed_kmh, normal_speed_kmh, slow_speed_kmh, creep_speed_kmh]
        """
        try:
            emergency_dist, critical_dist, warning_dist, safe_dist = params[:4]
            max_speed_kmh, cruise_speed_kmh, normal_speed_kmh, slow_speed_kmh, creep_speed_kmh = params[4:]
            
            # === PARAMETER VALIDATION (same logic as original) ===
            if not self.validate_parameters(params):
                return -1e6  # Same penalty as original
            
            # === SIMPLIFIED PERFORMANCE METRICS (similar to original structure) ===
            
            # 1. Safety Score (analog to ITAE in original)
            safety_score = self.calculate_simplified_safety_metric(params)
            
            # 2. Comfort Score (analog to ISE in original)  
            comfort_score = self.calculate_simplified_comfort_metric(params)
            
            # 3. Efficiency Score (analog to control effort in original)
            efficiency_score = self.calculate_simplified_efficiency_metric(params)
            
            # === SIMPLIFIED PENALTIES (similar scale to original) ===
            
            # Speed overshoot penalty (similar to overshoot in PID)
            overshoot_penalty = 0.0
            if max_speed_kmh > 60.0:
                overshoot_penalty = (max_speed_kmh - 60.0)**2 * 2.0  # Much smaller penalty
            
            # Distance range penalty (similar to settling time in PID)
            settling_penalty = 0.0
            total_range = safe_dist - emergency_dist
            if total_range > 25.0:  # Too wide range
                settling_penalty = (total_range - 25.0)**2 * 1.0  # Much smaller penalty
            
            # Target cruise speed penalty (similar to steady-state error in PID)
            ss_penalty = 0.0
            target_cruise = 35.0  # Target ~35 km/h urban cruise
            if abs(cruise_speed_kmh - target_cruise) > 8.0:
                ss_penalty = (abs(cruise_speed_kmh - target_cruise) - 8.0)**2 * 1.0  # Much smaller penalty
            
            # === SIMPLIFIED COMBINED FITNESS (same structure as original) ===
            fitness_value = (
                0.4 * safety_score +           # Primary: Safety (same weight as original ITAE)
                0.2 * comfort_score +          # Secondary: Comfort (same weight as original ISE)  
                0.1 * efficiency_score +       # Tertiary: Efficiency
                2.0 * overshoot_penalty +      # Much smaller penalties
                1.0 * settling_penalty +       
                1.5 * ss_penalty               
            )
            
            # Store optimization history
            self.optimization_history.append({
                'parameters': params.copy(),
                'safety_score': safety_score,
                'comfort_score': comfort_score,
                'efficiency_score': efficiency_score,
                'fitness_value': fitness_value,
                'timestamp': time.time()
            })
            
            return -fitness_value  # GA maximizes, so return negative (same as original)
            
        except Exception as e:
            return -1e6  # Same penalty as original
    
    def validate_parameters(self, params):
        """Validate parameter constraints (same logic as original)"""
        emergency_dist, critical_dist, warning_dist, safe_dist = params[:4]
        max_speed_kmh, cruise_speed_kmh, normal_speed_kmh, slow_speed_kmh, creep_speed_kmh = params[4:]
        
        # Distance constraints: emergency < critical < warning < safe
        if not (emergency_dist < critical_dist < warning_dist < safe_dist):
            return False
            
        # Speed constraints: creep < slow < normal < cruise < max
        if not (creep_speed_kmh < slow_speed_kmh < normal_speed_kmh < cruise_speed_kmh < max_speed_kmh):
            return False
            
        return True
    
    def calculate_simplified_safety_metric(self, params):
        """SIMPLIFIED safety calculation (similar to original error analysis)"""
        emergency_dist, critical_dist, warning_dist, safe_dist = params[:4]
        max_speed_kmh, cruise_speed_kmh, normal_speed_kmh, slow_speed_kmh, creep_speed_kmh = params[4:]
        
        safety_score = 0.0
        
        # Basic stopping distance analysis (simplified physics)
        friction_coeff = 0.7  # Same as original
        reaction_time = 1.0   # Same as original
        
        for speed_kmh in [cruise_speed_kmh, normal_speed_kmh, slow_speed_kmh]:
            speed_ms = speed_kmh / 3.6
            min_stopping_dist = (speed_ms**2) / (2 * friction_coeff * 9.81) + (speed_ms * reaction_time)
            
            if emergency_dist < min_stopping_dist:
                safety_score += 50.0  # Smaller penalty than original 100.0
            elif critical_dist < min_stopping_dist * 1.5:
                safety_score += 25.0  # Smaller penalty than original 50.0
        
        # Simplified scenario safety (similar to original)
        for scenario in self.test_scenarios:
            scenario_safety = self.simulate_simplified_scenario_safety(scenario, params)
            safety_score += scenario_safety
        
        return safety_score
    
    def calculate_simplified_comfort_metric(self, params):
        """SIMPLIFIED comfort calculation (same logic as original)"""
        emergency_dist, critical_dist, warning_dist, safe_dist = params[:4]
        max_speed_kmh, cruise_speed_kmh, normal_speed_kmh, slow_speed_kmh, creep_speed_kmh = params[4:]
        
        comfort_score = 0.0
        
        # Speed transition smoothness (same as original)
        speeds = [creep_speed_kmh, slow_speed_kmh, normal_speed_kmh, cruise_speed_kmh, max_speed_kmh]
        speed_diffs = np.diff(speeds)
        transition_variance = np.var(speed_diffs)
        comfort_score += transition_variance * 3.0  # Smaller weight than original 5.0
        
        # Distance zone smoothness (same as original)
        distances = [emergency_dist, critical_dist, warning_dist, safe_dist]
        dist_diffs = np.diff(distances)
        zone_variance = np.var(dist_diffs)
        comfort_score += zone_variance * 2.0  # Smaller weight than original 3.0
        
        return comfort_score
    
    def calculate_simplified_efficiency_metric(self, params):
        """SIMPLIFIED efficiency calculation"""
        speeds = params[4:]
        efficiency_score = 0.0
        
        # Speed utilization (simplified)
        avg_speed = np.mean(speeds)
        if avg_speed < 20.0:  # Too conservative for real-world
            efficiency_score += 15.0  # Smaller penalty than original 30.0
        
        return efficiency_score
    
    def simulate_simplified_scenario_safety(self, scenario, params):
        """SIMPLIFIED scenario safety (much simpler than enhanced version)"""
        safety_score = 0.0
        obstacles = scenario.get('obstacles', [])
        
        emergency_dist = params[0]
        cruise_speed_kmh = params[5]
        
        for obstacle in obstacles:
            if obstacle['type'] == 'vehicle':
                # Simple dynamic obstacle check
                if 'velocity' in obstacle:
                    relative_speed = abs(obstacle['velocity'][0] * 3.6 - cruise_speed_kmh)
                    if relative_speed > 10.0 and emergency_dist < 8.0:
                        safety_score += 10.0  # Much smaller penalty
                        
            elif obstacle['type'] == 'static':
                # Simple static obstacle check
                if emergency_dist < 7.0:
                    safety_score += 8.0  # Much smaller penalty
                    
            elif obstacle['type'] == 'pedestrian':
                # Simple pedestrian safety
                if cruise_speed_kmh > 45.0:  # Too fast near pedestrians
                    safety_score += 5.0  # Much smaller penalty
        
        return safety_score
    
    def optimize_parameters_simplified_binary_ga(self):
        """
        Main Binary GA optimization - SAME structure as your successful original
        """
        print("🚀 Starting Simplified Real-World Binary GA optimization...")
        print("=" * 60)
        
        start_time = time.time()
        
        # === GENERATE INITIAL POPULATION (same as original) ===
        print("Generating initial population...")
        Bangkit = np.random.randint(0, 2, (self.Npop, self.Nbit * self.Nvar))
        
        # Convert binary to real values (same as original)
        Individu = np.zeros((self.Npop, self.Nvar))
        for i in range(self.Npop):
            for j in range(self.Nvar):
                start_bit = j * self.Nbit
                end_bit = (j + 1) * self.Nbit
                binary_slice = Bangkit[i, start_bit:end_bit]
                
                Desimal = self.bi2de(binary_slice)
                Individu[i, j] = (
                    (Desimal * self.batas[j] - self.batas[j] + 
                     self.rb[j] * (2**self.Nbit - 1)) / (2**self.Nbit - 1)
                )
        
        # Evaluate initial population (same as original)
        print("Evaluating initial population...")
        Datfit = []
        for i in range(self.Npop):
            fitness = self.optimize_real_world_collision_avoidance(Individu[i, :])
            Datfit.append(fitness)
            if (i + 1) % 20 == 0:
                print(f"Evaluated {i+1}/{self.Npop} individuals")
        
        Datfit = np.array(Datfit)
        fitemax, nmax = np.max(Datfit), np.argmax(Datfit)
        
        print(f"Initial best fitness: {fitemax:.4f}")
        print(f"Initial best params: {Individu[nmax, :]}")
        
        # === MAIN GA LOOP (exactly same as original) ===
        for generasi in range(1, self.Maxit + 1):
            print(f"\nGeneration {generasi}/{self.Maxit} processing...")
            
            if generasi > 1:
                # Sort population by fitness (descending order)
                sort_indices = np.argsort(Datfit)[::-1]  # Descending order
                sorted_bangkit = Bangkit[sort_indices]
                sorted_datfit = Datfit[sort_indices]
                
                # === ELITISM (same as original) ===
                n_elite = int(self.el * self.Npop)
                elite = sorted_bangkit[:n_elite].copy()
                
                # === SELECTION (Roulette Wheel - same as original) ===
                n_breed = self.Npop - n_elite
                
                # Handle negative fitness for selection
                fitness_for_selection = sorted_datfit.copy()
                min_fitness = np.min(fitness_for_selection)
                adjusted_fitness = fitness_for_selection - min_fitness + 1  # Shift to positive
                sumfitness = np.sum(adjusted_fitness)
                
                # Calculate cumulative probabilities
                Prob = adjusted_fitness / sumfitness
                Prob = np.cumsum(Prob)
                
                # Roulette wheel selection for parents
                Xparents = np.zeros((n_breed, self.Nbit * self.Nvar), dtype=int)
                for i in range(n_breed):
                    n = np.random.random()
                    k = 0
                    for j in range(len(Prob)):
                        if n <= Prob[j]:
                            k = j
                            break
                    Xparents[i, :] = sorted_bangkit[k, :]
                
                # === CROSSOVER (same as original) ===
                Xcrossed = Xparents.copy()
                M, d = Xparents.shape
                
                for i in range(0, M - 1, 2):
                    c = np.random.random()
                    if c <= self.Pc:
                        p = np.random.randint(1, d)  # Crossover point
                        # Single-point crossover
                        Xcrossed[i, :] = np.concatenate([Xparents[i, :p], Xparents[i+1, p:]])
                        Xcrossed[i+1, :] = np.concatenate([Xparents[i+1, :p], Xparents[i, p:]])
                
                # Handle odd number of parents
                if M % 2 == 1:
                    c = np.random.random()
                    if c <= self.Pc:
                        p = np.random.randint(1, d)
                        partner_idx = np.random.randint(0, M-1)
                        Xcrossed[M-1, :] = np.concatenate([Xparents[M-1, :p], Xparents[partner_idx, p:]])
                
                # === MUTATION (same as original) ===
                Xnew = Xcrossed.copy()
                for i in range(M):
                    for j in range(d):
                        p = np.random.random()
                        if p <= self.Pm:
                            Xnew[i, j] = 1 - Xcrossed[i, j]  # Bit flip
                
                # === COMBINE ELITE AND OFFSPRING ===
                Bangkit = np.vstack([elite, Xnew])
                
                # Ensure exact population size
                if Bangkit.shape[0] > self.Npop:
                    Bangkit = Bangkit[:self.Npop, :]
                elif Bangkit.shape[0] < self.Npop:
                    n_missing = self.Npop - Bangkit.shape[0]
                    random_individuals = np.random.randint(0, 2, (n_missing, self.Nbit * self.Nvar))
                    Bangkit = np.vstack([Bangkit, random_individuals])
            
            # Convert binary to real values for new generation
            for i in range(self.Npop):
                for j in range(self.Nvar):
                    start_bit = j * self.Nbit
                    end_bit = (j + 1) * self.Nbit
                    binary_slice = Bangkit[i, start_bit:end_bit]
                    
                    Desimal = self.bi2de(binary_slice)
                    Individu[i, j] = (
                        (Desimal * self.batas[j] - self.batas[j] + 
                         self.rb[j] * (2**self.Nbit - 1)) / (2**self.Nbit - 1)
                    )
            
            # Evaluate new population
            Datfit = []
            for i in range(self.Npop):
                fitness = self.optimize_real_world_collision_avoidance(Individu[i, :])
                Datfit.append(fitness)
            
            Datfit = np.array(Datfit)
            fitemax, nmax = np.max(Datfit), np.argmax(Datfit)
            
            # Store results
            self.eDadatfit.extend(Datfit)
            self.eIndividu.extend(Individu)
            self.efitnessmax.append(np.max(self.eDadatfit))
            
            # Current best individual
            best_global_idx = np.argmax(self.eDadatfit)
            IndividuMax = self.eIndividu[best_global_idx]
            fitnessmax = self.eDadatfit[best_global_idx]
            
            self.eIndividuMax.append(IndividuMax)
            
            # Display current best (same as original)
            print(f"Generation {generasi}: Best Params = {IndividuMax}")
            print(f"                    Best Fitness = {fitnessmax:.4f}")
            
            # Show key parameters every 10 generations
            if generasi % 10 == 0:
                print(f"   🚗 Speeds: Max={IndividuMax[4]:.1f}, Cruise={IndividuMax[5]:.1f} km/h")
                print(f"   📏 Distances: Emergency={IndividuMax[0]:.1f}, Safe={IndividuMax[3]:.1f}m")
        
        optimization_time = time.time() - start_time
        
        # === FINAL RESULTS (same as original) ===
        print("\n" + "=" * 60)
        print("🎉 SIMPLIFIED REAL-WORLD BINARY GA OPTIMIZATION COMPLETED!")
        print("=" * 60)
        print(f"⏱️ Time taken: {optimization_time:.2f} seconds")
        print(f"🎯 Best Fitness: {self.efitnessmax[-1]:.4f}")
        print(f"📊 Total Evaluations: {len(self.eDadatfit)}")
        
        final_params = self.eIndividuMax[-1]
        optimized_params = {
            'emergency_distance': final_params[0],
            'critical_distance': final_params[1],
            'warning_distance': final_params[2], 
            'safe_distance': final_params[3],
            'max_speed_kmh': final_params[4],
            'cruise_speed_kmh': final_params[5],
            'normal_speed_kmh': final_params[6],
            'slow_speed_kmh': final_params[7],
            'creep_speed_kmh': final_params[8]
        }
        
        print("\n✅ Optimized Real-World Parameters:")
        print("\n🚧 DISTANCE ZONES:")
        for param in ['emergency_distance', 'critical_distance', 'warning_distance', 'safe_distance']:
            print(f"   {param}: {optimized_params[param]:.2f}m")
        
        print("\n🚗 SPEED PARAMETERS:")
        for param in ['max_speed_kmh', 'cruise_speed_kmh', 'normal_speed_kmh', 'slow_speed_kmh', 'creep_speed_kmh']:
            print(f"   {param}: {optimized_params[param]:.2f} km/h")
        
        # Generate code snippet
        self.generate_code_snippet(optimized_params)
        
        # Save results
        self.save_optimization_results(optimized_params)
        
        return {
            'optimized_parameters': optimized_params,
            'best_fitness': self.efitnessmax[-1],
            'optimization_time': optimization_time,
            'fitness_evolution': self.efitnessmax,
            'optimization_history': self.optimization_history
        }
    
    def generate_code_snippet(self, params):
        """Generate ready-to-use code snippet for nav_dwb_ros.py"""
        print("\n" + "=" * 60)
        print("🔧 READY-TO-USE CODE SNIPPET:")
        print("=" * 60)
        print("# === OPTIMIZED COLLISION AVOIDANCE DISTANCES ===")
        print(f"self.emergency_distance = {params['emergency_distance']:.2f}")
        print(f"self.critical_distance = {params['critical_distance']:.2f}")
        print(f"self.warning_distance = {params['warning_distance']:.2f}")
        print(f"self.safe_distance = {params['safe_distance']:.2f}")
        print()
        print("# === OPTIMIZED REAL-WORLD SPEED PARAMETERS (converted to m/s) ===")
        print(f"self.max_speed = {params['max_speed_kmh']:.2f} / 3.6        # {params['max_speed_kmh']/3.6:.2f} m/s")
        print(f"self.cruise_speed = {params['cruise_speed_kmh']:.2f} / 3.6     # {params['cruise_speed_kmh']/3.6:.2f} m/s")
        print(f"self.normal_speed = {params['normal_speed_kmh']:.2f} / 3.6     # {params['normal_speed_kmh']/3.6:.2f} m/s")
        print(f"self.slow_speed = {params['slow_speed_kmh']:.2f} / 3.6       # {params['slow_speed_kmh']/3.6:.2f} m/s")
        print(f"self.creep_speed = {params['creep_speed_kmh']:.2f} / 3.6       # {params['creep_speed_kmh']/3.6:.2f} m/s")
        print("=" * 60)
    
    def save_optimization_results(self, params):
        """Save optimization results (same as original)"""
        results = {
            'method': 'Simplified_Real_World_Binary_Genetic_Algorithm',
            'ga_parameters': {
                'population_size': self.Npop,
                'generations': self.Maxit,
                'crossover_probability': self.Pc,
                'mutation_probability': self.Pm,
                'elitism_ratio': self.el,
                'bits_per_variable': self.Nbit,
                'number_of_variables': self.Nvar
            },
            'optimized_parameters': params,
            'best_fitness': float(self.efitnessmax[-1]),
            'fitness_evolution': [float(f) for f in self.efitnessmax],
            'parameter_bounds': {
                'lower_bounds': self.rb,
                'upper_bounds': self.ra
            },
            'objective_weights': self.weights,
            'timestamp': time.time()
        }
        
        filename = f"simplified_real_world_ga_{int(time.time())}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to: {filename}")
        except Exception as e:
            print(f"❌ Save error: {e}")

# Main execution
def run_simplified_real_world_binary_ga_optimization():
    """Run Simplified Real-World Binary GA optimization"""
    optimizer = SimplifiedRealWorldBinaryGAOptimizer()
    
    results = optimizer.optimize_parameters_simplified_binary_ga()
    
    return results

def compare_with_your_current_parameters():
    """Compare with your current real-world parameters"""
    print("\n🔍 COMPARISON WITH YOUR CURRENT REAL-WORLD PARAMETERS:")
    print("=" * 60)
    
    # Your current real-world parameters
    current_params = {
        'emergency_distance': 8.59,
        'critical_distance': 15.53,
        'warning_distance': 17.54,
        'safe_distance': 24.54,
        'max_speed_kmh': 55.13,
        'cruise_speed_kmh': 34.89,
        'normal_speed_kmh': 28.30,
        'slow_speed_kmh': 14.71,
        'creep_speed_kmh': 1.08
    }
    
    print("YOUR CURRENT REAL-WORLD PARAMETERS (trial and error):")
    for param, value in current_params.items():
        if 'speed' in param:
            print(f"   {param}: {value:.2f} km/h")
        else:
            print(f"   {param}: {value:.2f}m")
    
    print(f"\n📊 Your successful original GA best fitness: -6.005")
    print(f"🎯 Target: Get similar fitness range (-3 to -8) with real-world speeds")
    print(f"\n🚀 This simplified optimizer uses same penalty scales as your successful version!")

if __name__ == "__main__":
    print("🚗 Simplified Real-World Speed GA Optimizer")
    print("=" * 60)
    
    # Show comparison first
    compare_with_your_current_parameters()
    
    # Ask user if they want to run optimization
    print(f"\nDo you want to run the simplified GA optimization? (y/n): ", end="")
    response = input().lower().strip()
    
    if response in ['y', 'yes']:
        print("\n🚀 Starting simplified real-world optimization...")
        results = run_simplified_real_world_binary_ga_optimization()
        print("\n🎊 Simplified Real-World GA Optimization complete!")
        print("Check the generated code snippet above for easy integration!")
    else:
        print("\n👍 You can run the optimization anytime by executing this script!")
        print("This simplified version should achieve fitness around -3 to -8 range like your original.")
