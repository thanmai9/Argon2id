from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from pymongo import MongoClient
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
import bcrypt
import hashlib
import time
import psutil
import csv
import os
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'dev-secret-key-change-in-production')

try:
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client.authdb
    users_collection = db.users
    benchmarks_collection = db.benchmarks
    print("✓ MongoDB connected successfully")
except Exception as e:
    print(f"⚠ MongoDB not connected: {e}")
    db = None
    users_collection = None
    benchmarks_collection = None

ph = PasswordHasher(
    time_cost=2,
    memory_cost=65536,
    parallelism=2,
    hash_len=32,
    salt_len=16
)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'}), 400
        
        if users_collection is None:
            return jsonify({'success': False, 'message': 'Database not available'}), 503
        
        if users_collection.find_one({'username': username}):
            return jsonify({'success': False, 'message': 'Username already exists'}), 400
        
        try:
            password_hash = ph.hash(password)
            
            user_doc = {
                'username': username,
                'hash': password_hash,
                'algorithm': 'argon2id',
                'params': {
                    'time_cost': 2,
                    'memory_cost': 65536,
                    'parallelism': 2
                },
                'created_at': datetime.utcnow()
            }
            
            users_collection.insert_one(user_doc)
            
            return jsonify({
                'success': True,
                'message': 'Registration successful! You can now log in.'
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Registration failed: {str(e)}'}), 500
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'}), 400
        
        if users_collection is None:
            return jsonify({'success': False, 'message': 'Database not available'}), 503
        
        user = users_collection.find_one({'username': username})
        
        if not user:
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
        
        try:
            ph.verify(user['hash'], password)
            
            if ph.check_needs_rehash(user['hash']):
                new_hash = ph.hash(password)
                users_collection.update_one(
                    {'username': username},
                    {'$set': {'hash': new_hash}}
                )
            
            session['username'] = username
            session['logged_in'] = True
            
            return jsonify({
                'success': True,
                'message': 'Login successful!',
                'username': username
            })
        except VerifyMismatchError:
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
        except Exception as e:
            return jsonify({'success': False, 'message': f'Login failed: {str(e)}'}), 500
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/benchmark')
def benchmark_page():
    return render_template('benchmark.html')

@app.route('/api/benchmark', methods=['POST'])
def run_benchmark():
    data = request.json
    
    time_costs = data.get('time_costs', [1, 2, 4])
    memory_costs = data.get('memory_costs', [16384, 65536, 262144])
    parallelisms = data.get('parallelisms', [1, 2, 4])
    password_length = data.get('password_length', 12)
    num_iterations = data.get('num_iterations', 5)
    
    test_password = 'x' * password_length
    
    results = []
    
    print(f"Starting benchmark with {len(time_costs) * len(memory_costs) * len(parallelisms)} Argon2id configurations...")
    
    for t in time_costs:
        for m in memory_costs:
            for p in parallelisms:
                try:
                    ph_test = PasswordHasher(
                        time_cost=t,
                        memory_cost=m,
                        parallelism=p
                    )
                    
                    times = []
                    cpu_usage = []
                    mem_usage = []
                    
                    for i in range(num_iterations):
                        process = psutil.Process()
                        
                        cpu_before = psutil.cpu_percent(interval=0.1)
                        mem_before = process.memory_info().rss / 1024 / 1024
                        
                        start = time.perf_counter()
                        _ = ph_test.hash(test_password)
                        end = time.perf_counter()
                        
                        cpu_after = psutil.cpu_percent(interval=0.1)
                        mem_after = process.memory_info().rss / 1024 / 1024
                        
                        times.append((end - start) * 1000)
                        cpu_usage.append(cpu_after)
                        mem_usage.append(mem_after - mem_before)
                    
                    avg_time = np.mean(times)
                    avg_cpu = np.mean(cpu_usage)
                    avg_mem = np.mean(mem_usage)
                    
                    result = {
                        'algorithm': 'argon2id',
                        'time_cost': t,
                        'memory_cost': m,
                        'parallelism': p,
                        'num_passwords': num_iterations,
                        'total_ms': sum(times),
                        'time_per_hash_ms': avg_time,
                        'mem_MB': avg_mem,
                        'cpu_percent': avg_cpu,
                        'timestamp': datetime.utcnow()
                    }
                    
                    results.append(result)
                    
                    if benchmarks_collection is not None:
                        benchmarks_collection.insert_one(result.copy())
                    
                except Exception as e:
                    print(f"Error with t={t}, m={m}, p={p}: {e}")
    
    print("Running comparison benchmarks for other algorithms...")
    
    for algorithm in ['bcrypt', 'scrypt', 'pbkdf2']:
        try:
            times = []
            cpu_usage = []
            mem_usage = []
            
            for i in range(num_iterations):
                process = psutil.Process()
                
                cpu_before = psutil.cpu_percent(interval=0.1)
                mem_before = process.memory_info().rss / 1024 / 1024
                
                start = time.perf_counter()
                
                if algorithm == 'bcrypt':
                    _ = bcrypt.hashpw(test_password.encode(), bcrypt.gensalt(rounds=12))
                elif algorithm == 'scrypt':
                    _ = hashlib.scrypt(test_password.encode(), salt=os.urandom(16), n=16384, r=8, p=1, dklen=32)
                elif algorithm == 'pbkdf2':
                    _ = hashlib.pbkdf2_hmac('sha256', test_password.encode(), os.urandom(16), 100000)
                
                end = time.perf_counter()
                
                cpu_after = psutil.cpu_percent(interval=0.1)
                mem_after = process.memory_info().rss / 1024 / 1024
                
                times.append((end - start) * 1000)
                cpu_usage.append(cpu_after)
                mem_usage.append(mem_after - mem_before)
            
            avg_time = np.mean(times)
            avg_cpu = np.mean(cpu_usage)
            avg_mem = np.mean(mem_usage)
            
            result = {
                'algorithm': algorithm,
                'time_cost': None,
                'memory_cost': None,
                'parallelism': None,
                'num_passwords': num_iterations,
                'total_ms': sum(times),
                'time_per_hash_ms': avg_time,
                'mem_MB': avg_mem,
                'cpu_percent': avg_cpu,
                'timestamp': datetime.utcnow()
            }
            
            results.append(result)
            
            if benchmarks_collection is not None:
                benchmarks_collection.insert_one(result.copy())
                
        except Exception as e:
            print(f"Error with {algorithm}: {e}")
    
    csv_file = os.path.join('results', 'benchmark_results.csv')
    fieldnames = ['algorithm', 'time_cost', 'memory_cost', 'parallelism', 'num_passwords', 'total_ms', 'time_per_hash_ms', 'mem_MB', 'cpu_percent']
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, '') for k in fieldnames}
            writer.writerow(row)
    
    print(f"Results saved to {csv_file}")
    
    visualizations = generate_visualizations(results)
    
    optimal_config = find_optimal_config(results)
    
    results_json = []
    for r in results:
        r_copy = r.copy()
        if 'timestamp' in r_copy:
            r_copy['timestamp'] = r_copy['timestamp'].isoformat()
        results_json.append(r_copy)
    
    return jsonify({
        'success': True,
        'results': results_json,
        'optimal_config': optimal_config,
        'visualizations': visualizations,
        'csv_file': csv_file
    })

def generate_visualizations(results):
    df = pd.DataFrame(results)
    
    argon2_df = df[df['algorithm'] == 'argon2id'].copy()
    
    viz_files = []
    
    if len(argon2_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Argon2id Performance Analysis', fontsize=16, fontweight='bold')
        
        memory_grouped = argon2_df.groupby('memory_cost')['time_per_hash_ms'].mean()
        axes[0, 0].plot(memory_grouped.index / 1024, memory_grouped.values, marker='o', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Memory Cost (MB)', fontsize=11)
        axes[0, 0].set_ylabel('Avg Time per Hash (ms)', fontsize=11)
        axes[0, 0].set_title('Hash Time vs Memory Cost', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        parallelism_grouped = argon2_df.groupby('parallelism')['time_per_hash_ms'].mean()
        throughput = 1000 / parallelism_grouped.values
        axes[0, 1].bar(parallelism_grouped.index.astype(str), throughput, color='skyblue', edgecolor='navy')
        axes[0, 1].set_xlabel('Parallelism', fontsize=11)
        axes[0, 1].set_ylabel('Throughput (hashes/sec)', fontsize=11)
        axes[0, 1].set_title('Throughput vs Parallelism', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        pivot = argon2_df.pivot_table(values='time_per_hash_ms', index='memory_cost', columns='time_cost', aggfunc='mean')
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1, 0], cbar_kws={'label': 'Time (ms)'})
        axes[1, 0].set_xlabel('Time Cost', fontsize=11)
        axes[1, 0].set_ylabel('Memory Cost (KB)', fontsize=11)
        axes[1, 0].set_title('Parameter Heatmap: Time Cost vs Memory Cost', fontsize=12, fontweight='bold')
        
        algo_df = df.groupby('algorithm')['time_per_hash_ms'].mean().sort_values()
        colors = ['green' if x == algo_df.min() else 'orange' if x == algo_df.max() else 'skyblue' for x in algo_df.values]
        axes[1, 1].barh(algo_df.index, algo_df.values, color=colors, edgecolor='black')
        axes[1, 1].set_xlabel('Avg Time per Hash (ms)', fontsize=11)
        axes[1, 1].set_ylabel('Algorithm', fontsize=11)
        axes[1, 1].set_title('Algorithm Comparison', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        matplotlib_file = os.path.join('results', 'benchmark_analysis.png')
        plt.savefig(matplotlib_file, dpi=150, bbox_inches='tight')
        plt.close()
        viz_files.append(matplotlib_file)
        print(f"Matplotlib visualization saved to {matplotlib_file}")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Hash Time vs Memory Cost', 'CPU Usage by Algorithm', 
                        'Memory Usage Comparison', 'Time Cost Impact'),
        specs=[[{'type': 'scatter'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'scatter'}]]
    )
    
    if len(argon2_df) > 0:
        for tc in argon2_df['time_cost'].unique():
            subset = argon2_df[argon2_df['time_cost'] == tc]
            fig.add_trace(
                go.Scatter(x=subset['memory_cost'] / 1024, y=subset['time_per_hash_ms'],
                          mode='lines+markers', name=f'time_cost={tc}'),
                row=1, col=1
            )
    
    algo_summary = df.groupby('algorithm').agg({'cpu_percent': 'mean'}).reset_index()
    fig.add_trace(
        go.Bar(x=algo_summary['algorithm'], y=algo_summary['cpu_percent'], 
               marker_color='lightcoral'),
        row=1, col=2
    )
    
    algo_mem = df.groupby('algorithm').agg({'mem_MB': 'mean'}).reset_index()
    fig.add_trace(
        go.Bar(x=algo_mem['algorithm'], y=algo_mem['mem_MB'], 
               marker_color='lightblue'),
        row=2, col=1
    )
    
    if len(argon2_df) > 0:
        time_impact = argon2_df.groupby('time_cost')['time_per_hash_ms'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=time_impact['time_cost'], y=time_impact['time_per_hash_ms'],
                      mode='lines+markers', marker=dict(size=12, color='purple'), line=dict(width=3)),
            row=2, col=2
        )
    
    fig.update_xaxes(title_text="Memory Cost (MB)", row=1, col=1)
    fig.update_yaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_xaxes(title_text="Algorithm", row=1, col=2)
    fig.update_yaxes(title_text="CPU %", row=1, col=2)
    fig.update_xaxes(title_text="Algorithm", row=2, col=1)
    fig.update_yaxes(title_text="Memory (MB)", row=2, col=1)
    fig.update_xaxes(title_text="Time Cost", row=2, col=2)
    fig.update_yaxes(title_text="Time (ms)", row=2, col=2)
    
    fig.update_layout(height=800, showlegend=True, title_text="Interactive Performance Dashboard")
    
    plotly_file = os.path.join('results', 'interactive_dashboard.html')
    fig.write_html(plotly_file)
    viz_files.append(plotly_file)
    print(f"Plotly visualization saved to {plotly_file}")
    
    return viz_files

def find_optimal_config(results):
    argon2_results = [r for r in results if r['algorithm'] == 'argon2id']
    
    if not argon2_results:
        return None
    
    for r in argon2_results:
        security_score = (r['time_cost'] * 2) + (r['memory_cost'] / 10000) + (r['parallelism'] * 1.5)
        performance_score = 1000 / r['time_per_hash_ms']
        r['balanced_score'] = (security_score * 0.6) + (performance_score * 0.4)
    
    optimal = max(argon2_results, key=lambda x: x['balanced_score'])
    
    return {
        'algorithm': 'Argon2id',
        'time_cost': optimal['time_cost'],
        'memory_cost': optimal['memory_cost'],
        'memory_cost_kb': optimal['memory_cost'],
        'memory_cost_mb': optimal['memory_cost'] / 1024,
        'parallelism': optimal['parallelism'],
        'avg_hash_time_ms': round(optimal['time_per_hash_ms'], 2),
        'cpu_percent': round(optimal['cpu_percent'], 2),
        'mem_mb': round(optimal['mem_MB'], 2)
    }

@app.route('/api/results')
def get_results():
    if benchmarks_collection is None:
        return jsonify({'success': False, 'message': 'Database not available'}), 503
    
    results = list(benchmarks_collection.find().sort('timestamp', -1).limit(100))
    
    for r in results:
        r['_id'] = str(r['_id'])
        if 'timestamp' in r:
            r['timestamp'] = r['timestamp'].isoformat()
    
    return jsonify({'success': True, 'results': results})

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join('results', filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'success': False, 'message': 'File not found'}), 404

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
