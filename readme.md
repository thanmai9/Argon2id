# Performance Analysis and Parameter Optimization of Argon2id for Secure Password Hashing

## Project Overview
A comprehensive Flask + MongoDB web application for analyzing and optimizing password hashing algorithms, specifically Argon2id, with comparative benchmarking against bcrypt, scrypt, and PBKDF2.

## Features
- **User Authentication**: Secure registration and login using Argon2id hashing
- **Algorithm Benchmarking**: Compare performance of Argon2id, bcrypt, scrypt, and PBKDF2
- **Parameter Optimization**: Experiment with various Argon2id configurations (time_cost, memory_cost, parallelism)
- **Performance Monitoring**: Real-time CPU and memory usage tracking
- **Data Visualization**: Interactive charts and heatmaps using Matplotlib/Plotly
- **Data Export**: CSV and PNG export of benchmark results

## Project Structure
```
/
├── app.py                 # Flask backend with all API endpoints
├── templates/             # HTML templates
│   ├── base.html         # Base template with navigation
│   ├── home.html         # Landing page with Argon2id introduction
│   ├── register.html     # User registration form
│   ├── login.html        # User login form
│   └── benchmark.html    # Benchmarking interface
├── static/
│   ├── css/
│   │   └── style.css     # Custom styling
│   └── js/
│       └── main.js       # Frontend JavaScript
├── results/              # CSV and PNG exports
└── requirements.txt      # Python dependencies
```

## Tech Stack
- **Backend**: Python 3.11, Flask
- **Database**: MongoDB (authdb)
- **Frontend**: HTML, CSS, Bootstrap 5, JavaScript
- **Visualization**: Matplotlib, Plotly
- **Monitoring**: psutil

## Database Schema
### Collections
- **users**: {username, hash, algorithm, params, created_at}
- **benchmarks**: {algorithm, time_cost, memory_cost, parallelism, num_passwords, total_ms, time_per_hash_ms, mem_MB, cpu_percent, timestamp}

## Recent Changes
- 2025-10-26: Initial project setup with complete authentication and benchmarking system

## User Preferences
None specified yet.

## Architecture
- RESTful API design with Flask
- MongoDB for persistent storage
- Session-based authentication
- Asynchronous benchmarking with performance monitoring
- Responsive Bootstrap UI
  
## Output
<img width="1639" height="865" alt="Screenshot 2026-02-09 140054" src="https://github.com/user-attachments/assets/b39d5d9d-37a7-478c-9416-a8e35fb981ee" />
<img width="1566" height="787" alt="Screenshot 2026-02-09 140103" src="https://github.com/user-attachments/assets/02aa9d67-3ccc-4df1-bdf9-287c1c0e5bbd" />
<img width="1010" height="867" alt="Screenshot 2026-02-09 140112" src="https://github.com/user-attachments/assets/e47b202e-57a8-4ba9-b646-d5fd97e6091b" />



