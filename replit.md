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
