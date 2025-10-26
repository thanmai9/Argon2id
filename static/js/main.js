document.addEventListener('DOMContentLoaded', function() {
    console.log('Argon2id Performance Analysis Platform Loaded');
    
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(alert => {
        setTimeout(() => {
            const closeBtn = alert.querySelector('.btn-close');
            if (closeBtn) {
                closeBtn.click();
            }
        }, 5000);
    });
});

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

function formatDuration(ms) {
    if (ms < 1000) return ms.toFixed(2) + ' ms';
    return (ms / 1000).toFixed(2) + ' s';
}

async function fetchBenchmarkHistory() {
    try {
        const response = await fetch('/api/results');
        const data = await response.json();
        
        if (data.success) {
            return data.results;
        }
    } catch (error) {
        console.error('Error fetching benchmark history:', error);
    }
    return [];
}

function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3`;
    alertDiv.style.zIndex = '9999';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

document.querySelectorAll('a[download]').forEach(link => {
    link.addEventListener('click', function(e) {
        const filename = this.getAttribute('href').split('/').pop();
        showNotification(`Downloading ${filename}...`, 'info');
    });
});
