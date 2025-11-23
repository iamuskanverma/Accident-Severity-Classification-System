document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const uploadSection = document.getElementById('upload-section');
    const resultsSection = document.getElementById('results-section');
    const previewImg = document.getElementById('preview-img');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceVal = document.getElementById('confidence-val');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingText = loadingOverlay.querySelector('.loading-text');
    const loadingSubtext = loadingOverlay.querySelector('.loading-subtext');
    const imageWrapper = document.getElementById('image-wrapper');
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = themeToggle.querySelector('i');

    // Theme Management
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);

    themeToggle.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeIcon(newTheme);
    });

    function updateThemeIcon(theme) {
        if (theme === 'dark') {
            themeIcon.classList.remove('fa-moon');
            themeIcon.classList.add('fa-sun');
        } else {
            themeIcon.classList.remove('fa-sun');
            themeIcon.classList.add('fa-moon');
        }
    }

    // Handle file selection
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            const reader = new FileReader();

            reader.onload = (e) => {
                startAnalysis(e.target.result);
            }

            reader.readAsDataURL(file);
        }
    });

    // Drag and Drop support
    uploadSection.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadSection.classList.add('drag-active');
    });

    uploadSection.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadSection.classList.remove('drag-active');
    });

    uploadSection.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadSection.classList.remove('drag-active');
        
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            fileInput.files = e.dataTransfer.files;
            const event = new Event('change');
            fileInput.dispatchEvent(event);
        }
    });

    function startAnalysis(imageSrc) {
        // 1. Show Loading Overlay
        loadingOverlay.classList.add('active');
        loadingText.textContent = "Uploading Image...";
        loadingSubtext.textContent = "Encrypting and securing data";

        setTimeout(() => {
            // 2. Simulate Processing Steps
            loadingText.textContent = "Analyzing Damage...";
            loadingSubtext.textContent = "Running neural network models";
            
            setTimeout(() => {
                loadingText.textContent = "Classifying Severity...";
                loadingSubtext.textContent = "Comparing with accident database";
                
                setTimeout(() => {
                    // 3. Show Results
                    loadingOverlay.classList.remove('active');
                    showResults(imageSrc);
                }, 1500);
            }, 1500);
        }, 1000);
    }

    function showResults(imageSrc) {
        previewImg.src = imageSrc;
        uploadSection.style.display = 'none';
        resultsSection.style.display = 'block';
        
        // Start scanning animation
        imageWrapper.classList.add('scanning');
        
        // Animate Confidence Counter
        let start = 0;
        const end = 88.5;
        const duration = 1500;
        const startTime = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Ease out quart
            const ease = 1 - Math.pow(1 - progress, 4);
            
            const currentVal = start + (end - start) * ease;
            confidenceVal.textContent = currentVal.toFixed(1) + '%';
            confidenceBar.style.width = currentVal + '%';

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }

        requestAnimationFrame(update);
        
        // Stop scanning after a few seconds
        setTimeout(() => {
            imageWrapper.classList.remove('scanning');
        }, 3000);
    }
});

function resetApp() {
    location.reload();
}
