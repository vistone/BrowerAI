// Initialization script for BrowerAI integrated pipeline
(function() {
    'use strict';
    
    // Initialize components
    function initComponents() {
        const components = document.querySelectorAll('[data-component]');
        components.forEach(component => {
            const name = component.dataset.component;
            console.log(`Initialized component: ${name}`);
        });
    }
    
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initComponents);
    } else {
        initComponents();
    }
})();
